import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.technical import build_technical_features
from src.features.microstructure import build_microstructure_features
from src.features.funding_rates import build_funding_features
from src.features.fracdiff import load_d_values, apply_fracdiff_transform
from src.features.cross_sectional import apply_cross_sectional_ranks
from src.features.regime import (
    load_hmm_artifacts,
    get_regime_probs,
    apply_adx_fallback,
    fit_bocpd,
    get_changepoint_distance,
)
from src.models.model_versioning import get_latest_model
from src.utils.logger import get_logger

logger = get_logger("live_features")

# Columns eligible for fracdiff — same list as feature_pipeline._FRACDIFF_COLS
_FRACDIFF_COLS = [
    "close_5_mean", "close_10_mean", "close_20_mean",
    "close_50_mean", "close_100_mean", "close_200_mean",
    "obv", "vwap_20",
]

# HTF interval → Binance kline interval string
_HTF_INTERVALS = {"1h": "1h", "4h": "4h", "1d": "1d"}

# ffill limits per HTF (bars in 15m units)
_HTF_FFILL = {"1h": 4, "4h": 16, "1d": 96}


def get_lookback_bars_needed(cfg) -> int:
    # Cap at 1500 — Binance FAPI hard limit for klines endpoint
    return min(int(cfg.features.warmup_bars) + 200, 1500)


def _build_hmm_input(klines_15m: pd.DataFrame) -> pd.DataFrame:
    # Replicate feature_pipeline._build_hmm_features_df
    log_ret = np.log(klines_15m["close"] / klines_15m["close"].shift(1))
    r2 = log_ret ** 2
    realized_vol = r2.rolling(20, min_periods=20).mean().apply(np.sqrt)
    vol_mean = klines_15m["volume"].rolling(20, min_periods=20).mean()
    vol_std = klines_15m["volume"].rolling(20, min_periods=20).std()
    volume_zscore = (klines_15m["volume"] - vol_mean) / (vol_std + 1e-9)
    hmm_df = pd.DataFrame({
        "log_return": log_ret,
        "realized_vol_20": realized_vol,
        "volume_zscore": volume_zscore,
    }, index=klines_15m.index)
    return hmm_df.dropna()


def _merge_htf(base_15m: pd.DataFrame, htf_df: pd.DataFrame, tf: str) -> pd.DataFrame:
    ffill_limit = _HTF_FFILL.get(tf, 4)
    htf_reindexed = htf_df.reindex(base_15m.index, method="ffill", limit=ffill_limit)
    htf_reindexed.columns = [f"{c}_{tf}" for c in htf_reindexed.columns]
    return pd.concat([base_15m, htf_reindexed], axis=1)


def _load_macro_onchain(cfg) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    # Load last-known macro and onchain values from stage-1 processed panels (ffill only).
    # These are slow-moving; using the last saved row is acceptable for live inference.
    processed_dir = Path(cfg.data.processed_dir)
    macro_panel, onchain_panel = None, None
    macro_path = processed_dir / "macro_panel_15m.parquet"
    onchain_path = processed_dir / "onchain_panel_15m.parquet"
    if macro_path.exists():
        try:
            macro_panel = pd.read_parquet(macro_path)
        except Exception as exc:
            logger.warning(f"Could not load macro_panel: {exc}")
    if onchain_path.exists():
        try:
            onchain_panel = pd.read_parquet(onchain_path)
        except Exception as exc:
            logger.warning(f"Could not load onchain_panel: {exc}")
    return macro_panel, onchain_panel


def compute_live_features(
    symbol: str,
    cfg,
    klines_15m: pd.DataFrame,
    klines_1h: pd.DataFrame | None = None,
    klines_4h: pd.DataFrame | None = None,
    klines_1d: pd.DataFrame | None = None,
    btc_klines_15m: pd.DataFrame | None = None,
    client=None,  # BinanceClient — if provided, real funding rates injected
) -> pd.Series:
    """
    Build the full inference-time feature vector for a symbol.

    The feature set mirrors feature_pipeline.build_features_for_symbol:
    technical (15m + HTF) → microstructure → funding → HMM regime →
    BOCPD changepoint → fracdiff → macro/onchain → global shift(1) → last row.

    The global shift(1) is applied to match training, so the returned feature
    vector corresponds to data from the penultimate bar (t-1), which is what
    the model saw during training for prediction at bar t.
    """
    if klines_15m.index.tz is None:
        raise ValueError(f"{symbol}: klines_15m index must be UTC-aware")

    checkpoints_dir = Path(cfg.data.checkpoints_dir)

    # 1. Technical features — 15m
    all_features = build_technical_features(klines_15m, cfg)

    # 2. HTF technical features (1h, 4h, 1d) — ffill onto 15m grid
    for tf, df_htf in [("1h", klines_1h), ("4h", klines_4h), ("1d", klines_1d)]:
        if df_htf is not None and len(df_htf) > 0:
            try:
                tech_htf = build_technical_features(df_htf, cfg)
                all_features = _merge_htf(all_features, tech_htf, tf)
            except Exception as exc:
                logger.warning(f"{symbol}: HTF {tf} feature build failed — {exc}")

    # 3. Microstructure features
    try:
        micro = build_microstructure_features(klines_15m, cfg)
        all_features = pd.concat([all_features, micro], axis=1)
    except Exception as exc:
        logger.warning(f"{symbol}: microstructure features failed — {exc}")

    # 4. Funding rate features
    try:
        btc_df_for_funding = btc_klines_15m if (symbol != "BTCUSDT" and btc_klines_15m is not None) else None
        # Inject real funding rate from Binance API if client available — overrides proxy
        klines_for_funding = klines_15m.copy()
        if client is not None:
            try:
                fr_history = client.get_funding_rate_history(symbol, limit=500)
                if not fr_history.empty:
                    # Funding is settled every 8h — forward-fill to 15m bars
                    fr_aligned = fr_history["fundingRate"].reindex(klines_for_funding.index, method="ffill")
                    klines_for_funding["real_funding_rate"] = fr_aligned
                    logger.debug(f"{symbol}: injected real funding rates ({len(fr_history)} records)")
            except Exception as _fr_exc:
                logger.debug(f"{symbol}: real funding rate fetch failed — using proxy: {_fr_exc}")
        funding = build_funding_features(klines_for_funding, btc_df_for_funding, cfg)
        all_features = pd.concat([all_features, funding], axis=1)
    except Exception as exc:
        logger.warning(f"{symbol}: funding features failed — {exc}")

    # 5. HMM regime probabilities — load saved model, predict on full lookback window
    hmm_dir = checkpoints_dir / "hmm" / symbol
    hmm_input = _build_hmm_input(klines_15m)
    if hmm_dir.exists() and (hmm_dir / "hmm_model.pkl").exists():
        try:
            hmm_model, scaler, _state_labels = load_hmm_artifacts(hmm_dir)
            regime_probs = get_regime_probs(hmm_model, hmm_input, scaler)
            regime_probs_aligned = regime_probs.reindex(all_features.index)
            all_features = pd.concat([all_features, regime_probs_aligned], axis=1)
        except Exception as exc:
            logger.warning(f"{symbol}: HMM regime features failed — {exc}")
    else:
        logger.warning(f"{symbol}: HMM artifacts not found at {hmm_dir} — regime_prob_* features will be NaN")

    # 6. ADX trend fallback flag (uses adx column from technical features)
    if "adx" in all_features.columns:
        try:
            adx_flag = apply_adx_fallback(all_features["adx"])
            all_features = pd.concat([all_features, adx_flag.to_frame()], axis=1)
        except Exception as exc:
            logger.warning(f"{symbol}: ADX fallback failed — {exc}")

    # 7. BOCPD changepoint distance — re-fit on lookback log_return (O(n log n), fast enough per bar)
    if "log_return" in hmm_input.columns:
        try:
            bocpd_model = fit_bocpd(hmm_input["log_return"], cfg)
            cp_dist = get_changepoint_distance(hmm_input["log_return"], bocpd_model)
            cp_dist_aligned = cp_dist.reindex(all_features.index)
            all_features = pd.concat([all_features, cp_dist_aligned.to_frame()], axis=1)
        except Exception as exc:
            logger.warning(f"{symbol}: BOCPD changepoint features failed — {exc}")

    # 8. Fracdiff transforms — load d-values cached from training, apply to full window
    fracdiff_cache = checkpoints_dir / "fracdiff"
    price_vol_cols = [c for c in _FRACDIFF_COLS if c in all_features.columns]
    if price_vol_cols:
        d_values = load_d_values(symbol, "15m", fracdiff_cache)
        if d_values:
            try:
                all_features = apply_fracdiff_transform(all_features, d_values)
            except Exception as exc:
                logger.warning(f"{symbol}: fracdiff transform failed — {exc}")
        else:
            logger.warning(f"{symbol}: no fracdiff d-values cached — price/vol columns not differenced")

    # 9. Macro and onchain panels — load last-known values from stage-1 processed files
    macro_panel, onchain_panel = _load_macro_onchain(cfg)
    if macro_panel is not None and len(macro_panel) > 0:
        try:
            macro_aligned = macro_panel.reindex(all_features.index, method="ffill")
            all_features = pd.concat([all_features, macro_aligned], axis=1)
        except Exception as exc:
            logger.warning(f"{symbol}: macro panel merge failed — {exc}")
    if onchain_panel is not None and len(onchain_panel) > 0:
        try:
            onchain_aligned = onchain_panel.reindex(all_features.index, method="ffill")
            all_features = pd.concat([all_features, onchain_aligned], axis=1)
        except Exception as exc:
            logger.warning(f"{symbol}: onchain panel merge failed — {exc}")

    # 10. Remove duplicate columns (can arise from HTF merge or multi-module concat)
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]

    # 10b. Apply cross-sectional rank transforms — uses pre-fitted stats from training
    cs_stats_path = checkpoints_dir / "cross_sectional_stats.pkl"
    if cs_stats_path.exists():
        feature_cols_for_rank = all_features.select_dtypes(include=[np.float32, np.float64, float]).columns.tolist()
        all_features = apply_cross_sectional_ranks(all_features, cs_stats_path, feature_cols_for_rank)
    else:
        logger.debug(f"{symbol}: cross_sectional_stats not found at {cs_stats_path} — rank features will be NaN")

    # 11. Apply global shift(1) to match training pipeline
    # In feature_pipeline.py all feature columns are shifted by 1 bar so that feature[t]
    # is built from data available at t-1. We replicate this here so the model sees
    # the same relative data ordering it was trained on.
    all_features = all_features.shift(1)

    # Take the last valid row — after shift this contains features derived from the
    # second-to-last bar, consistent with how training features are constructed.
    last_row = all_features.iloc[[-1]]

    # 12. Load imputer and scaler (fit on train split only — no leakage)
    imputer_path = checkpoints_dir / "imputers" / f"imputer_{symbol}_15m.pkl"
    if not imputer_path.exists():
        raise FileNotFoundError(f"Imputer not found: {imputer_path}")
    with open(imputer_path, "rb") as f:
        imputer = pickle.load(f)

    scaler_path = checkpoints_dir / "imputers" / f"scaler_{symbol}_15m.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # 13. Subset to the exact feature names the model was trained on
    registry_entry = get_latest_model(symbol, "15m", model_type="primary")
    if registry_entry is None:
        raise RuntimeError(f"No registered primary model for {symbol} 15m")

    selected_features = registry_entry["feature_names"]

    missing_cols = [c for c in selected_features if c not in last_row.columns]
    if missing_cols:
        logger.warning(
            f"{symbol}: {len(missing_cols)}/{len(selected_features)} feature cols missing "
            f"from live compute — filling NaN. Examples: {missing_cols[:5]}"
        )
        fill_df = pd.DataFrame(np.nan, index=last_row.index, columns=missing_cols)
        last_row = pd.concat([last_row, fill_df], axis=1)

    last_row = last_row[selected_features]

    # 14. Transform with imputer then scaler (transform-only, never fit)
    transformed = imputer.transform(last_row.values)
    transformed = scaler.transform(transformed)

    result = pd.Series(transformed[0], index=selected_features, name=last_row.index[0])
    logger.debug(f"{symbol}: live features computed — {len(result)} features, {len(missing_cols)} filled NaN")
    return result
