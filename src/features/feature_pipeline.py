import numpy as np
import pandas as pd
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.logger import get_logger
from src.utils.time_utils import assert_no_future_leakage
from src.features.technical import build_technical_features
from src.features.microstructure import build_microstructure_features
from src.features.funding_rates import build_funding_features
from src.features.market_positioning import build_market_positioning_features
from src.features.fracdiff import fit_and_save_d_values, load_d_values, apply_fracdiff_transform
from src.features.regime import (
    fit_hmm, get_regime_probs, label_regime_states,
    apply_adx_fallback, fit_bocpd, get_changepoint_distance,
    save_hmm_artifacts, load_hmm_artifacts,
)

logger = get_logger("feature_pipeline")

# Columns to apply fracdiff — must match columns actually produced by build_technical_features.
# Rolling-mean closes (close_N_mean) are never produced; the pipeline uses log-return rolling
# stats (log_ret_N_mean) which are already stationary. Only OBV and VWAP are price-level
# accumulators that benefit from fractional differencing.
_FRACDIFF_COLS = [
    "obv", "vwap_20",
]


def _merge_htf(base_15m: pd.DataFrame, htf_df: pd.DataFrame, tf: str, cfg) -> pd.DataFrame:
    if htf_df is None:
        return base_15m
    ffill_limit = cfg.features.htf_ffill_limits.get(tf, 4)
    # Reindex HTF to 15m index, ffill only (never bfill)
    htf_reindexed = htf_df.reindex(base_15m.index, method="ffill", limit=ffill_limit)
    # Rename columns to avoid collisions
    htf_reindexed.columns = [f"{c}_{tf}" for c in htf_reindexed.columns]
    return pd.concat([base_15m, htf_reindexed], axis=1)


def _build_hmm_features_df(df: pd.DataFrame) -> pd.DataFrame:
    # Features fed to HMM: log_return, realized_vol_20, volume_zscore
    log_ret = np.log(df["close"] / df["close"].shift(1))
    r2 = log_ret ** 2
    realized_vol = np.sqrt(r2.rolling(20, min_periods=20).mean())
    vol_mean = df["volume"].rolling(20, min_periods=20).mean()
    vol_std = df["volume"].rolling(20, min_periods=20).std()
    volume_zscore = (df["volume"] - vol_mean) / (vol_std + 1e-9)
    hmm_df = pd.DataFrame({
        "log_return": log_ret,
        "realized_vol_20": realized_vol,
        "volume_zscore": volume_zscore,
    }, index=df.index)
    return hmm_df.dropna()


def build_features_for_symbol(
    symbol: str,
    df_15m: pd.DataFrame,
    df_1h,
    df_4h,
    df_1d,
    macro_panel: pd.DataFrame,
    onchain_panel: pd.DataFrame,
    btc_df,
    cfg,
    train_end_date: str,
) -> pd.DataFrame:

    train_end = pd.Timestamp(train_end_date, tz="UTC")
    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    hmm_dir = checkpoints_dir / "hmm" / symbol
    fracdiff_cache = checkpoints_dir / "fracdiff"

    # 1. Technical features on 15m
    logger.info(f"{symbol}: computing technical features")
    tech_15m = build_technical_features(df_15m, cfg)

    # 2. Technical features on HTF if available — merge to 15m
    all_features = tech_15m.copy()
    if df_1h is not None:
        tech_1h = build_technical_features(df_1h, cfg)
        all_features = _merge_htf(all_features, tech_1h, "1h", cfg)
    if df_4h is not None:
        tech_4h = build_technical_features(df_4h, cfg)
        all_features = _merge_htf(all_features, tech_4h, "4h", cfg)
    if df_1d is not None:
        tech_1d = build_technical_features(df_1d, cfg)
        all_features = _merge_htf(all_features, tech_1d, "1d", cfg)

    # 3. Microstructure on 15m
    logger.info(f"{symbol}: computing microstructure features")
    micro = build_microstructure_features(df_15m, cfg)
    all_features = pd.concat([all_features, micro], axis=1)

    # 4. Funding features on 15m
    logger.info(f"{symbol}: computing funding features")
    funding = build_funding_features(df_15m, btc_df, cfg)
    all_features = pd.concat([all_features, funding], axis=1)

    # 4b. BTC lag spillover for altcoins (lags 1-4)
    # Global shift(1) at step 9 makes these fully backward-looking at bar t
    if btc_df is not None and symbol != "BTCUSDT":
        btc_ret = np.log(btc_df["close"] / btc_df["close"].shift(1))
        btc_ret_aligned = btc_ret.reindex(df_15m.index)
        btc_lag_cols = {f"btc_lag_{lag}": btc_ret_aligned.shift(lag) for lag in [1, 2, 3, 4]}
        all_features = pd.concat([all_features, pd.DataFrame(btc_lag_cols, index=df_15m.index)], axis=1)

    # 4c. Market positioning features (OI, LS ratio, taker ratio)
    # Reindexed via ffill from 15m FAPI data — no look-ahead (all backward-looking)
    raw_dir = Path(cfg.data.raw_dir)
    positioning = build_market_positioning_features(symbol, df_15m.index, raw_dir, cfg)
    if not positioning.empty:
        all_features = pd.concat([all_features, positioning], axis=1)

    # 5. HMM regime features
    logger.info(f"{symbol}: computing regime features")
    hmm_input = _build_hmm_features_df(df_15m)

    if hmm_dir.exists() and (hmm_dir / "hmm_model.pkl").exists():
        # Load existing HMM fitted on train data
        hmm_model, scaler, state_labels = load_hmm_artifacts(hmm_dir)
        regime_probs = get_regime_probs(hmm_model, hmm_input, scaler)
    elif len(hmm_input) >= int(cfg.regime.hmm_burnin_bars):
        # Fit on train portion only — never fall back to full series (would leak val/test)
        train_hmm_input = hmm_input[hmm_input.index <= train_end]
        if len(train_hmm_input) < int(cfg.regime.hmm_burnin_bars):
            # Insufficient train bars — skip HMM rather than fit on future data (leakage)
            logger.warning(f"{symbol}: insufficient train bars for HMM ({len(train_hmm_input)} < {cfg.regime.hmm_burnin_bars}), emitting NaN regime probs")
            n_states = int(cfg.regime.n_states)
            regime_probs = pd.DataFrame(
                np.nan, index=hmm_input.index,
                columns=[f"regime_prob_{i}" for i in range(n_states)]
            )
        else:
            hmm_model, scaler = fit_hmm(train_hmm_input, int(cfg.regime.n_states), cfg)
            state_labels = label_regime_states(hmm_model, train_hmm_input)
            save_hmm_artifacts(hmm_model, scaler, state_labels, hmm_dir)
            regime_probs = get_regime_probs(hmm_model, hmm_input, scaler)
    else:
        n_states = int(cfg.regime.n_states)
        regime_probs = pd.DataFrame(
            np.nan, index=hmm_input.index,
            columns=[f"regime_prob_{i}" for i in range(n_states)]
        )

    # BOCPD changepoint distance
    bocpd_model = fit_bocpd(hmm_input["log_return"][hmm_input.index <= train_end], cfg)
    cp_dist = get_changepoint_distance(hmm_input["log_return"], bocpd_model, cfg)

    # ADX fallback for trend
    if "adx" in all_features.columns:
        adx_threshold = float(getattr(cfg.regime, 'adx_trend_threshold', 25.0))
        adx_flag = apply_adx_fallback(all_features["adx"], threshold=adx_threshold)
        all_features = pd.concat([all_features, adx_flag.to_frame()], axis=1)

    # Merge regime features
    regime_probs_aligned = regime_probs.reindex(all_features.index)
    cp_dist_aligned = cp_dist.reindex(all_features.index)
    all_features = pd.concat([all_features, regime_probs_aligned, cp_dist_aligned.to_frame()], axis=1)

    # 6. Fracdiff — fit on train only, then apply to full series
    price_vol_cols = [c for c in _FRACDIFF_COLS if c in all_features.columns]
    if price_vol_cols:
        d_step = float(cfg.features.fracdiff_d_step)
        threshold = float(cfg.features.fracdiff_threshold)
        d_values = load_d_values(symbol, "15m", fracdiff_cache)
        if not d_values:
            # Fit on train-only subset — leakage-safe regardless of whether full series spans beyond train_end
            train_df_for_fracdiff = all_features[all_features.index <= train_end]
            if len(train_df_for_fracdiff) >= 100:
                d_values = fit_and_save_d_values(train_df_for_fracdiff, price_vol_cols, symbol, "15m", fracdiff_cache)
            else:
                logger.warning(f"{symbol}: insufficient train bars for fracdiff ({len(train_df_for_fracdiff)}), skipping")
        if d_values:
            all_features = apply_fracdiff_transform(all_features, d_values)

    # 7. Merge macro and onchain panels (forward-fill alignment already done in stage 1)
    # limit=2880 = 30 days of 15m bars — prevents stale monthly releases propagating indefinitely
    _macro_ffill_limit = 2880
    if macro_panel is not None and len(macro_panel) > 0:
        macro_aligned = macro_panel.reindex(all_features.index, method="ffill", limit=_macro_ffill_limit)
        all_features = pd.concat([all_features, macro_aligned], axis=1)

    if onchain_panel is not None and len(onchain_panel) > 0:
        onchain_aligned = onchain_panel.reindex(all_features.index, method="ffill", limit=_macro_ffill_limit)
        all_features = pd.concat([all_features, onchain_aligned], axis=1)

    # 8. Mark warmup bars — use concat to avoid fragmenting an already-wide DataFrame
    warmup_bars = int(cfg.features.warmup_bars)
    is_warmup = pd.Series(0, index=all_features.index, name="is_warmup")
    is_warmup.iloc[:warmup_bars] = 1
    all_features = pd.concat([all_features, is_warmup], axis=1)

    # Remove duplicate columns before shift (can arise from HTF merge or concat)
    dup_cols = all_features.columns[all_features.columns.duplicated()].tolist()
    if dup_cols:
        logger.warning(f"{symbol}: dropping {len(dup_cols)} duplicate columns: {dup_cols[:10]}")
    all_features = all_features.loc[:, ~all_features.columns.duplicated()]

    # 9. SHIFT ALL FEATURES by 1 bar to prevent look-ahead
    # This ensures feature at time t uses only data available before t
    non_shift_cols = {"is_warmup"}
    shift_cols = [c for c in all_features.columns if c not in non_shift_cols]
    all_features[shift_cols] = all_features[shift_cols].shift(1)

    # 10. Drop fully-NaN rows (warmup artifacts)
    all_features = all_features.dropna(how="all")

    logger.info(f"{symbol}: feature pipeline complete — {all_features.shape[1]} features, {len(all_features)} bars")
    return all_features



def save_feature_manifest(all_feature_cols: dict, manifest_path: Path) -> None:
    # all_feature_cols: {symbol: [col_names]} or {symbol: col_names_list}
    manifest = {}
    for symbol, cols in all_feature_cols.items():
        if isinstance(cols, list):
            for col in cols:
                # Infer type from name convention
                col_type = "unknown"
                if any(x in col for x in ["rsi", "bb_", "macd", "adx", "atr", "obv", "vwap"]):
                    col_type = "technical"
                elif any(x in col for x in ["ofi", "amihud", "kyle", "parkinson", "roll_measure"]):
                    col_type = "microstructure"
                elif any(x in col for x in ["funding", "pre_funding", "hours_to"]):
                    col_type = "funding"
                elif any(x in col for x in ["regime", "bocpd", "hmm"]):
                    col_type = "regime"
                elif any(x in col for x in ["rv_", "bv", "jump", "realized"]):
                    col_type = "volatility"
                elif any(x in col for x in ["lag_"]):
                    col_type = "lag"
                window = None
                for part in col.split("_"):
                    if part.isdigit():
                        window = int(part)
                        break
                manifest[col] = {"type": col_type, "window": window, "shift": 1}

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Feature manifest saved: {manifest_path} ({len(manifest)} features)")
