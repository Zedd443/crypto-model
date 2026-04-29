from pathlib import Path
import numpy as np
import pandas as pd
import json
from src.utils.config_loader import get_symbols
from src.utils.state_manager import is_stage_complete, update_project_state
from src.utils.logger import get_logger
from src.utils.io_utils import read_features, read_checkpoint, checkpoint_exists
from src.models.model_versioning import get_latest_model
from src.models.primary_model import load_model
from src.models.meta_labeler import load_meta_model
from src.models.imputer import transform_with_imputer, transform_with_scaler
from src.models.htf_model import load_htf_model, predict_htf_proba
from src.portfolio.signal_generator import generate_signals, apply_uncertainty_scaling
from src.portfolio.position_sizer import compute_half_kelly, compute_position_size, apply_memecoin_rules
from src.portfolio.correlation import fit_garch_per_asset, compute_dcc_correlations
from src.portfolio.optimizer import optimize_portfolio_weights, equal_weight_fallback

logger = get_logger("stage_06_portfolio")

_TF = "15m"
# HTF timeframes checked in the approval filter — must match stage_04b._HTF_TIMEFRAMES
_HTF_APPROVAL_TFS = ["4h", "1d"]


def _load_htf_macro_panels(cfg) -> dict:
    processed_dir = Path(cfg.data.processed_dir)
    panels = {}
    for tf in _HTF_APPROVAL_TFS:
        p = processed_dir / f"macro_panel_{tf}.parquet"
        if p.exists():
            try:
                panels[tf] = pd.read_parquet(p)
            except Exception as exc:
                logger.warning(f"stage_06: could not load macro_panel_{tf}: {exc}")
    return panels


def _apply_htf_approval(
    signals: pd.DataFrame,
    symbol: str,
    cfg,
    checkpoints_dir: Path,
    models_dir: Path,
    htf_macro_panels: dict | None = None,
) -> pd.DataFrame:
    # Load HTF predictions for each timeframe and zero out signals where HTF disagrees.
    # Graceful degradation: missing model or missing checkpoint → skip that timeframe's filter.
    threshold = float(cfg.features.get("htf_approval_threshold", 0.45))

    for tf in _HTF_APPROVAL_TFS:
        model_path = models_dir / f"{symbol}_{tf}_htf_model.json"
        if not model_path.exists():
            logger.warning(f"  {symbol}: HTF model missing for {tf}, skipping approval filter")
            continue

        if not checkpoint_exists("ingest", symbol, tf, checkpoints_dir):
            logger.warning(f"  {symbol}: no {tf} ingest checkpoint, skipping HTF approval for {tf}")
            continue

        try:
            df_htf = read_checkpoint("ingest", symbol, tf, checkpoints_dir)
            model, calibrator, feature_names = load_htf_model(symbol, tf, models_dir)
            macro_panel = (htf_macro_panels or {}).get(tf, None)
            htf_pred = predict_htf_proba(
                model, calibrator, feature_names, df_htf, tf,
                macro_panel=macro_panel,
            )
        except Exception as e:
            logger.warning(f"  {symbol}: HTF {tf} prediction failed ({e}), skipping approval filter")
            continue

        # Forward-fill HTF predictions onto 15m index — bounded by htf_ffill_limits
        ffill_limit = int(cfg.features.htf_ffill_limits.get(tf, 4))
        htf_aligned = htf_pred.reindex(signals.index, method="ffill", limit=ffill_limit)

        # Approval logic: LONG signal requires pred >= threshold; SHORT requires pred <= (1 - threshold)
        # Bars where htf_aligned is NaN (beyond ffill window) are not penalised — only active disagreement zeroes the signal
        htf_vals = htf_aligned.values
        long_disapproved = (signals["direction"].values == 1) & (~np.isnan(htf_vals)) & (htf_vals < threshold)
        short_disapproved = (signals["direction"].values == -1) & (~np.isnan(htf_vals)) & (htf_vals > (1.0 - threshold))
        disapproved = long_disapproved | short_disapproved

        n_disapproved = int(disapproved.sum())
        if n_disapproved > 0:
            logger.info(f"  {symbol}: HTF {tf} approval filter zeroed {n_disapproved} signals")
            signals.loc[disapproved, "signal_strength"] = 0.0
            signals.loc[disapproved, "direction"] = 0
            signals.loc[disapproved, "is_signal"] = 0

    return signals


def _generate_symbol_signals(
    symbol: str,
    cfg,
    checkpoints_dir: Path,
    features_dir: Path,
    models_dir: Path,
    htf_macro_panels: dict | None = None,
) -> tuple:
    # Load primary model
    primary_entry = get_latest_model(symbol, _TF, model_type="primary")
    if primary_entry is None:
        return symbol, None, "No primary model found"

    meta_entry = get_latest_model(symbol, _TF, model_type="meta")

    version = primary_entry["version"]
    selected_features = primary_entry.get("feature_names", [])

    # Fallback: load feature list from checkpoint if registry entry predates this field
    if not selected_features:
        sel_path = checkpoints_dir / "feature_selection" / f"{symbol}_{_TF}_selected.json"
        if sel_path.exists():
            with open(sel_path) as _f:
                selected_features = json.load(_f)

    if not selected_features:
        return symbol, None, "Feature list not found in registry or checkpoint"

    try:
        primary_model, calibrator = load_model(symbol, _TF, version, models_dir)
    except Exception as e:
        return symbol, None, f"Cannot load primary model: {e}"

    # Load features
    try:
        features_df = read_features(symbol, _TF, features_dir)
    except Exception as e:
        return symbol, None, f"Cannot load features: {e}"

    # Apply imputer and scaler (train-fitted)
    imp_dir = checkpoints_dir / "imputers"
    # Select only the features used at training time
    avail_feats = [f for f in selected_features if f in features_df.columns]
    if len(avail_feats) != len(selected_features):
        logger.warning(f"  {symbol}: {len(selected_features) - len(avail_feats)} selected features missing from feature file")
    X = features_df[avail_feats].select_dtypes(include=[np.number])

    try:
        X_imp = transform_with_imputer(X.values, symbol, _TF, imp_dir)
        X_scaled = transform_with_scaler(X_imp, symbol, _TF, imp_dir)
    except Exception:
        X_scaled = X.values

    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    # Primary model predictions
    # Use RAW (uncalibrated) proba for direction/signal_strength — calibrated proba collapses
    # to a narrow band (e.g. 0.97-0.98) when val set is heavily imbalanced (>90% label=+1),
    # causing direction to always be 1 and signal_strength to have near-zero variance.
    # OOF proba (used to train meta) is raw — must stay consistent (ISSUE-051).
    # Calibrated proba is only used for the uncertainty_proxy (1-2|p-0.5|) below.
    try:
        raw_proba = primary_model.predict_proba(X_scaled)
        cal_proba = calibrator.predict(raw_proba[:, 1])
        proba_df = pd.DataFrame({
            "prob_long": raw_proba[:, 1],
            "prob_short": raw_proba[:, 0],
        }, index=features_df.index)
    except Exception as e:
        return symbol, None, f"Primary model prediction failed: {e}"

    # Meta model predictions
    meta_proba_series = pd.Series(0.5, index=features_df.index, name="meta_prob")
    if meta_entry is not None:
        try:
            meta_model = load_meta_model(symbol, _TF, meta_entry["version"], models_dir)
            # Build meta features (minimal set)
            regime_cols = [c for c in features_df.columns if c.startswith("regime_prob_")]
            regime_probs = features_df[regime_cols] if regime_cols else None
            rv = features_df.get("rv_daily", pd.Series(0.0, index=features_df.index))
            vol_z = features_df.get("volume_surprise_20", pd.Series(0.0, index=features_df.index))
            ofi = features_df.get("ofi_20", pd.Series(0.0, index=features_df.index))

            from src.models.meta_labeler import build_meta_features
            spread_series = (
                features_df.get("spread_proxy_20", pd.Series(np.nan, index=features_df.index))
            )
            atr_series = (
                features_df.get("atr_14", pd.Series(np.nan, index=features_df.index))
            )
            meta_X = build_meta_features(
                raw_proba, regime_probs, rv, vol_z, ofi,
                spread_series=spread_series,
                atr_series=atr_series,
            ).fillna(0.0)
            # Align columns to what meta model was trained on (prevents shape mismatch)
            if meta_entry.get("feature_names"):
                expected_cols = meta_entry["feature_names"]
                for col in expected_cols:
                    if col not in meta_X.columns:
                        meta_X[col] = 0.0
                meta_X = meta_X[expected_cols]
            meta_raw = meta_model.predict_proba(meta_X.values)[:, 1]
            meta_proba_series = pd.Series(meta_raw, index=features_df.index, name="meta_prob")
        except Exception as e:
            logger.warning(f"  {symbol}: meta prediction failed ({e}), using 0.5")

    # Get conformal q90 from model metrics
    conformal_q90 = float(primary_entry.get("metrics", {}).get("conformal_q90", 0.20))

    # Build regime DataFrame for signal generator
    regime_cols = [c for c in features_df.columns if c.startswith("regime_prob_")]
    regime_df = features_df[regime_cols] if regime_cols else pd.DataFrame(index=features_df.index)

    # Generate signals (ALL pre-computed before backtest)
    signals = generate_signals(proba_df, meta_proba_series, regime_df, cfg)

    # HTF approval filter — zero out 15m signals where 1h/4h/1d models disagree
    # Must run after generate_signals and before size scaling so disapproved signals
    # carry zero strength through the entire position sizing path.
    signals = _apply_htf_approval(signals, symbol, cfg, checkpoints_dir, models_dir, htf_macro_panels)

    # uncertainty_proxy = 1 - 2|p - 0.5|: 0 = maximally certain (|p-0.5|=0.5), 1 = uncertain (p≈0.5)
    # This is NOT a conformal interval width — use apply_uncertainty_scaling, not apply_conformal_size_scaling
    uncertainty_proxy = 1.0 - abs(raw_proba[:, 1] - 0.5) * 2
    signals["uncertainty_proxy"] = uncertainty_proxy

    # Add ATR for position sizing
    atr_col = "atr_14"
    if atr_col in features_df.columns:
        signals["atr"] = features_df[atr_col].values
    else:
        # Fallback: 1% of close
        if "close_20_mean" in features_df.columns:
            signals["atr"] = features_df["close_20_mean"].values * 0.01
        else:
            signals["atr"] = 0.01

    # Compute position sizes
    equity = float(getattr(getattr(cfg, "account", None), "current_equity", None) or 120.0)
    backtest_summary_path = Path(cfg.data.results_dir) / "backtest_summary.json"
    win_rate = 0.52
    if backtest_summary_path.exists():
        with open(backtest_summary_path) as _f:
            _bs = json.load(_f)
        _wr = _bs.get("metrics", {}).get("hit_rate")
        if _wr is not None and 0.3 <= float(_wr) <= 0.8:
            win_rate = float(_wr)
        else:
            logger.warning(f"backtest_summary metrics.hit_rate={_wr} out of range or missing, using fallback 0.52")
    else:
        logger.warning("backtest_summary.json not found, using win_rate=0.52 fallback")
    avg_win = float(cfg.labels.tp_max_pct)
    avg_loss = float(cfg.labels.sl_max_pct)
    half_kelly = compute_half_kelly(win_rate, avg_win, avg_loss)

    leverage = 1
    tier = primary_entry.get("metrics", {}).get("tier", "B")
    if tier == "A":
        leverage = int(cfg.model.tier_A_leverage_default)

    base_size = compute_position_size(0.65, half_kelly, equity, leverage, cfg)
    base_notional = base_size["notional"]

    # Kelly returns 0 when R:R is unfavorable (e.g. 1%TP/5%SL needs >80% win rate).
    # Fall back to 5% of equity so the backtest has meaningful position sizes.
    if base_notional <= 0:
        base_notional = equity * 0.05
        logger.warning(f"{symbol}: Kelly=0 (unfavorable R:R), using fixed 5% equity fallback={base_notional:.2f} USDT")

    # Apply memecoin rules
    base_notional = apply_memecoin_rules(symbol, base_notional, cfg)

    # Scale by uncertainty_proxy and signal strength
    def _compute_size(row):
        if not row["is_signal"]:
            return 0.0
        scaled = apply_uncertainty_scaling(base_notional, float(row["uncertainty_proxy"]), cfg)
        return scaled * float(row["signal_strength"])

    signals["position_size_usd"] = signals.apply(_compute_size, axis=1)

    return symbol, signals, None


def run(cfg, force: bool = False, symbol_filter: str = None) -> None:
    if not force and is_stage_complete("portfolio"):
        logger.info("Stage 6 already complete, skipping.")
        return

    all_symbols = get_symbols(cfg)
    if symbol_filter:
        _sf = set(symbol_filter) if isinstance(symbol_filter, list) else {symbol_filter}
        all_symbols = [s for s in all_symbols if s.get("name", s.get("symbol")) in _sf]
    symbol_names = [s.get("name", s.get("symbol")) for s in all_symbols]

    checkpoints_dir = Path(cfg.data.checkpoints_dir)
    features_dir = Path(cfg.data.features_dir)
    models_dir = Path(cfg.data.models_dir)
    signals_dir = checkpoints_dir / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)

    htf_macro_panels = _load_htf_macro_panels(cfg)

    issues = []
    all_signals = {}

    # Phase 1: Generate per-symbol signals (ALL pre-computed before backtest)
    logger.info("Phase 1: generating pre-computed signals")
    for symbol in symbol_names:
        sym, signals, err = _generate_symbol_signals(
            symbol, cfg, checkpoints_dir, features_dir, models_dir,
            htf_macro_panels=htf_macro_panels,
        )
        if err:
            logger.error(f"{sym}: signal generation failed — {err}")
            issues.append(f"{sym}: {str(err)[:200]}")
        else:
            all_signals[sym] = signals
            n_active = int(signals["is_signal"].sum()) if signals is not None else 0
            logger.info(f"{sym}: {n_active} active signals generated")
            # Save signals checkpoint
            signals.to_parquet(signals_dir / f"{sym}_{_TF}_signals.parquet")

    # Phase 2: DCC-GARCH correlation fitting (on train data only)
    logger.info("Phase 2: fitting DCC-GARCH correlations")
    corr_matrix_df = None
    if len(all_signals) > 1:
        try:
            from tqdm import tqdm
            # Build returns panel from feature files (log_return column, train period only to avoid leakage)
            returns_parts = {}
            train_end = pd.Timestamp(cfg.data.train_end, tz="UTC")
            syms_list = list(all_signals.keys())
            for sym in tqdm(syms_list, desc="Phase 2 — loading returns", unit="sym", dynamic_ncols=True):
                try:
                    feat_df = read_features(sym, _TF, features_dir)
                    if "log_return" in feat_df.columns:
                        # Fit GARCH on train data only — never on test/val
                        returns_parts[sym] = feat_df.loc[:train_end, "log_return"]
                except Exception:
                    pass

            if returns_parts:
                returns_panel = pd.DataFrame(returns_parts).dropna(how="all")
                garch_results = fit_garch_per_asset(returns_panel, train_end)
                corr_matrix_df = compute_dcc_correlations(returns_panel, garch_results, cfg)
                corr_path = checkpoints_dir / "corr_matrix.parquet"
                corr_matrix_df.to_parquet(corr_path)
                logger.info(f"Correlation matrix saved: {corr_path}")
        except Exception as e:
            logger.warning(f"DCC-GARCH failed: {e}")
            issues.append(f"dcc_garch: {e}")

    # Phase 3: Portfolio optimization (per rebalance period)
    logger.info("Phase 3: portfolio optimization")
    try:
        _run_portfolio_optimization(all_signals, corr_matrix_df, cfg, checkpoints_dir)
    except Exception as e:
        logger.warning(f"Portfolio optimization failed: {e}")
        issues.append(f"portfolio_opt: {e}")

    update_project_state("portfolio", "done", issues, output_dir=str(signals_dir))
    logger.info(f"Stage 6 complete. {len(all_signals)}/{len(symbol_names)} symbols with signals.")


def _run_portfolio_optimization(all_signals: dict, corr_matrix_df, cfg, checkpoints_dir: Path) -> None:
    if not all_signals:
        return

    symbols = list(all_signals.keys())
    n_assets = len(symbols)
    max_weight = float(cfg.portfolio.max_position_size)

    # Initial equal weights
    prev_weights = equal_weight_fallback(n_assets, max_weight)

    # Compute expected returns from signal strength on val period only.
    # Excluding test period prevents test-period signal_strength from influencing
    # portfolio weights that are subsequently evaluated on that same test period.
    val_start_ts = pd.Timestamp(cfg.data.val_start, tz="UTC")
    test_start_ts = pd.Timestamp(cfg.data.test_start, tz="UTC")
    expected_returns = np.array([
        float(
            all_signals[sym].loc[
                (all_signals[sym].index >= val_start_ts) & (all_signals[sym].index < test_start_ts),
                "signal_strength",
            ].mean()
        )
        if ((all_signals[sym].index >= val_start_ts) & (all_signals[sym].index < test_start_ts)).any()
        else float(all_signals[sym].loc[all_signals[sym].index < test_start_ts, "signal_strength"].mean())
        for sym in symbols
    ])

    # Get latest correlation matrix snapshot
    corr_matrix = np.eye(n_assets)
    if corr_matrix_df is not None and len(corr_matrix_df) > 0:
        last_row = corr_matrix_df.iloc[-1]
        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols):
                key1 = f"corr_{s1}_{s2}"
                key2 = f"corr_{s2}_{s1}"
                if key1 in last_row.index:
                    corr_matrix[i, j] = float(last_row[key1])
                elif key2 in last_row.index:
                    corr_matrix[i, j] = float(last_row[key2])

    # CVaR placeholder (computed in backtest stage)
    cvar_val = 0.02

    optimal_weights = optimize_portfolio_weights(
        expected_returns, corr_matrix, cvar_val, prev_weights, n_assets, cfg
    )

    weights_dict = {sym: float(w) for sym, w in zip(symbols, optimal_weights)}
    logger.info(f"Optimal weights: {weights_dict}")

    # Save weights
    weights_path = checkpoints_dir / "portfolio_weights.json"
    with open(weights_path, "w") as f:
        json.dump(weights_dict, f, indent=2)
