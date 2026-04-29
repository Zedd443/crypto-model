import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from src.utils.logger import get_logger
from src.models.primary_model import _SigmoidCalibrator
from src.features.technical import compute_atr, compute_rsi

logger = get_logger("htf_model")

_HTF_WINDOWS = {
    "4h": [5, 10, 20, 50],
    "1d": [5, 10, 20, 50],
}

# Macro columns to include from the pre-built panels (value + staleness proxy)
# Skip *_days_since to keep feature count manageable — data_quality included as coverage signal
_MACRO_VALUE_COLS = [
    "macro_bond_10y", "macro_vix", "macro_usd_index", "macro_sp500",
    "macro_inflation_cpi", "macro_unemployment", "market_gold", "macro_data_quality",
]


def _build_htf_features(
    df: pd.DataFrame,
    tf: str,
    macro_panel: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    windows = _HTF_WINDOWS.get(tf, [5, 10, 20, 50])
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    log_ret = np.log(close / close.shift(1))
    out["log_return"] = log_ret

    for w in windows:
        out[f"mom_{w}"] = close / close.shift(w) - 1.0
        out[f"vol_{w}"] = log_ret.rolling(w, min_periods=w).std()
        sma = close.rolling(w, min_periods=w).mean()
        out[f"sma_dist_{w}"] = (close - sma) / (sma + 1e-9)

    # ATR-based features — normalized ATR useful for regime context
    atr14 = compute_atr(high, low, close, 14)
    out["atr_14"] = atr14
    out["atr_14_pct"] = atr14 / (close + 1e-9)  # normalized ATR as % of price

    out["rsi_14"] = compute_rsi(close, 14)

    vol_ma = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std()
    out["volume_zscore"] = (volume - vol_ma) / (vol_std + 1e-9)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd_line - signal_line

    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std_roll = close.rolling(20, min_periods=20).std()
    out["bb_pct"] = (close - (bb_mid - 2 * bb_std_roll)) / (4 * bb_std_roll + 1e-9)

    # ── HTF-privileged features ──────────────────────────────────────────────
    # These are meaningful specifically at higher timeframes (multi-week/month context)

    # Long-term SMA distances (50/200 bars = ~8 months / ~3.3 years for 4h; ~10 months / ~4 years for 1d)
    for long_w in [50, 100, 200]:
        sma_long = close.rolling(long_w, min_periods=max(long_w // 2, 20)).mean()
        out[f"sma_dist_{long_w}"] = (close - sma_long) / (sma_long + 1e-9)

    # Long-window RSI (50/100 bars) — structural overbought/oversold at HTF resolution
    out["rsi_50"] = compute_rsi(close, 50)
    out["rsi_100"] = compute_rsi(close, 100)

    # Weekly/monthly momentum (20/50/100 bars for 4h = 5 days/12.5 days/25 days)
    for long_mom in [20, 50, 100]:
        out[f"mom_{long_mom}_long"] = close / close.shift(long_mom) - 1.0

    # 52-week high/low distance (4h: 2184 bars = 364 days; 1d: 252 bars ≈ 1 year)
    lookback_52w = {"4h": 2184, "1d": 252}.get(tf, 252)
    min_p52 = max(lookback_52w // 4, 50)
    high_52w = close.rolling(lookback_52w, min_periods=min_p52).max()
    low_52w = close.rolling(lookback_52w, min_periods=min_p52).min()
    out["dist_52w_high"] = (close - high_52w) / (high_52w + 1e-9)   # always ≤ 0
    out["dist_52w_low"] = (close - low_52w) / (low_52w + 1e-9)      # always ≥ 0
    # Position within 52-week range [0, 1]
    range_52w = high_52w - low_52w
    out["pct_52w_range"] = (close - low_52w) / (range_52w + 1e-9)

    # Volume trend: short-window vol relative to long-window baseline
    vol_short = volume.rolling(10, min_periods=5).mean()
    vol_long = volume.rolling(50, min_periods=25).mean()
    out["volume_trend_ratio"] = vol_short / (vol_long + 1e-9)

    # Long-window volume z-score
    vol_ma50 = volume.rolling(50, min_periods=25).mean()
    vol_std50 = volume.rolling(50, min_periods=25).std()
    out["volume_zscore_50"] = (volume - vol_ma50) / (vol_std50 + 1e-9)

    # Day-of-week cyclical encoding — only meaningful for 1d bars
    if tf == "1d":
        dow = pd.Series(df.index.dayofweek, index=df.index, dtype=float)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        # Month cyclical (seasonality)
        month = pd.Series(df.index.month, index=df.index, dtype=float)
        out["month_sin"] = np.sin(2 * np.pi * month / 12)
        out["month_cos"] = np.cos(2 * np.pi * month / 12)

    # Hour-of-day for 4h bars (4 session slots: 0/4/8/12/16/20 UTC)
    if tf == "4h":
        hour = pd.Series(df.index.hour, index=df.index, dtype=float)
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # ── Macro features — merged from pre-built timeframe-native panel ────────
    if macro_panel is not None and len(macro_panel) > 0:
        available_macro = [c for c in _MACRO_VALUE_COLS if c in macro_panel.columns]
        if available_macro:
            # Forward-fill with generous limit (macro data is monthly/weekly)
            ffill_limit = {"4h": 2160, "1d": 90}.get(tf, 90)  # 4h: ~90 days; 1d: 3 months
            macro_aligned = (
                macro_panel[available_macro]
                .reindex(out.index, method="ffill", limit=ffill_limit)
            )
            out = pd.concat([out, macro_aligned], axis=1)

    # SHIFT by 1 bar — all features must use only past data at inference time
    out = out.shift(1)
    return out.dropna(how="all")


def _build_htf_label(df_htf: pd.DataFrame, tf: str, cfg) -> pd.Series:
    """
    ATR-based multi-bar barrier label on NATIVE HTF bars.

    Looks N bars ahead and checks whether close moves >= tp_mult×ATR14 (long)
    or <= -sl_mult×ATR14 (short) before the time barrier expires.
    Returns 1 (long) or 0 (short/no-signal) based on which barrier is hit first,
    or majority direction if only time barrier triggers.
    """
    htf_cfg = cfg.htf_models
    tp_mult = float(htf_cfg.get("label_tp_mult", 1.5))
    sl_mult = float(htf_cfg.get("label_sl_mult", 1.0))
    # N bars look-ahead per timeframe
    n_bars_defaults = {"4h": 6, "1d": 5}  # 4h: 24h window; 1d: 1 trading week
    n_bars = int(htf_cfg.get("label_n_bars", {}).get(tf, n_bars_defaults.get(tf, 5)))

    close = df_htf["close"].astype(float)
    high = df_htf["high"].astype(float)
    low = df_htf["low"].astype(float)
    atr14 = compute_atr(high, low, close, 14)

    labels = pd.Series(np.nan, index=df_htf.index, dtype=float)
    close_arr = close.values
    atr_arr = atr14.values
    n = len(close_arr)

    for i in range(n - n_bars):
        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            continue
        entry = close_arr[i]
        tp = entry + tp_mult * atr_arr[i]
        sl = entry - sl_mult * atr_arr[i]
        label = np.nan
        for j in range(i + 1, min(i + n_bars + 1, n)):
            hi = df_htf["high"].iloc[j]
            lo = df_htf["low"].iloc[j]
            # TP hit first (intra-bar check: high crosses TP before low crosses SL)
            if hi >= tp:
                label = 1
                break
            if lo <= sl:
                label = 0
                break
        else:
            # Time barrier: use simple return direction as tiebreak
            end_close = close_arr[min(i + n_bars, n - 1)]
            label = 1 if end_close > entry else 0
        labels.iloc[i] = label

    return labels.dropna().astype(int)


def train_htf_model(
    symbol: str,
    tf: str,
    df_htf: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    cfg,
    macro_panel: "pd.DataFrame | None" = None,
) -> tuple:
    _min_bars_defaults = {"4h": 200, "1d": 100}
    _min_bars_cfg = cfg.htf_models.get("min_train_bars", None)
    min_bars = int(
        _min_bars_cfg.get(tf, _min_bars_defaults.get(tf, 100)) if _min_bars_cfg is not None
        else _min_bars_defaults.get(tf, 100)
    )

    feat_df = _build_htf_features(df_htf, tf, macro_panel=macro_panel)
    y_all = _build_htf_label(df_htf, tf, cfg)

    # Align features and labels on common index
    common_idx = feat_df.index.intersection(y_all.index)
    X_all = feat_df.loc[common_idx]
    y_all = y_all.loc[common_idx]

    train_mask = common_idx <= train_end
    val_mask = (common_idx > train_end) & (common_idx <= val_end)

    X_train = X_all[train_mask].select_dtypes(include=[np.number]).fillna(0)
    y_train = y_all[train_mask]
    X_val = X_all[val_mask].select_dtypes(include=[np.number]).fillna(0)
    y_val = y_all[val_mask]

    if len(X_train) < min_bars:
        raise ValueError(f"{symbol} {tf}: insufficient train bars ({len(X_train)} < {min_bars})")

    feature_names = X_train.columns.tolist()

    n_long = int(y_train.sum())
    n_short = int((y_train == 0).sum())
    spw = n_short / max(n_long, 1)

    n_estimators = int(cfg.htf_models.get("n_estimators", 300))
    max_depth = int(cfg.htf_models.get("max_depth", 4))
    learning_rate = float(cfg.htf_models.get("learning_rate", 0.05))
    early_stopping = int(cfg.htf_models.get("early_stopping_rounds", 30))

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping,
        verbosity=0,
    )

    val_es_ratio = float(cfg.htf_models.get("val_es_ratio", 0.8))
    min_cal_bars = int(cfg.htf_models.get("min_cal_bars", 2))
    if len(X_val) >= 10:
        cal_split = min(int(len(X_val) * val_es_ratio), len(X_val) - min_cal_bars)
        es_X, es_y = X_val.values[:cal_split], y_val.values[:cal_split]
        cal_X, cal_y = X_val.values[cal_split:], y_val.values[cal_split:]
    else:
        es_X = X_train.values[-20:]
        es_y = y_train.values[-20:]
        cal_X = X_val.values if len(X_val) > 0 else es_X
        cal_y = y_val.values if len(y_val) > 0 else es_y

    model.fit(
        X_train.values, y_train.values,
        eval_set=[(es_X, es_y)],
        verbose=False,
    )

    cal_raw = model.predict_proba(cal_X)[:, 1]
    if len(np.unique(cal_y)) < 2:
        logger.warning(
            f"{symbol} {tf}: cal set has only 1 class ({len(cal_y)} samples) — "
            "augmenting with one synthetic opposite-class sample to allow LR fit"
        )
        from sklearn.linear_model import LogisticRegression as _LR
        _identity_lr = _LR(C=1e9, max_iter=1000, random_state=42, warm_start=False)
        _aug_X = np.concatenate([cal_raw.reshape(-1, 1), [[0.5]]])
        _aug_y = np.concatenate([cal_y, [1 - int(cal_y[0])]])
        _identity_lr.fit(_aug_X, _aug_y)
        calibrator = _SigmoidCalibrator(_identity_lr)
    else:
        lr_cal = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr_cal.fit(cal_raw.reshape(-1, 1), cal_y)
        calibrator = _SigmoidCalibrator(lr_cal)

    pct_long = float(y_train.mean())
    logger.info(
        f"{symbol} {tf}: HTF model trained — {len(X_train)} train bars, "
        f"best_iter={model.best_iteration}, n_features={len(feature_names)}, "
        f"pct_long={pct_long:.2%}, spw={spw:.2f}"
    )
    return model, calibrator, feature_names


def save_htf_model(model, calibrator, feature_names: list, symbol: str, tf: str, models_dir: Path) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{symbol}_{tf}_htf_model.json"
    cal_path = models_dir / f"{symbol}_{tf}_htf_calibrator.pkl"
    feat_path = models_dir / f"{symbol}_{tf}_htf_features.json"

    model.save_model(str(model_path))
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    import json
    with open(feat_path, "w") as f:
        json.dump(feature_names, f)
    logger.debug(f"HTF model saved: {model_path}")


def load_htf_model(symbol: str, tf: str, models_dir: Path) -> tuple:
    import json
    models_dir = Path(models_dir)
    model_path = models_dir / f"{symbol}_{tf}_htf_model.json"
    cal_path = models_dir / f"{symbol}_{tf}_htf_calibrator.pkl"
    feat_path = models_dir / f"{symbol}_{tf}_htf_features.json"

    if not model_path.exists():
        raise FileNotFoundError(f"HTF model not found: {model_path}")

    model = XGBClassifier()
    model.load_model(str(model_path))
    with open(cal_path, "rb") as f:
        calibrator = pickle.load(f)
    with open(feat_path) as f:
        feature_names = json.load(f)
    return model, calibrator, feature_names


def predict_htf_proba(
    model,
    calibrator,
    feature_names: list,
    df_htf: pd.DataFrame,
    tf: str,
    macro_panel: "pd.DataFrame | None" = None,
) -> pd.Series:
    feat_df = _build_htf_features(df_htf, tf, macro_panel=macro_panel)
    missing = [c for c in feature_names if c not in feat_df.columns]
    if missing:
        logger.warning(
            f"HTF predict ({tf}): {len(missing)} trained features missing from live compute — "
            f"filling NaN. First 5: {missing[:5]}"
        )
        for col in missing:
            feat_df[col] = np.nan
    X = feat_df[feature_names].fillna(0).values
    raw = model.predict_proba(X)[:, 1]
    cal = calibrator.predict(raw)
    return pd.Series(cal, index=feat_df.index, name=f"htf_pred_{tf}")
