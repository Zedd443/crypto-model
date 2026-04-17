import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from src.utils.logger import get_logger

logger = get_logger("htf_model")

# Technical features built directly from raw OHLCV of 4h/1d bars
# Kept minimal — only features that are well-defined with ~few-hundred rows (1d) or ~few-thousand (4h)
_HTF_WINDOWS = {
    "4h": [5, 10, 20, 50],
    "1d": [5, 10, 20, 50],
}


def _build_htf_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    windows = _HTF_WINDOWS.get(tf, [5, 10, 20, 50])
    out = pd.DataFrame(index=df.index)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # Log returns
    log_ret = np.log(close / close.shift(1))
    out["log_return"] = log_ret

    for w in windows:
        # Price momentum
        out[f"mom_{w}"] = close / close.shift(w) - 1.0
        # Rolling vol
        out[f"vol_{w}"] = log_ret.rolling(w, min_periods=w).std()
        # SMA distance
        sma = close.rolling(w, min_periods=w).mean()
        out[f"sma_dist_{w}"] = (close - sma) / (sma + 1e-9)

    # ATR
    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_14 = tr.rolling(14, min_periods=14).mean()
    out["atr_14"] = atr_14

    # RSI-14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / (loss + 1e-9)
    out["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    # Volume z-score
    vol_ma = volume.rolling(20, min_periods=20).mean()
    vol_std = volume.rolling(20, min_periods=20).std()
    out["volume_zscore"] = (volume - vol_ma) / (vol_std + 1e-9)

    # MACD signal line diff (12-26-9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd_line - signal_line

    # Bollinger band position
    bb_mid = close.rolling(20, min_periods=20).mean()
    bb_std = close.rolling(20, min_periods=20).std()
    out["bb_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-9)

    # SHIFT by 1 bar — all features must use only past data
    out = out.shift(1)
    return out.dropna(how="all")


class _SigmoidCalibrator:
    def __init__(self, lr_model):
        self._lr = lr_model

    def predict(self, raw_probs):
        arr = np.asarray(raw_probs).reshape(-1, 1)
        return self._lr.predict_proba(arr)[:, 1]


def train_htf_model(
    symbol: str,
    tf: str,
    df_htf: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    cfg,
) -> tuple:
    """
    Train a lightweight XGBoost classifier directly on 4h or 1d OHLCV bars.
    Labels: +1 if next-bar log_return > 0, -1 otherwise (binary).
    Returns (model, calibrator, feature_names) or raises on failure.
    """
    _min_bars_defaults = {"4h": 200, "1d": 100}
    _min_bars_cfg = cfg.htf_models.get("min_train_bars", None)
    min_bars = int(
        _min_bars_cfg.get(tf, _min_bars_defaults.get(tf, 100)) if _min_bars_cfg is not None
        else _min_bars_defaults.get(tf, 100)
    )

    feat_df = _build_htf_features(df_htf, tf)

    # Label: next bar direction (shift(-1) removed to prevent forward-looking leakage)
    # Predict actual next bar's return without lookahead
    next_ret = np.log(df_htf["close"].astype(float).shift(-1) / df_htf["close"].astype(float))
    y_all = (next_ret > 0).astype(int)

    # Align
    common_idx = feat_df.index.intersection(y_all.dropna().index)
    X_all = feat_df.loc[common_idx]
    y_all = y_all.loc[common_idx]

    # Split
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
        use_label_encoder=False,
        tree_method="hist",
        device="cpu",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=early_stopping,
        verbosity=0,
    )

    eval_X = X_val.values if len(X_val) > 0 else X_train.values[-20:]
    eval_y = y_val.values if len(y_val) > 0 else y_train.values[-20:]

    model.fit(
        X_train.values, y_train.values,
        eval_set=[(eval_X, eval_y)],
        verbose=False,
    )

    # Calibrate on val set (Platt scaling)
    val_raw = model.predict_proba(eval_X)[:, 1]
    lr_cal = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_cal.fit(val_raw.reshape(-1, 1), eval_y)
    calibrator = _SigmoidCalibrator(lr_cal)

    logger.info(
        f"{symbol} {tf}: HTF model trained — {len(X_train)} train bars, "
        f"best_iter={model.best_iteration}, n_features={len(feature_names)}"
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
    model, calibrator, feature_names: list, df_htf: pd.DataFrame, tf: str
) -> pd.Series:
    """
    Return calibrated long-probability series aligned to the HTF bar index.
    """
    feat_df = _build_htf_features(df_htf, tf)
    # Use only the features the model was trained on
    avail = [c for c in feature_names if c in feat_df.columns]
    X = feat_df[avail].fillna(0).values
    raw = model.predict_proba(X)[:, 1]
    cal = calibrator.predict(raw)
    return pd.Series(cal, index=feat_df.index, name=f"htf_pred_{tf}")
