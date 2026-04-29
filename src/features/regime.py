import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from src.utils.logger import get_logger

logger = get_logger("regime")

try:
    import ruptures
    _RUPTURES_AVAILABLE = True
except ImportError:
    _RUPTURES_AVAILABLE = False
    logger.warning("ruptures not installed — BOCPD changepoint detection disabled")


def fit_hmm(features_df: pd.DataFrame, n_states: int, cfg, scaler: StandardScaler = None) -> tuple:
    # features_df: [log_return, realized_vol_20, volume_zscore]
    # Returns (GaussianHMM, StandardScaler)
    X = features_df.values
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    # Replace NaN with 0 after scaling (warmup rows)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    model = GaussianHMM(
        n_components=n_states,
        covariance_type=cfg.regime.hmm_covariance_type,
        n_iter=cfg.regime.hmm_n_iter,
        random_state=42,
    )
    model.fit(X_scaled)
    logger.info(f"HMM fitted: {n_states} states, converged={model.monitor_.converged}")
    return model, scaler


def get_regime_probs(hmm_model: GaussianHMM, features_df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    X = features_df.values
    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    probs = hmm_model.predict_proba(X_scaled)
    n_states = probs.shape[1]
    cols = [f"regime_prob_{i}" for i in range(n_states)]
    return pd.DataFrame(probs, index=features_df.index, columns=cols)


def label_regime_states(hmm_model: GaussianHMM, train_returns_df: pd.DataFrame) -> dict:
    # Assign semantic labels using means and variances of each state
    # means shape: (n_states, n_features) — features are [log_return, realized_vol, volume_zscore]
    means = hmm_model.means_  # shape (n_states, n_features)
    # covars shape: (n_states, n_features, n_features) for 'full'
    covars = hmm_model.covars_

    n_states = means.shape[0]
    ret_means = means[:, 0]      # log return mean per state
    vol_means = means[:, 1]      # realized vol mean per state

    state_labels = {}
    remaining = list(range(n_states))

    # Crisis: highest vol + most negative return
    crisis_score = -ret_means + vol_means
    crisis_idx = int(np.argmax(crisis_score))
    state_labels[crisis_idx] = "crisis"
    remaining = [i for i in remaining if i != crisis_idx]

    if remaining:
        # Low-vol range: lowest vol among remaining, return near 0
        vol_remaining = vol_means[remaining]
        low_vol_idx = remaining[int(np.argmin(vol_remaining))]
        state_labels[low_vol_idx] = "low_vol_range"
        remaining = [i for i in remaining if i != low_vol_idx]

    if remaining:
        # High-vol range: highest vol among remaining, return near 0
        vol_remaining = vol_means[remaining]
        high_vol_idx = remaining[int(np.argmax(vol_remaining))]
        state_labels[high_vol_idx] = "high_vol_range"
        remaining = [i for i in remaining if i != high_vol_idx]

    # Remaining is trending
    for idx in remaining:
        state_labels[idx] = "trending"

    logger.info(f"HMM state labels: {state_labels}")
    return state_labels


def apply_adx_fallback(adx_series: pd.Series, threshold: float = 25.0) -> pd.Series:
    # Binary: 1 if trending (ADX > threshold), 0 otherwise
    return (adx_series > threshold).astype(int).rename("adx_trend_flag")


def fit_bocpd(train_series: pd.Series, cfg) -> object:
    if not _RUPTURES_AVAILABLE:
        logger.warning("ruptures not available, returning None for BOCPD model")
        return None
    signal = train_series.dropna().values.reshape(-1, 1)
    # Use "l2" (mean-shift) — O(n log n), feasible for large series
    # "rbf" requires O(n^2) kernel matrix, infeasible for >10k bars
    model = ruptures.Binseg(model="l2", min_size=int(cfg.regime.bocpd_min_size))
    model.fit(signal)
    return model


def get_changepoint_distance(series: pd.Series, bocpd_model, cfg=None) -> pd.Series:
    # Returns bars_since_last_changepoint for each timestamp
    if bocpd_model is None or not _RUPTURES_AVAILABLE:
        return pd.Series(np.nan, index=series.index, name="bars_since_changepoint")

    signal = series.dropna().values.reshape(-1, 1)
    n = len(signal)
    if n < 2:
        return pd.Series(np.nan, index=series.index, name="bars_since_changepoint")

    try:
        # Penalty = max(floor, log(n) * multiplier) — tuneable via config
        # bocpd_penalty_floor and bocpd_penalty_mult live in cfg.regime
        _floor = int(getattr(getattr(cfg, 'regime', cfg), 'bocpd_penalty_floor', 5)) if cfg is not None else 5
        _mult = float(getattr(getattr(cfg, 'regime', cfg), 'bocpd_penalty_mult', 2.0)) if cfg is not None else 2.0
        penalty = max(_floor, int(np.log(n) * _mult))
        breakpoints = bocpd_model.predict(pen=penalty)
    except Exception as e:
        logger.warning(f"BOCPD predict failed: {e}")
        return pd.Series(np.nan, index=series.index, name="bars_since_changepoint")

    # Build bars_since array on the dropna index
    dropna_index = series.dropna().index
    # breakpoints[-1] == n (sentinel end-of-series), exclude it
    cp_indices = np.array(sorted(breakpoints[:-1]), dtype=np.intp)

    # Vectorized "distance to last event" using cumsum trick:
    # Mark changepoint positions in a boolean array, then for each bar
    # compute position - position_of_last_True_at_or_before_this_bar.
    is_cp = np.zeros(n, dtype=np.float64)
    if len(cp_indices) > 0:
        # Clip to valid range (ruptures guarantees 0 <= bp < n, but guard anyway)
        valid = cp_indices[(cp_indices >= 0) & (cp_indices < n)]
        is_cp[valid] = 1.0

    # cumsum of is_cp gives a monotone counter; positions array is just arange.
    # For each i: last_cp_i = max index j<=i where is_cp[j]==1, else 0.
    # bars_since[i] = i - last_cp_i, with last_cp_i=0 before any changepoint.
    cum_cp = np.cumsum(is_cp)  # increases at each CP position
    # For each CP count value c, the "last CP position" when cum_cp==c is the
    # index of the c-th changepoint. Map each bar to its last_cp position via searchsorted.
    cp_positions = np.concatenate([[0], cp_indices]) if len(cp_indices) > 0 else np.array([0])
    # cum_cp[i] gives how many CPs have occurred up to and including i.
    # last_cp[i] = cp_positions[cum_cp[i]] (0-indexed into cp_positions which starts at 0).
    last_cp_arr = cp_positions[cum_cp.astype(np.intp)]
    bars_since = np.arange(n, dtype=np.float64) - last_cp_arr

    result = pd.Series(bars_since, index=dropna_index, name="bars_since_changepoint")
    return result.reindex(series.index)


def save_hmm_artifacts(hmm_model: GaussianHMM, scaler: StandardScaler, state_labels: dict, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "hmm_model.pkl", "wb") as f:
        pickle.dump(hmm_model, f)
    with open(path / "hmm_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    import json
    with open(path / "hmm_state_labels.json", "w") as f:
        json.dump({str(k): v for k, v in state_labels.items()}, f)
    logger.info(f"HMM artifacts saved to {path}")


def load_hmm_artifacts(path: Path) -> tuple:
    path = Path(path)
    with open(path / "hmm_model.pkl", "rb") as f:
        hmm_model = pickle.load(f)
    with open(path / "hmm_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    import json
    with open(path / "hmm_state_labels.json") as f:
        raw = json.load(f)
    state_labels = {int(k): v for k, v in raw.items()}
    logger.info(f"HMM artifacts loaded from {path}")
    return hmm_model, scaler, state_labels
