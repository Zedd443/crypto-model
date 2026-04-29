import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class PurgedTimeSeriesSplit(BaseCrossValidator):
    def __init__(self, n_splits=8, embargo_pct=0.01, embargo_bars_min=50, val_start_date=None):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
        self.embargo_bars_min = embargo_bars_min
        self.val_start_date = val_start_date

    def split(self, X, y=None, groups=None):
        # groups should be the t1 series (barrier end times)
        n = len(X)
        indices = np.arange(n)
        test_size = n // (self.n_splits + 1)
        embargo = max(int(n * self.embargo_pct), self.embargo_bars_min)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size

            # embargo gap between train end and test start
            train_end = test_start - embargo
            if train_end <= 0:
                continue

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            # purge: remove train samples where t1 > test_start_time
            if groups is not None and hasattr(X, "index"):
                test_start_time = X.index[test_start]
                g_series = pd.Series(groups).values[:train_end]
                g_vals = pd.DatetimeIndex(
                    pd.to_datetime(g_series, utc=True, errors="coerce")
                )
                # Normalize test_start_time to UTC for consistent comparison
                if test_start_time.tzinfo is None:
                    test_start_time = pd.Timestamp(test_start_time, tz="UTC")
                else:
                    test_start_time = test_start_time.tz_convert("UTC")
                # NaT barrier ends (failed label events) are treated as no-overlap — keep in train
                mask = (g_vals <= test_start_time) | g_vals.isna()
                train_idx = train_idx[mask]

            if self.val_start_date is not None and hasattr(X, "index"):
                assert X.index[test_end - 1] <= pd.Timestamp(self.val_start_date, tz="UTC"), (
                    f"CV fold bleeds into validation: fold end {X.index[test_end - 1]} > val_start {self.val_start_date}"
                )

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        raise NotImplementedError("Use split() directly")


def compute_pbo_cscv(trial_fold_scores: list[list[float]]) -> float:
    # Probability of Backtest Overfitting — Bailey, Borwein, Lopez de Prado, Zhu (2014).
    # Requires a (K_trials × N_folds) matrix from Optuna trial results.
    # Algorithm:
    #   For each C(N, N/2) split of folds into IS and OOS halves:
    #     1. Rank all K trials by mean IS score → pick best IS trial
    #     2. Compute logit of that trial's OOS rank relative to all K trials
    #   PBO = fraction of splits where best-IS trial has below-median OOS performance (logit < 0)
    # Returns 0.5 (maximum uncertainty) when insufficient data.
    if len(trial_fold_scores) < 2:
        return 0.5
    matrix = np.array(trial_fold_scores, dtype=float)  # shape (K, N)
    K, N = matrix.shape
    if N < 2:
        return 0.5

    from itertools import combinations
    n_is = N // 2
    all_fold_indices = list(range(N))
    splits = list(combinations(all_fold_indices, n_is))
    # Cap at 256 splits to keep runtime bounded (C(8,4)=70, C(16,8)=12870 — cap matters for large N)
    if len(splits) > 256:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(splits), size=256, replace=False)
        splits = [splits[i] for i in idx]

    logit_oos_ranks = []
    for is_folds in splits:
        oos_folds = [f for f in all_fold_indices if f not in is_folds]
        is_scores = matrix[:, is_folds].mean(axis=1)   # shape (K,)
        oos_scores = matrix[:, oos_folds].mean(axis=1) # shape (K,)
        best_is_trial = int(np.argmax(is_scores))
        # OOS rank: fraction of trials with OOS score <= best_is_trial's OOS score
        oos_rank = float(np.mean(oos_scores <= oos_scores[best_is_trial]))
        # Logit of rank — negative means below-median OOS performance
        oos_rank = np.clip(oos_rank, 1e-6, 1 - 1e-6)
        logit_oos_ranks.append(np.log(oos_rank / (1 - oos_rank)))

    pbo = float(np.mean([l < 0 for l in logit_oos_ranks]))
    return pbo


def compute_fold_consistency(fold_sharpes: list) -> float:
    # Fraction of folds with positive Sharpe. Range [0,1], higher=better.
    if not fold_sharpes:
        return 0.0
    return sum(1 for s in fold_sharpes if s > 0) / len(fold_sharpes)
