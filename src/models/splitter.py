import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator
from typing import Iterator


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
                mask = pd.Series(groups).values[:train_end] <= test_start_time
                train_idx = train_idx[mask]

            if self.val_start_date is not None and hasattr(X, "index"):
                assert X.index[test_end - 1] <= pd.Timestamp(self.val_start_date, tz="UTC"), (
                    f"CV fold bleeds into validation: fold end {X.index[test_end - 1]} > val_start {self.val_start_date}"
                )

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def _iter_test_indices(self, X=None, y=None, groups=None):
        # Required by BaseCrossValidator but logic is in split()
        pass


def compute_pbo(fold_sharpes: list) -> float:
    # Probability of Backtest Overfitting (Bailey et al. 2014)
    # Fraction of folds with below-median Sharpe
    if len(fold_sharpes) == 0:
        return 0.5
    median_sharpe = np.median(fold_sharpes)
    below = sum(1 for s in fold_sharpes if s < median_sharpe)
    return below / len(fold_sharpes)
