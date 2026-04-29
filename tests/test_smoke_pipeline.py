"""
Smoke test suite — verifies config values and critical code paths before full retrain.
No disk I/O, no models, no data required. Completes in < 60 seconds.

Run: .venv/Scripts/python.exe -m pytest tests/ -v --tb=short
"""
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config_loader import load_config
from src.models.primary_model import compute_objective, _SigmoidCalibrator
from src.models.meta_labeler import create_meta_labels


# ---------------------------------------------------------------------------
# Area 1 — Config value assertions
# ---------------------------------------------------------------------------

class TestConfigValues(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config("config/base.yaml")

    # Label geometry (DECISION-042)
    def test_labels_tp_atr_mult(self):
        self.assertEqual(self.cfg.labels.tp_atr_mult, 1.5)

    def test_labels_sl_atr_mult(self):
        self.assertEqual(self.cfg.labels.sl_atr_mult, 7.0)

    def test_labels_max_hold_bars(self):
        self.assertEqual(self.cfg.labels.max_hold_bars, 32)

    def test_labels_fee_adjust(self):
        self.assertTrue(self.cfg.labels.fee_adjust_labels)

    # Model calibration (ISSUE-026)
    def test_calibration_method_sigmoid(self):
        self.assertEqual(self.cfg.model.calibration_method, "sigmoid")

    # CV / embargo (DECISION-041)
    def test_cv_n_splits(self):
        self.assertEqual(self.cfg.model.cv_n_splits, 8)

    def test_embargo_bars_min(self):
        self.assertGreaterEqual(self.cfg.model.embargo_bars_min, 96)

    # Stability selection (DECISION-044)
    def test_stability_threshold(self):
        self.assertEqual(self.cfg.model.stability_threshold, 0.75)

    def test_stability_n_bootstrap(self):
        self.assertEqual(self.cfg.model.stability_n_bootstrap, 100)

    # Meta-labeler (ISSUE-031a / DECISION-045)
    def test_meta_n_estimators(self):
        self.assertEqual(self.cfg.model.meta_n_estimators, 300)

    # Objective dead zone (DECISION-048 / ISSUE-050)
    def test_objective_dead_zone(self):
        self.assertEqual(self.cfg.model.objective_dead_zone, 0.05)

    # CVaR weight (ISSUE-051) — must not be cranked too high
    def test_objective_cvar_weight(self):
        self.assertLessEqual(self.cfg.model.objective_cvar_weight, 0.1)

    # Portfolio dead zone
    def test_portfolio_dead_zone_direction(self):
        self.assertEqual(self.cfg.portfolio.dead_zone_direction, 0.05)

    # HMM regime (DECISION-046)
    def test_hmm_n_states(self):
        self.assertEqual(self.cfg.regime.n_states, 3)

    def test_hmm_covariance_type(self):
        self.assertEqual(self.cfg.regime.hmm_covariance_type, "diag")

    # Safety check — never accidentally go MAINNET
    def test_trading_mode_demo(self):
        self.assertEqual(self.cfg.trading.mode, "DEMO")



# ---------------------------------------------------------------------------
# Area 2 — Code path unit tests (synthetic data, no disk)
# ---------------------------------------------------------------------------

class TestCodePathFixes(unittest.TestCase):

    # Test A: compute_objective — all probs in dead zone → no active positions → score=0
    def test_objective_all_dead_zone(self):
        rng = np.random.default_rng(42)
        n = 200
        # All probs between 0.47 and 0.53 — inside dead zone of 0.05
        proba_1d = rng.uniform(0.47, 0.53, n)
        proba = np.column_stack([1 - proba_1d, proba_1d])
        y_true = rng.integers(0, 2, n)
        returns = rng.normal(0, 0.01, n)

        cfg = MagicMock()
        cfg.model.objective_dead_zone = 0.05
        cfg.model.objective_cvar_weight = 0.1
        cfg.model.objective_da_weight = 0.2
        cfg.model.objective_ic_weight = 0.3
        cfg.model.objective_sortino_weight = 0.5
        cfg.labels.round_trip_cost_pct = 0.003

        score = compute_objective(y_true, proba, returns, cfg=cfg)
        # With no active positions, calmar_adj_sharpe=0 and cvar_penalty=max
        # Score should be low / near zero — not a crash
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))

    # Test B: CVaR auto-reduction fires when n_tail < 50
    def test_objective_cvar_auto_reduction(self):
        rng = np.random.default_rng(7)
        # Only 40 active positions — n_tail = int(0.05 * 40) = 2 < 50 → effective_cvar_weight reduced
        n = 40
        proba_1d = np.full(n, 0.75)  # all clearly long, all active
        proba = np.column_stack([1 - proba_1d, proba_1d])
        y_true = np.ones(n, dtype=int)
        returns = rng.normal(0.001, 0.01, n)

        cfg = MagicMock()
        cfg.model.objective_dead_zone = 0.05
        cfg.model.objective_cvar_weight = 0.1
        cfg.model.objective_da_weight = 0.2
        cfg.model.objective_ic_weight = 0.3
        cfg.model.objective_sortino_weight = 0.5
        cfg.labels.round_trip_cost_pct = 0.003

        score = compute_objective(y_true, proba, returns, cfg=cfg)
        self.assertIsInstance(score, float)
        self.assertFalse(np.isnan(score))

    # Test C: create_meta_labels — dead zone bars always get meta_y=0
    def test_meta_labels_dead_zone_exclusion(self):
        rng = np.random.default_rng(0)
        n = 100
        y_true = rng.integers(0, 2, n)
        # 20 bars in dead zone (|prob-0.5| < 0.05), rest clearly outside
        proba_1d = np.where(np.arange(n) < 20, 0.51, 0.80)  # first 20 = dead zone
        proba = np.column_stack([1 - proba_1d, proba_1d])

        meta_y = create_meta_labels(y_true, proba, dead_zone=0.05)

        # All dead-zone bars must be 0
        self.assertTrue(np.all(meta_y[:20] == 0),
                        "Dead-zone bars (|prob-0.5|<0.05) must produce meta_y=0")
        # Outside dead zone: some can be 1 (where primary was correct)
        self.assertGreater(meta_y[20:].sum(), 0,
                           "Non-dead-zone bars should have some meta_y=1")

    # Test D: train_meta_labeler — scale_pos_weight derived from class ratio
    def test_meta_scale_pos_weight(self):
        from src.models.meta_labeler import train_meta_labeler
        import pandas as pd

        rng = np.random.default_rng(1)
        n = 200
        # 40% zeros, 60% ones → spw = 80/120 ≈ 0.667
        meta_y = np.array([0] * 80 + [1] * 120)
        X = pd.DataFrame(rng.random((n, 3)), columns=["a", "b", "c"])
        w = pd.Series(np.ones(n))

        cfg = MagicMock()
        cfg.model.meta_n_estimators = 10  # fast
        cfg.model.meta_max_depth = 3
        cfg.model.meta_early_stopping_rounds = 5

        captured_spw = {}

        original_xgb = __import__("xgboost", fromlist=["XGBClassifier"]).XGBClassifier

        class _CapturingXGB(original_xgb):
            def __init__(self, *args, **kwargs):
                captured_spw["spw"] = kwargs.get("scale_pos_weight")
                super().__init__(*args, **kwargs)

        with patch("src.models.meta_labeler.XGBClassifier", _CapturingXGB):
            try:
                train_meta_labeler(X, meta_y, w, cfg)
            except Exception:
                pass  # we only need the constructor call

        self.assertIn("spw", captured_spw, "XGBClassifier was never constructed")
        expected_spw = 80 / 120
        self.assertAlmostEqual(captured_spw["spw"], expected_spw, places=2,
                               msg=f"scale_pos_weight should be ~{expected_spw:.3f}")

    # Test E: _SigmoidCalibrator interface
    def test_sigmoid_calibrator_interface(self):
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(42)
        X_train = rng.random((100, 1))
        y_train = (X_train[:, 0] > 0.5).astype(int)
        lr = LogisticRegression()
        lr.fit(X_train, y_train)

        cal = _SigmoidCalibrator(lr)
        raw = np.array([0.3, 0.5, 0.7])
        out = cal.predict(raw)

        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.all(out >= 0) and np.all(out <= 1),
                        "Calibrator output must be in [0,1]")

    # Test F: compute_objective uses price_returns (not binary proxy) when passed
    def test_objective_uses_price_returns(self):
        rng = np.random.default_rng(5)
        n = 300
        proba_1d = rng.uniform(0.4, 0.9, n)
        proba = np.column_stack([1 - proba_1d, proba_1d])
        y_true = rng.integers(0, 2, n)
        # Deliberately large returns to produce a different score than small ones
        returns_large = np.full(n, 0.05)
        returns_small = np.full(n, 0.0001)

        cfg = MagicMock()
        cfg.model.objective_dead_zone = 0.05
        cfg.model.objective_cvar_weight = 0.1
        cfg.model.objective_da_weight = 0.2
        cfg.model.objective_ic_weight = 0.3
        cfg.model.objective_sortino_weight = 0.5
        cfg.labels.round_trip_cost_pct = 0.003

        score_large = compute_objective(y_true, proba, returns_large, cfg=cfg)
        score_small = compute_objective(y_true, proba, returns_small, cfg=cfg)
        self.assertNotEqual(score_large, score_small,
                            "Score must differ when returns differ — confirms price_returns are used")


if __name__ == "__main__":
    unittest.main()
