# CLAUDE.md — Session Instructions

## Session Startup Protocol
1. Read `project_state.json` ONLY at session start.
2. Read `DECISIONS.md` — check open issues and critical path before suggesting anything.
3. Read `SESSION_PROTOCOL.md` if starting a new investigation or retrain.
4. Do NOT read other files unless explicitly listed in `token_hints.read_if_relevant`.
5. Check `project_state.json` stages before re-running any stage — pipeline is resumable.
6. **DECISIONS.md is APPEND-ONLY.** Never edit existing entry text or status in-place.
   To mark fixed: add new entry `### ISSUE-XXX-FIX (date)` below the original with what changed.
   To reopen: add `### ISSUE-XXX-REOPEN (date)`. Never delete or overwrite old entries.
7. **Check Anti-Patterns section below before proposing any architectural change.** If a proposal matches an anti-pattern, it was already evaluated and rejected — do not re-propose without new evidence.

## Core Conventions
- **XGBoost is the PRIMARY model.** LightGBM is additive only, never a replacement.
- **No negative `.shift()` on features.** Negative shifts are only permitted on targets.
- **All times are UTC.** Naive timestamps are a bug.
- **All magic numbers belong in `config/base.yaml`.** No hardcoded constants in source files.
- **Use `loguru` not `print`.** Import logger from `src/utils/logger.py`.
- **No docstrings.** Use inline comments only for non-obvious logic.
- **Pipeline is resumable.** Always call `is_stage_complete(stage)` before re-running.

## End-of-Session Protocol
- Update `project_state.json` before exit.
- Record any failed symbols, new issues, and updated stage statuses.
- Update `next_scheduled_retrain` if a retrain was performed.

## End-of-Session Required Actions
At end of EVERY session, **append** to DECISIONS.md — do NOT edit existing entries.
- New fix applied this session: `### ISSUE-XXX-FIX (date)` — what changed, why, which files.
- New issue discovered: `### ISSUE-NNN (date)` — problem, location, status=NOT FIXED.
- This is mandatory and append-only. Treat it like a commit message — never amend history.

## Critical Leakage Rules
- StandardScaler: fit on TRAIN, save to disk, load for val/test transform.
- IterativeImputer: fit() on train X only, transform() val/test.
- Fracdiff d: estimate ADF on train series only, cache d values in JSON.
- HMM regime: fit GaussianHMM on train, soft-prob predict on val/test.
- DCC-GARCH: fit on train return data only.
- Macro forward-fill: ffill only, NEVER bfill. OECD shifted +1 period before merge.
- Meta-labeler: train on OOF predictions from cross_val_predict, NEVER in-sample.
- Signals: pre-compute ALL signals before backtest loop starts.

## Model Architecture (CURRENT — read before proposing any changes)
- **XGBoost PRIMARY per symbol.** LightGBM additive only, NEVER replacement.
- **Labels**: Triple-barrier {-1,0,+1}. Neutral (~55% of bars) dropped before training.
  CORRECT per Lopez de Prado — do not propose replacing with trend labels without ablation first.
- **Per-symbol model** (57 symbols). Cross-symbol full pooling NOT adopted — altcoin distributions
  not exchangeable. If revisited: cluster-pooling only (8-12 syms/cluster by HMM state + market cap).
- **HMM**: 3-state (bull/bear/sideways) after ablation from 4-state. Covariance=diag. Retrain every 6h live.
- **BOCPD** changepoint detection (ruptures) supplements HMM.
- **Meta-labeling**: meta_y = (primary_pred == actual), signal = primary_prob × meta_prob.
  scale_pos_weight = n_meta0/n_meta1 REQUIRED in meta-labeler (ISSUE-049-FIX).
  Dead-zone bars (|prob-0.5| < objective_dead_zone) excluded from meta_y — they are noise (ISSUE-050-FIX).
- **Minimum probability floor** = 0.55 (no forced trading rule). Dead zone = 0.05 from config.
- **scale_pos_weight** = n_short/n_long injected into XGBoost to handle class imbalance.
- **CVaR weight**: keep objective_cvar_weight ≤ 0.1. ~3000 samples → ~150 tail obs — auto-reduced
  in compute_objective when n_tail < 50.
- **Test period degradation**: primarily HMM regime mismatch in bear market, NOT model failure.
  Mitigation: hmm_retrain_hours=6 (already set). Full fix: retrain with updated train_end.
- **cross_sectional_stats.pkl** anchored to training distribution. Rank features may be stale in OOS
  bear markets. Monitor PSI on rank features before assuming model degradation.

## Train/Val/Test Split Rules
- **Never declare a DA metric good without checking class balance first.**
  - `pct_positive_train` must be in `training_summary.csv`. If missing, metric is unverified.
  - DA ≈ majority class rate = model has no signal.
- **Test period = last 3 months of available data. Val = 3 months before that. Train = rest.**
- **Update dates in `config/base.yaml` before each retrain:**
  - Current (as of 2026-04-03): train_end=2024-06-30, val_end=2024-09-30, test_start=2024-10-01 (STALE)
  - Recommended next: train_end=2025-09-30, val_end=2025-12-31, test_start=2026-01-01
- Backtest must filter signals to `>= test_start`. First trade in trade_log must be >= test_start.

## Live Trading (Stage 8)
- Mode: Binance Demo FAPI. Controlled by `cfg.trading.mode`. NEVER accidentally set to MAINNET.
- Dead-man-switch (DMS): cancels all positions if no heartbeat for `dead_man_switch_seconds` (60s).
  - DMS firing mid-sleep is expected behavior when process stops — NOT a bug.
  - **ISSUE-010 FIXED:** DMS was firing during 900s bar-wait sleep. Fixed via 30s heartbeat loop: the bar-wait sleep is broken into 30s chunks, calling `order_manager.heartbeat()` after each chunk. DMS (60s timeout) never fires during normal operation.
- stage_08 symbol handling: `_get_forecast_symbols` returns ALL symbols with a trained primary model. `_get_trade_limit` enforces the growth gate max open positions limit. Forecasts always run on all symbols; growth gate only restricts position opens.
- API keys loaded from `.env` via python-dotenv at stage_08 startup.
- Binance FAPI klines hard limit: 1500 bars per request.

## Growth Gate
- Tier 1 (equity <= $150): max_symbols=2, leverage_a_max=2
- Tier 2 (equity <= $300): max_symbols=3, leverage_a_max=3
- Tier 3 (equity <= $600): max_symbols=6, leverage_a_max=3
- `_get_trade_limit` reads tier from growth_gate.tiers in config based on current equity.

## Kaggle Pipeline Notes
- Stage 2 (features) has 3 modes: A=symlink dataset, B=compute from raw, C=upload local.
- Features split into 4 Kaggle datasets (~4.7GB each). Do NOT consolidate.
- `project_state.json` restored from checkpoints dataset for pipeline resume.

## Anti-Patterns — Do Not Re-Propose Without New Evidence + Checking DECISIONS.md
- **Replacing triple-barrier with trend labels** — not adopted, ablation required first
- **Full cross-symbol pooling** — not adopted, altcoin distributions not exchangeable
- **Replacing XGBoost with LightGBM as primary** — XGBoost is and stays primary
- **Fixed TP/SL pct instead of ATR-based** — reverted, ATR matches training label geometry
- **IterativeImputer(BayesianRidge)** — replaced with SimpleImputer(median) + missing indicators
- **Isotonic calibration** — replaced with sigmoid (Platt) scaling
- **Hardcoded dead zone in compute_objective** — now reads from config (objective_dead_zone)
- **meta_n_estimators < 300** — do not lower, was 10 then 100, now 300 final
- **stability_threshold < 0.75** — do not lower, was 0.60 then 0.70, now 0.75 final
- **4-state HMM with full covariance** — ablated to 3-state diag, do not revert without evidence
- **optuna_n_trials > 20** — diminishing returns, Calmar+CVaR objective converges faster

## File Paths
- Raw data: `data/raw/`
- Processed: `data/processed/`
- Features: `data/features/`
- Labels: `data/labels/`
- Models: `models/`
- Results: `results/`
- Checkpoints: `data/checkpoints/`
