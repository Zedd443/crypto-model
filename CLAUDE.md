# CLAUDE.md — Session Instructions

## Session Startup Protocol
1. Read `project_state.json` ONLY at session start.
2. Read `DECISIONS.md` — check open issues and critical path before suggesting anything.
3. Read `SESSION_PROTOCOL.md` if starting a new investigation or retrain.
4. Do NOT read other files unless explicitly listed in `token_hints.read_if_relevant`.
5. Check `project_state.json` stages before re-running any stage — pipeline is resumable.

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
At end of EVERY session, update DECISIONS.md with any new decisions, fixes, or changes made. Format: add to "Resolved Decisions" if fixed, "Open Issues" if new. This is mandatory — treat it like a commit message.

## Critical Leakage Rules
- StandardScaler: fit on TRAIN, save to disk, load for val/test transform.
- IterativeImputer: fit() on train X only, transform() val/test.
- Fracdiff d: estimate ADF on train series only, cache d values in JSON.
- HMM regime: fit GaussianHMM on train, soft-prob predict on val/test.
- DCC-GARCH: fit on train return data only.
- Macro forward-fill: ffill only, NEVER bfill. OECD shifted +1 period before merge.
- Meta-labeler: train on OOF predictions from cross_val_predict, NEVER in-sample.
- Signals: pre-compute ALL signals before backtest loop starts.

## Model Architecture
- 4-state HMM: low_vol_range, high_vol_range, trending, crisis
- BOCPD changepoint detection (ruptures) supplements HMM
- Meta-labeling: meta_y = (primary_pred == actual), signal = primary_prob × meta_prob
- Minimum probability floor = 0.55 (no forced trading rule)
- Labels: {-1, 0, +1} — neutral (0) dropped before training, {-1→0 short, +1→1 long}
- scale_pos_weight = n_short/n_long injected into XGBoost to handle class imbalance

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

## File Paths
- Raw data: `data/raw/`
- Processed: `data/processed/`
- Features: `data/features/`
- Labels: `data/labels/`
- Models: `models/`
- Results: `results/`
- Checkpoints: `data/checkpoints/`
