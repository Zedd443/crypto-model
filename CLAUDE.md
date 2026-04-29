# CLAUDE.md — Session Instructions

## Session Startup Protocol
1. Read `project_state.json`only at session start and when needed.
2. Do NOT read other files unless explicitly listed in `token_hints.read_if_relevant`.
3. Check `project_state.json` stages before re-running any stage — pipeline is resumable.
4. read `SESSION_PROTOCOL.md` to know what we at.

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
- Update `DECISION.md` before exit.
- Record any failed symbols, new issues, and updated stage statuses.
- Update `next_scheduled_retrain` if a retrain was performed.

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

## File Paths
- Raw data: `data/raw/`
- Processed: `data/processed/`
- Features: `data/features/`
- Labels: `data/labels/`
- Models: `models/`
- Results: `results/`
- Checkpoints: `data/checkpoints/`
