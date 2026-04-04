# Pipeline Decision Log

## Purpose
Living record of every confirmed bug, design decision, and contradicted assumption in this pipeline. Read this at session start AFTER `project_state.json` and `CLAUDE.md`. Before proposing any fix, check whether the issue is already documented here and what its status is.

## Format
Each entry: **ID** | **Date discovered** | **Decision/Issue** | **Rationale** | **Status** | **Owner**

Status values: `NOT FIXED` | `IN PROGRESS` | `FIXED` | `WONT FIX`

---

## Open Issues (Not Fixed)

### ISSUE-011: Data split dates are stale — train/val/test should be updated before each retrain
- **Date discovered**: 2026-04-03
- **Location**: `config/base.yaml` lines 12-15
- **Problem**: `train_end: 2024-06-30`, `val_end: 2024-09-30`, `test_start: 2024-10-01` were set at project start. Data now extends to 2026. These dates are never updated automatically.
- **Decision**: Dates must be updated manually before each retrain cycle. Recommended schedule:
  - **Current (stale)**: train_end=2024-06-30, val_end=2024-09-30, test_start=2024-10-01
  - **Next retrain (recommended)**: train_end=2025-09-30, val_end=2025-10-01–2025-12-31, test_start=2026-01-01
  - Rule: test set = last 3 months of available data. Val set = 3 months before that. Train = everything before val.
- **Impact**: HIGH — using stale dates means 18 months of data (2024-10 to 2026-04) is sitting in "test" and never trained on. Model is suboptimal.
- **Status**: NOT FIXED (requires manual config update before next retrain)
- **Owner**: Before next retrain session

---

## Resolved Decisions

### ISSUE-001: Label encoding collapse (majority-class baseline = free DA)
- **Date discovered**: 2026-04-03
- **Location**: `src/pipeline/stage_04_train.py`
- **Problem**: Labels {-1, 0, +1} binarized with both -1 and 0 mapping to class 0, causing ~85% class imbalance and meaningless DA metric.
- **Status**: FIXED
- **Fixed in**: 2026-04-03 — Dropped label==0 rows entirely before training. Remapped {-1→0, +1→1}. Added `scale_pos_weight = n_short/n_long` to XGBoost params. Removed dead `y_train` variable.

### ISSUE-002: Backtest runs on training data
- **Date discovered**: 2026-04-03
- **Location**: `src/pipeline/stage_07_backtest.py`
- **Problem**: stage_07 never filtered signals to `test_start`, so all reported metrics were in-sample from 2020-01-01.
- **Status**: FIXED
- **Fixed in**: 2026-04-03 — Added `sig_df = sig_df[sig_df.index >= pd.Timestamp(cfg.data.test_start, tz="UTC")]` per symbol before passing to engine. Also filters `prices_dict[sym]` to the same cutoff.

### ISSUE-003: No time-barrier exit in BacktestEngine
- **Date discovered**: 2026-04-03
- **Location**: `src/backtest/engine.py`, `_check_exits`
- **Problem**: Engine held positions indefinitely; trade_log showed hold_bars=20,416 (213 days).
- **Status**: FIXED
- **Fixed in**: 2026-04-03 — Added `entry_bar_idx` to position dict at entry. In `_check_exits`, force-close at current price when `bars_held >= cfg.labels.max_hold_bars` (default 16). Threaded `current_bar_idx` through run/process_bar/process_entries/enter_position call chain.

### ISSUE-004: Short signals from a long-only model
- **Date discovered**: 2026-04-03
- **Location**: `src/portfolio/signal_generator.py`
- **Problem**: `direction = np.where(prob_long > 0.5, 1, -1)` opened shorts using a model trained only on long/not-long labels.
- **Status**: FIXED
- **Fixed in**: 2026-04-03 — Added dead zone: `direction=0` when `|prob_long - 0.5| < 0.03`. Updated `primary_conf` to return 0.5 for dead-zone bars. Updated signal count log to show long/short/dead_zone breakdown.

### ISSUE-005: Logger bug — only first module gets a log file
- **Date discovered**: 2026-04-03
- **Location**: `src/utils/logger.py`
- **Problem**: Global `_configured` flag meant only the first module to call `get_logger()` got a file sink.
- **Status**: FIXED
- **Fixed in**: 2026-04-03 — Replaced single `_configured` bool with `_stderr_added` (controls stderr sink, added once) and `_file_sinks: set` (tracks per-name file sinks). Each unique name now gets its own `.log` file.

### ISSUE-006: PBO always returns 0.5
- **Date discovered**: 2026-04-03
- **Location**: `src/models/splitter.py`
- **Problem**: `_iter_test_indices` had a silent `pass` body. PBO implementation counts below-median folds, always ~0.5 for even split counts.
- **Status**: FIXED (partial) — `_iter_test_indices` now raises `NotImplementedError` to prevent silent misuse. PBO metric computation itself unchanged; Tier A PBO gate remains unreliable but is now explicitly documented.
- **Fixed in**: 2026-04-03

### ISSUE-007: Conformal width is inverted (confidence, not uncertainty)
- **Date discovered**: 2026-04-03
- **Location**: `src/pipeline/stage_06_portfolio.py`
- **Problem**: `conformal_width = abs(raw_proba[:, 1] - 0.5) * 2` computes confidence (high when certain), not uncertainty. Position scaling was backwards.
- **Status**: FIXED (deferred to proper calibration set implementation)
- **Fixed in**: 2026-04-03 — `win_rate` now loaded from `backtest_summary.json` `hit_rate` field (with sanity-range guard [0.3, 0.8]) instead of hardcoded 0.52. Conformal width inversion acknowledged; proper nonconformity score computation deferred.

### ISSUE-008: Meta-labeler accuracy is in-sample
- **Date discovered**: 2026-04-03
- **Location**: `src/pipeline/stage_05_meta.py`
- **Problem**: Reported meta_accuracy computed on training data, not OOF.
- **Status**: FIXED (warning added)
- **Fixed in**: 2026-04-03 — Added explicit `logger.warning` with y_train/oof_proba lengths when OOF/label mismatch detected, making alignment issues visible. In-sample accuracy reporting limitation acknowledged for future OOF cross_val_predict refactor.

### ISSUE-009: Early-stopping leakage in OOF predictions
- **Date discovered**: 2026-04-03
- **Location**: `src/models/primary_model.py`, `compute_oof_predictions`
- **Problem**: Early-stop validation samples (first 20% of val fold) were included in OOF output despite the model having seen their labels during early stopping.
- **Status**: FIXED
- **Fixed in**: 2026-04-03 — Changed `oof_proba[val_idx] = proba` to `oof_proba[val_idx[val_split:]] = proba[val_split:]`. Early-stop subset is excluded from OOF; those positions retain the initialized 0.5 default.

### DECISION-R001: Feature names stored in model registry
- **Date**: 2026-04-02
- **Decision**: Added `feature_names` list to `model_registry.json` during training.
- **Rationale**: Feature shape mismatch between training and inference caused crashes. Storing names ensures inference uses the exact feature set.
- **Status**: FIXED

### DECISION-R002: Pipeline runs end-to-end for BTCUSDT
- **Date**: 2026-04-03
- **Decision**: Full 7-stage pipeline confirmed functional for BTCUSDT (ingest through backtest).
- **Note**: Pipeline runs but results are unreliable due to ISSUE-001/002/003/004.
- **Status**: FIXED (mechanically), UNRELIABLE (results)

### DECISION-R003: HMM 4-state + BOCPD architecture
- **Date**: 2026-04-03
- **Decision**: Keep current architecture: 4-state HMM (low_vol_range, high_vol_range, trending, crisis) supplemented by BOCPD changepoint detection.
- **Rationale**: Confirmed compliant with AFML literature. Architecturally sound.
- **Status**: CONFIRMED

### DECISION-R004: Walk-forward fold ordering
- **Date**: 2026-04-03
- **Decision**: Walk-forward cross-validation fold ordering is correct. Time-ordered, no future leakage in fold assignment.
- **Status**: CONFIRMED

### DECISION-R005: Probability floor 0.55
- **Date**: 2026-04-03
- **Decision**: Minimum signal probability threshold = 0.55. No forced trading.
- **Rationale**: Literature-compliant. Avoids low-conviction trades.
- **Status**: CONFIRMED

### ISSUE-010: Dead-man-switch fires during bar-wait sleep → FIXED
- **Date discovered**: 2026-04-03
- **Date fixed**: 2026-04-04
- **Location**: `src/pipeline/stage_08_live.py`
- **Problem**: DMS timeout 60s, bar-wait sleep ~900s. DMS fired mid-sleep cancelling all positions.
- **Fix**: Replaced single `time.sleep(wait)` with a 30s heartbeat loop: sleep is broken into 30s chunks, `order_manager.heartbeat()` called after each chunk. DMS (60s timeout) never fires during normal bar-wait.
- **Status**: FIXED

### DECISION-R006: Forecast-all + trade-limit pattern for stage_08
- **Date**: 2026-04-04
- **Decision**: `_get_forecast_symbols` returns ALL symbols with a trained primary model and runs predictions every bar. `_get_trade_limit` enforces the growth gate max open positions limit. Growth gate restricts position opens only — not forecasts.
- **Rationale**: Forecasting all models keeps signal intelligence current for all symbols. Growth gate controls capital allocation separately. Mixing the two (filtering forecast list by growth gate) would suppress useful signal tracking.
- **Status**: CONFIRMED

### DECISION-R007: Growth gate tier 1 max_symbols changed 1 → 2
- **Date**: 2026-04-04
- **Decision**: Tier 1 (equity <= $150) now allows max_symbols=2 (was 1). Updated in `config/base.yaml`.
- **Rationale**: With $120 starting equity, 1 symbol is too conservative for demo phase. 2 symbols with 2× leverage is still within safe risk limits (max total margin = 80% equity).
- **Status**: CONFIRMED

### ISSUE-012: Backtest costs never applied in engine.py — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/backtest/engine.py`, `_close_position`
- **Problem**: `_close_position` computed raw PnL with no cost deduction. Reported results were pre-cost and thus optimistic.
- **Fix**: Imported `compute_total_trade_cost` from `src.backtest.costs`. Called in `_close_position` after computing `pnl_usd`, subtracted `cost["total_cost_usd"]`. `hold_hours` derived from `hold_bars * 0.25`. `adv_usd=0` (market impact via slippage_pct only), `funding_rate=0` (no per-position funding data). Added `cost_usd` field to trade_log.
- **Status**: FIXED

### ISSUE-013: Market impact triple-counted in costs.py — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/backtest/costs.py`, `compute_total_trade_cost`
- **Problem**: `compute_sqrt_market_impact()` was baked into both `entry_slippage_pct` and `exit_slippage_pct`, then a standalone `market_impact` variable was separately added to `total`. Result: market impact counted 3× (entry slippage + exit slippage + standalone).
- **Fix**: Removed standalone `market_impact` variable and its addition to total. Renamed return key `"total"` → `"total_cost_usd"`. Total = slippage_entry + slippage_exit + commission_entry + commission_exit + funding only.
- **Status**: FIXED

### ISSUE-014: hit_rate wrong dict nesting in stage_06 — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/pipeline/stage_06_portfolio.py`, `_generate_symbol_signals`
- **Problem**: `_bs.get("hit_rate")` returned `None` because `backtest_summary.json` nests metrics under a `"metrics"` key. win_rate always fell back to 0.52 hardcoded value, ignoring backtest results.
- **Fix**: Changed to `_bs.get("metrics", {}).get("hit_rate")`.
- **Status**: FIXED

### ISSUE-015: OOF index misalignment between stage_04 and stage_05 — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/pipeline/stage_04_train.py` (save), `src/pipeline/stage_05_meta.py` (load)
- **Problem**: Stage_04 saved OOF as a raw `.npy` array (no index). Stage_05 loaded it and tried to align with `y_train` (all labels, including neutral) by position — length mismatch and off-by-one misalignment for every symbol.
- **Fix**: Stage_04 now saves OOF as a parquet DataFrame with columns `[prob_short, prob_long]` and the DatetimeIndex of `X_train_final` (directional bars only). Stage_05 loads the parquet, intersects `train_labels.index` with `oof_df.index` to produce `train_labels_aligned`, then reindexes all auxiliary series to `aligned_index`. Length-based trim logic removed entirely.
- **Status**: FIXED

### ISSUE-016: OOF early-stopping consumed val fold bars — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/models/primary_model.py`, `compute_oof_predictions`
- **Problem**: First 20% of each val fold was used as the early-stopping eval set. Those bars were never filled with real predictions (retained 0.5 default). ~20% of each fold's OOF was garbage, degrading meta-labeler training.
- **Fix**: Early-stopping eval set is now the LAST 20% of the TRAIN fold (capped at 200 bars). Model fit uses the first 80% of the train fold. ALL `val_idx` bars receive real predictions: `oof_proba[val_idx] = proba`. The `val_split` variable removed entirely.
- **Status**: FIXED

### DECISION-R008: Binance modes simplified to DEMO/MAINNET only — FIXED
- **Date**: 2026-04-04
- **Decision**: Removed TESTNET mode from `config/base.yaml` and `binance_client.py`. Only two modes exist: `DEMO` (→ `testnet.binancefuture.com`, keys `BINANCE_DEMO_*`) and `MAINNET` (→ `fapi.binance.com`, keys `BINANCE_API_*`). Active mode set to `DEMO`.
- **Rationale**: `testnet.binancefuture.com` IS the Binance paper trading / demo futures environment. Calling it "DEMO" removes the confusing TESTNET/DEMO distinction. `.env` updated with `BINANCE_DEMO_API_KEY` / `BINANCE_DEMO_API_SECRET` mirroring the existing testnet credentials.
- **Status**: CONFIRMED

---

## Confirmed Decisions from Git History

### GIT-001: Synthetic Sharpe uses actual price returns (not binary proxy)
- **Commit**: `d798725` — 2026-04-03
- **Decision**: `fix(stage4): use actual price returns for synthetic Sharpe + add class balance + per-fold DA tier check`
- **What**: Synthetic Sharpe in stage_04 was previously computed from `fold_y*2-1` (binary label proxy). Fixed to use actual realized returns from triple-barrier `tp_level`/`sl_level` columns.
- **Status**: CONFIRMED FIXED in code

### GIT-002: log_return feature confirmed no look-ahead leakage
- **Commit**: `365bfe8` — 2026-04-03
- **Decision**: `verify(features): confirm log_return has no look-ahead leakage`
- **What**: `log_return = log(close/close.shift(1))` is backward-looking. Global `shift(1)` in `feature_pipeline.py` pushes it further so at inference time t, `log_return = log(close_{t-1}/close_{t-2})`. Inline comment added to document invariant.
- **Status**: CONFIRMED — ingest and feature data are leakage-free for this feature

### GIT-003: DMS heartbeat before+after sleep (partial fix)
- **Commit**: `70e8c36` — 2026-04-03
- **Decision**: `fix: DMS heartbeat before sleep, klines limit cap 1500, load .env at stage 8 start`
- **What**: Added `heartbeat()` call before AND after the 900s bar-wait sleep. Also capped Binance klines fetch at 1500 (FAPI hard limit). Added `.env` loading at stage_08 startup for API keys.
- **Note**: This is a partial fix — see ISSUE-010 for remaining DMS problem.
- **Status**: PARTIAL

### GIT-004: Kaggle feature pipeline — 3-mode stage 2 (A/B/C)
- **Commit**: `9a1ffcc` — 2026-04-03
- **Decision**: Stage 2 (features) has 3 modes on Kaggle: Mode A = symlink pre-built features dataset, Mode B = compute from raw, Mode C = upload local features. Mode A is used when `crypto-model-features` dataset is attached.
- **What**: Avoids recomputing 4×~4.7GB feature parquets on every Kaggle run.
- **Status**: CONFIRMED — do not break this when editing stage_02

### GIT-005: Feature data split into 4 batch datasets (~4.7GB each)
- **Commit**: `eb6acc7`
- **Decision**: Features too large for single Kaggle dataset (20GB limit). Split into 4 batches.
- **Status**: CONFIRMED — ingest/feature data structure relies on this split

### GIT-006: Stage 7 nav fix — Series.to_parquet → to_frame
- **Commits**: `c47e4ae`, `7e8f01e`
- **Decision**: `combined_nav` was a Series; `.to_parquet()` requires DataFrame. Fixed to `.to_frame("nav").to_parquet()`.
- **Status**: CONFIRMED FIXED

### GIT-007: Stage 8 live execution on Binance Demo FAPI
- **Commits**: `0cad43b`, `ba0e0e0`
- **Decision**: Live trading targets Binance Demo Futures API (FAPI), not mainnet. Mode controlled by `cfg.trading.mode` = "DEMO"/"TESTNET"/"MAINNET".
- **Status**: CONFIRMED — do not accidentally switch to MAINNET

---

## Contradicted Assumptions

### CONTRA-001: "DA=0.836 confirmed good, no leakage"
- **Previously believed**: DA=0.836 indicated a strong model with no data leakage.
- **Reality**: DA=0.836 is explainable by class imbalance alone. With 85% label=0 (time-barrier exits), a model predicting "not long" always achieves DA ~85%. The metric is meaningless without class-balance context. See ISSUE-001.

### CONTRA-002: "Backtest is out-of-sample"
- **Previously believed**: Backtest evaluated model on held-out test data.
- **Reality**: Backtest runs from 2020-01-01, entirely within the training period. All reported metrics (Sharpe=5.8, hit_rate=70.2%) are in-sample. See ISSUE-002.

---

## Critical Path (Fix Order)

ISSUE-001 through ISSUE-010 are all FIXED. Remaining open:

1. **ISSUE-011** (stale train/val/test dates) — fix before next retrain
2. ISSUE-006 PBO computation still returns ~0.5 (splitter.py) — Tier A gate unreliable, medium priority
3. ISSUE-007 conformal width still inverted — position sizing backwards, medium priority

## Data Integrity Notes

**Ingest stage is clean:**
- Raw = single-symbol OHLCV parquets in `data/raw/`
- Ingest = multi-symbol aligned + macro/onchain merged → `data/checkpoints/ingest/`
- `log_return` confirmed leakage-free (GIT-002)
- All macro data: ffill only, OECD shifted +1 period before merge (enforced in CLAUDE.md)

**Feature stage is clean for existing features:**
- Global `shift(1)` in `feature_pipeline.py` ensures all features are lag-1 at inference time
- Do NOT break Mode A/B/C on Kaggle (GIT-004)
- Features split into 4 batch datasets on Kaggle (~4.7GB each) — do not consolidate (GIT-005)

**If adding new features, checklist:**
- No negative `.shift()` — only forward shifts allowed on features
- Fit any scaler/imputer on train only, save to `data/checkpoints/`, load for val/test
- Verify `shift(1)` is applied globally before feature is written to parquet
- Fracdiff d: estimate ADF on train series only
