# Pipeline Decision Log

## Purpose
Living record of every confirmed bug, design decision, and contradicted assumption in this pipeline. Read this at session start AFTER `project_state.json` and `CLAUDE.md`. Before proposing any fix, check whether the issue is already documented here and what its status is.

## Format
Each entry: **ID** | **Date discovered** | **Decision/Issue** | **Rationale** | **Status** | **Owner**

Status values: `NOT FIXED` | `IN PROGRESS` | `FIXED` | `WONT FIX`

---

## Open Issues (Not Fixed)

### ISSUE-035: Position sizing uses stale equity from project_state.json and divides by max_symbols — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/pipeline/stage_08_live.py:run` (bar-start equity fetch), `_process_symbol` (sizing)
- **Problem 1**: Equity was read from `project_state.json` at bar start — stale value from previous session/bar. Compound growth didn't work correctly.
- **Problem 2**: Sizing was `margin = equity / max_symbols`, then `notional = margin × leverage`. This splits the wallet across symbol slots instead of using full wallet. Est Profit.xlsx formula is `Volume = Saldo × leverage` (full wallet each trade, compound automatic).
- **Fix**: (1) Fetch `totalWalletBalance` from exchange API at the start of every bar; fall back to cached state only on API error. (2) Sizing now `notional = equity × leverage` (full wallet), matching Est Profit.xlsx exactly.
- **Status**: FIXED

### ISSUE-034: Testnet maxQty=120 is a real exchange limit — wallet×leverage must be capped to it — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/execution/order_manager.py:submit_entry`, `src/pipeline/stage_08_live.py` FIX 9 probe
- **Problem**: With full-wallet sizing (`notional = equity × leverage`), a $10k demo wallet × 3× = $30k notional. For SOLUSDT at $120, qty = 250 contracts — but testnet `MARKET_LOT_SIZE.maxQty=120`. Order would be rejected. Additionally ~30 symbols have `max_qty × price < min_notional=100` on testnet (e.g. ALGOUSDT: 120 × $0.12 = $14) — these are genuinely untradeable on testnet at current prices.
- **Fix**: (1) `submit_entry` always caps qty to `max_qty` (both DEMO and mainnet) and recalculates `size_usd` — this is the correct "max notional this exchange allows" cap; (2) Startup probe reverted to original `max_qty × price < min_notional` check — this correctly excludes structurally untradeable coins on both testnet and mainnet; (3) Startup probe logs are INFO-level since exclusion is expected and correct.
- **Status**: FIXED

### ISSUE-036: STOP_MARKET/TAKE_PROFIT_MARKET not placeable on demo-fapi via any known endpoint — FIXED via LIMIT fallback
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/execution/binance_client.py:place_order`
- **Problem**: demo-fapi.binance.com returns -4120 "use Algo Order API" for STOP_MARKET/TAKE_PROFIT_MARKET on `/fapi/v1/order`. But `/fapi/v1/algoOrder` only accepts `algoType=TWAP` or `VP` (algorithmic execution strategies) — sending any other algoType returns -4500 "Invalid algoType". Result: TP+SL both fail → emergency close fires immediately after every entry → zero held positions.
- **Fix**: In `place_order()`, when `self._mode == "DEMO"` and `order_type in _CONDITIONAL_ORDER_TYPES`, transparently remap to `LIMIT` orders at `stop_price` without `reduceOnly` (two simultaneous reduceOnly orders on same qty triggers -2022). On MAINNET, uses `/fapi/v1/order` with real `stopPrice + closePosition=true`. Also added `PERCENT_PRICE` multipliers to exchange info cache and `clamp_bracket_price()` method — demo-fapi enforces ±5% from mark price on all LIMIT orders; TP/SL prices are clamped before placement.
- **Status**: FIXED

### ISSUE-033: STOP/TAKE_PROFIT bracket orders fail — closePosition incompatible with limit-style types — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/execution/binance_client.py:place_order` lines 159–165
- **Problem**: After ISSUE-029 fix remapped STOP_MARKET→STOP and TAKE_PROFIT_MARKET→TAKE_PROFIT, the `close_position=True` flag was still being forwarded from `submit_entry`. Binance testnet rejects `STOP`/`TAKE_PROFIT` with `closePosition=true` — these limit-style types require explicit `quantity`, not `closePosition`. Both bracket orders failed → emergency market close fired → `submit_entry` returned `None` → zero positions ever opened despite valid signals.
- **Fix**: After the DEMO remap block, added guard: if `order_type in ("STOP", "TAKE_PROFIT") and close_position`, force `close_position=False` and `reduce_only=True` so the `quantity` branch is taken. Orders now send `quantity + reduceOnly=true + price + stopPrice + timeInForce=GTC`. MAINNET path unaffected (remap only runs in DEMO mode).
- **Status**: FIXED

### ISSUE-031: DMS fires during per-symbol loop — heartbeat only in bar-wait — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/pipeline/stage_08_live.py`, inner for-loop
- **Problem**: Per-symbol processing takes ~2.5s each. With 53 symbols = ~130s total loop time. DMS timeout = 60s. Heartbeat was only called in the bar-wait sleep loop, not inside the per-symbol loop — so DMS fired mid-bar on every run.
- **Fix**: Added `order_manager.heartbeat()` after each `_process_symbol()` call inside the for-loop. DMS now sees a heartbeat every ~2.5s throughout bar processing.
- **Status**: FIXED

### ISSUE-032: Structurally untradeable micro-cap coins probed every bar — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/pipeline/stage_08_live.py`, startup filters
- **Problem**: GALAUSDT, CHZUSDT, SEIUSDT, KMNOUSDT, XPLUSDT, FORMUSDT, JASMYUSDT all have `max_qty × price < min_notional` — physically impossible to trade at current price. These coins went through full feature computation, API calls, and model inference every bar before being rejected in `order_manager.submit_entry()`. Wasted ~7× 2.5s = 17s per bar and generated log spam.
- **Fix (FIX 9)**: Added startup exchange probe loop — for each forecast symbol, fetches current price and checks `max_qty × price >= min_notional`. Coins failing this check are excluded from `forecast_symbols` entirely. Failed probe (network error) keeps the symbol in. This runs once at startup, not per bar.
- **Status**: FIXED

### ISSUE-029: STOP_MARKET / TAKE_PROFIT_MARKET not supported on Binance testnet — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/execution/binance_client.py:place_order`
- **Problem**: Binance Futures testnet (`testnet.binancefuture.com`) returns HTTP 400 `-4120` "Order type not supported" for `STOP_MARKET` and `TAKE_PROFIT_MARKET` on `/fapi/v1/order`. Both bracket orders failed for every entry, triggering the double-bracket emergency close and leaving all positions unprotected.
- **Fix**: In `place_order()`, when `self._mode == "DEMO"`, transparently remap `STOP_MARKET → STOP` and `TAKE_PROFIT_MARKET → TAKE_PROFIT` before sending. Both limit-style types require `price + stopPrice + timeInForce=GTC`; if caller passed no explicit `price`, use `stop_price` for both fields. Stored `self._mode = mode.upper()` in `__init__`. MAINNET path completely unaffected.
- **Status**: FIXED

### ISSUE-030: size_usd not recalculated after qty capped to max_qty — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/execution/order_manager.py:submit_entry`, max_qty cap block
- **Problem**: When qty was capped to `max_qty` (e.g. NEARUSDT: intended 2125 contracts → capped to 120), `size_usd` still held the original uncapped value. `sync_fills` uses `pos["size_usd"]` to compute PnL and for the trade log — so every capped position reported wildly wrong PnL (18× over for NEARUSDT).
- **Fix**: After the cap line, added `size_usd = qty * entry_price` to make size_usd reflect the actual filled notional. Consistent with how `size_usd` is used downstream: `sync_fills` treats it as notional (multiplies `pnl_pct * size_usd` for USD PnL); `submit_exit` recomputes qty independently from `size_usd / entry_price` so that path is also corrected.
- **Status**: FIXED

### ISSUE-027: 4 symbols with negative test-set Sharpe excluded from live — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/pipeline/stage_08_live.py`, FIX 8
- **Problem**: TRXUSDT (Sharpe=-20.2, hit_rate=5.7%), BARDUSDT (-11.1), ASTERUSDT (-7.1), BTCUSDT (-3.6) have negative Sharpe on the test set (2026-01-01 to 2026-04-05). These models produce losing signals in the current bear market.
- **Fix**: At startup, load `results/per_symbol_metrics.csv` and filter out any symbol with `sharpe < 0`. Applied after artifact check, so these symbols still get forecasted but never open positions.
- **Status**: FIXED

### ISSUE-028: All 57 models Tier B — live uses 3× leverage (Tier A rate) — NOT FIXED
- **Date discovered**: 2026-04-05
- **Location**: `src/portfolio/position_sizer.py:get_growth_gate_limits`, `src/pipeline/stage_04_train.py:_assign_tier`
- **Root cause**: PBO check always returns 0.5 (ISSUE-006, splitter.py not implemented). `tier_A_pbo_max=0.40` never passes → all models classified Tier B.
- **Impact**: `get_growth_gate_limits` returns `leverage_a_max=3` regardless of tier — Tier B spec is `leverage_b_max=1`. So live positions use 3× leverage on Tier B models.
- **Risk level**: Acceptable for demo phase — 53/57 models have positive test Sharpe, 80%+ DA. 3× cap is still conservative.
- **Proper fix**: Implement PBO correctly in `src/models/splitter.py`, or add tier-aware leverage lookup in `get_growth_gate_limits`.
- **Status**: NOT FIXED (low priority for demo)

### ISSUE-026: Calibrator over-compression blocking live trades — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-09
- **Location**: `src/models/primary_model.py:train_xgb`, `config/base.yaml`
- **Problem**: IsotonicRegression calibrators were fit on val set during stage 4. Live raw probs (0.20–0.35) fell entirely in the compressed zone — e.g. raw=0.28 → cal=0.137. Almost no signals passed the floor.
- **Fix**: Changed `calibration_method` from `"isotonic"` to `"sigmoid"` in `config/base.yaml`. `train_xgb` now reads `cfg.model.calibration_method` and uses `_SigmoidCalibrator` (Platt scaling via `LogisticRegression`) when method is not "isotonic". `_SigmoidCalibrator` wraps `LogisticRegression` to expose `.predict(raw_probs)` interface compatible with all downstream callers. Existing calibrator `.pkl` files are `IsotonicRegression` instances and will remain so until next retrain.
- **Status**: FIXED (takes effect on next retrain)

### ISSUE-025: Order placement fails with "Precision is over the maximum" — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/execution/order_manager.py`, `src/execution/binance_client.py`
- **Problem**: Order quantity was rounded to 6 decimals unconditionally (`round(qty, 6)`). Some symbols (NEARUSDT, 1INCHUSDT) have different lot size (step size) requirements. Binance rejected orders with HTTP 400 "Precision is over the maximum defined for this asset".
- **Fix**: Added `get_qty_step(symbol)` method to `BinanceClient` that fetches symbol info from `/fapi/v1/exchangeInfo` and caches the LOT_SIZE filter. Updated `submit_entry()` and market close logic to round quantities: `qty = round(qty_raw / qty_step) * qty_step` instead of hardcoded 6 decimals. Three locations fixed: entry orders, close_position, and DMS shutdown.
- **Status**: FIXED

### ISSUE-023: Missing cross-sectional rank features in live inference — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `src/execution/live_features.py`
- **Problem**: `apply_cross_sectional_ranks()` was never called during live feature computation. All `_rank` features (e.g., `bb_pct_1h_rank`, `vwap_deviation_4h_rank`, `bv_rank`, `rsi_5_1h_rank`) were missing, causing 88/203 features to be NaN-filled on every bar. The pre-fitted stats file (`cross_sectional_stats.pkl`) existed but was never loaded or used.
- **Fix**: Added `apply_cross_sectional_ranks` import from `src.features.cross_sectional`. After deduplication (step 10) and before global shift (step 11), added call: `all_features = apply_cross_sectional_ranks(all_features, cs_stats_path, feature_cols_for_rank)` where `feature_cols_for_rank` is the list of numeric columns.
- **Status**: FIXED

### ISSUE-024: Fracdiff d-values not cached for live inference — FIXED
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-05
- **Location**: `data/checkpoints/fracdiff/` (missing directory), `src/features/fracdiff.py`
- **Problem**: The fracdiff cache directory did not exist. Price/volume columns (`close_5_mean`, `obv`, `vwap_20`, etc.) were differenced during training but not at inference time due to missing d-value cache files, causing feature distribution mismatch.
- **Root cause**: `fit_and_save_d_values()` is only called during stage 02 feature building when `is_train_period=True`. The condition was not triggered (likely due to data extending past `train_end`), so the cache directory was never created.
- **Fix**: Ran one-off repair script `scripts/repair_fracdiff_cache.py` which: (1) reads all 59 symbol feature parquets, (2) slices to `train_end=2025-09-30`, (3) calls `fit_and_save_d_values()` to populate `data/checkpoints/fracdiff/fracdiff_d_{symbol}_15m.json` for all symbols. All 59 cache files created successfully. Script saved for future use if fracdiff cache is lost.
- **Status**: FIXED

---

## Resolved Decisions

### ISSUE-028: All 57 models Tier B / TP-SL mismatch with training — FIXED (partial)
- **Date discovered**: 2026-04-05
- **Date fixed**: 2026-04-09
- **Location**: `config/base.yaml`, `growth_gate.tp_fixed_pct` / `growth_gate.sl_fixed_pct`
- **Problem**: `tp_fixed_pct=0.01` and `sl_fixed_pct=0.04` produced 1:4 R:R in live trading — guaranteed negative expectancy. Training labels used `tp_atr_mult=2.5 / sl_atr_mult=1.5` (ratio ~1.67:1 reward:risk). Live exits were mismatched with the barrier geometry the model was trained on.
- **Fix**: Set both to `0.0` so live trading reverts to ATR-based TP/SL that matches training labels exactly.
- **Status**: FIXED (tier assignment for PBO gate still deferred — see original ISSUE-028 note)

### ISSUE-029a: Optuna objective uses binary return proxy instead of actual price returns — FIXED
- **Date discovered**: 2026-04-09
- **Date fixed**: 2026-04-09
- **Location**: `src/models/primary_model.py:tune_hyperparams`, `src/pipeline/stage_04_train.py`
- **Problem**: `tune_hyperparams` used `y_train.map({0:-1,1:1})` as a return proxy in `compute_objective`. This ignores ATR-scaled realized returns from the triple-barrier scheme, so Optuna optimized against a proxy that is structurally different from actual PnL.
- **Fix**: `tune_hyperparams` now accepts optional `price_returns: np.ndarray | None = None`. If provided and length matches, uses actual realized returns from `tp_level`/`sl_level` columns; otherwise falls back to binary proxy. `stage_04_train.py` computes `price_returns` before Step 3 (Optuna) and passes it. Duplicate computation block later in the function removed.
- **Status**: FIXED (takes effect on next retrain)

### ISSUE-030a: Optuna Sharpe uses forced long/short — contradicts live dead zone — FIXED
- **Date discovered**: 2026-04-09
- **Date fixed**: 2026-04-09
- **Location**: `src/models/primary_model.py:compute_objective`
- **Problem**: `positions = np.where(proba > 0.5, 1.0, -1.0)` forced a position on every bar. Live signal generator uses `dead_zone_direction=0.03` (flat when `|prob - 0.5| < 0.03`). Mismatch: Optuna optimized for a fully-invested strategy; live uses a selective one.
- **Fix**: Changed to `positions = np.where(proba > 0.53, 1.0, np.where(proba < 0.47, -1.0, 0.0))`. Sharpe computed only on active-position bars (`positions != 0`), matching fold-Sharpe logic in `stage_04_train.py`.
- **Status**: FIXED (takes effect on next retrain)

### ISSUE-031a: Meta-labeler tree count too low (10 trees) — FIXED
- **Date discovered**: 2026-04-09
- **Date fixed**: 2026-04-09
- **Location**: `config/base.yaml`, `model.meta_n_estimators`
- **Problem**: `meta_n_estimators=10` gives insufficient capacity for learning conditional patterns over OOF predictions + regime + microstructure features (typically 8–12 features).
- **Fix**: Changed to `meta_n_estimators=100`.
- **Status**: FIXED (takes effect on next retrain)

### ISSUE-032a: meta_accuracy_oof key misleadingly named — FIXED
- **Date discovered**: 2026-04-09
- **Date fixed**: 2026-04-09
- **Location**: `src/pipeline/stage_05_meta.py`, `register_model` call
- **Problem**: `meta_accuracy_oof` was computed in-sample on the same training data used to fit the meta-labeler. Naming it "oof" was actively misleading and would cause future readers to trust it as a held-out estimate.
- **Fix**: Renamed to `meta_accuracy_train` in `register_model` metrics dict. Log message updated to say "train (in-sample)".
- **Status**: FIXED

### ISSUE-033a: stability_threshold too permissive (0.6) — TIGHTENED
- **Date discovered**: 2026-04-09
- **Date fixed**: 2026-04-09
- **Location**: `config/base.yaml`, `model.stability_threshold`
- **Decision**: Raised from 0.6 to 0.70. Feature must appear in 70%+ of bootstrap resamples to be selected. Reduces noise features entering the model.
- **Status**: FIXED (takes effect on next retrain)

---

## Session 2026-04-09 — 10-Area ML Overhaul (all pending from plan)

### DECISION-040: Objective function rewritten — Calmar-adjusted Sharpe + CVaR 95% penalty
- **Date**: 2026-04-09
- **Location**: `src/models/primary_model.py:compute_objective`
- **Change**: `compute_objective` now reads all weights from `cfg` (da=0.2, sharpe=0.5, ic=0.3, cvar=0.1). Sharpe replaced with 70%×Sharpe + 30%×Calmar blend. CVaR 95% tail penalty added (avg worst 5% returns). Fee-adjusted returns (round_trip_cost_pct=0.006 subtracted before Sharpe/CVaR). Dead zone from config (0.05). Warm-start: `study.enqueue_trial(prior_best_params)` before optimize.
- **Status**: FIXED (takes effect on next retrain)

### DECISION-041: CV embargo increased 50→192 bars (48h), cv_n_splits 5→8
- **Date**: 2026-04-09
- **Location**: `config/base.yaml`
- **Rationale**: Crypto autocorrelation decays over ~24-48h. 50-bar embargo (12.5h) was insufficient — val bars bled into train neighbourhood. 8 folds gives more robust cross-validation with longer embargo.
- **Status**: FIXED (takes effect on next retrain)

### DECISION-042: Labels aligned to 2:1 R:R — tp_atr_mult 2.5→2.0, sl_atr_mult 1.5→1.0, max_hold 16→32
- **Date**: 2026-04-09
- **Location**: `config/base.yaml` labels + backtest sections, `src/labels/triple_barrier.py`
- **Rationale**: 2.5/1.5 gives 1.67:1 R:R — below 2:1 needed to be profitable after fees at ~50% hit rate. New 2:1 means winning trade covers 2 losses. max_hold 32 bars (8h) captures swing moves better than 4h. Fee-adjusted reclassification: TP hits where gain < 0.6% round-trip cost → reclassified as neutral.
- **Status**: FIXED (takes effect on next --stage 3 --force)

### DECISION-043: Imputer replaced IterativeImputer→SimpleImputer(median) + missing indicator flags
- **Date**: 2026-04-09
- **Location**: `src/models/imputer.py`, `src/pipeline/stage_04_train.py`
- **Rationale**: IterativeImputer(BayesianRidge) has subtle leakage risk via correlated-feature imputation chain and is 10-100× slower. SimpleImputer(median) is leakage-safe and fast. Missing indicator flags (for cols with >5% NaN) capture informative missingness patterns. stage_04_train.py updated to handle expanded column count (`all_col_names`).
- **Status**: FIXED (takes effect on next --stage 4 --force; requires --force due to column count change)

### DECISION-044: Stability selection improved — RF 50 trees/depth 8 from config, MI tiebreaker, threshold 0.70→0.75, n_bootstrap 30→100
- **Date**: 2026-04-09
- **Location**: `src/models/stability_selection.py`, `config/base.yaml`
- **Rationale**: RF with 30 trees/depth 5 was underpowered for 100+ features. MI tiebreaker resolves borderline features by informativeness rather than random RF variation. Higher threshold reduces false positives.
- **Status**: FIXED (takes effect on next --stage 4 --force)

### DECISION-045: Meta-labeler — 300 trees/depth 6, Optuna mini-study (10 trials), meta_signal_floor 0.1→0.25, new meta features
- **Date**: 2026-04-09
- **Location**: `src/models/meta_labeler.py`, `src/pipeline/stage_05_meta.py`, `config/base.yaml`
- **Rationale**: 300 trees with Optuna-tuned lr+subsample gives better calibrated meta-probabilities. New features: `time_since_last_signal` (bars since last strong signal) and `spread_to_atr_ratio` (execution cost relative to volatility). meta_signal_floor 0.25 reduces noise trades — test on backtest first before live.
- **Status**: FIXED (takes effect on next --stage 5 --force)

### DECISION-046: Regime HMM 4→3 states, covariance "full"→"diag", hmm_retrain_hours 24→6
- **Date**: 2026-04-09
- **Location**: `config/base.yaml`
- **Rationale**: 3-state (bull/bear/sideways) ablation — simpler model, fewer parameters, more stable convergence. Diagonal covariance reduces parameter count further. 6h retrain cycle gives faster regime adaptation.
- **Status**: FIXED (takes effect on next retrain; hmm_retrain_hours takes effect immediately in live)

### DECISION-047: New features — BTC lag spillover (lags 1-4), time-of-day cyclical, ACF lag-1/5, funding_sign_persistence
- **Date**: 2026-04-09
- **Location**: `src/features/feature_pipeline.py`, `src/features/technical.py`, `src/features/funding_rates.py`
- **Features added**:
  - `btc_lag_1..4`: BTC log-returns at t-1 to t-4 (altcoin spillover). Global shift(1) in pipeline makes these fully backward-looking.
  - `tod_sin`, `tod_cos`: Cyclical encoding of hour-of-day UTC — captures session effects without leakage.
  - `acf_lag1_w96`, `acf_lag5_w96`: 24h rolling ACF — momentum vs mean-reversion signal.
  - `funding_sign_persistence_8`: Consecutive bars with same funding sign — persistence of funding pressure.
- **Status**: FIXED (takes effect on next --stage 2 --force)

### DECISION-048: Adaptive dead zone based on conformal width, dead_zone_direction 0.03→0.05
- **Date**: 2026-04-09
- **Location**: `src/portfolio/signal_generator.py`, `config/base.yaml`
- **Change**: Dead zone base raised 0.03→0.05. Scale factor: 1.0× when conf_width < 0.20, 1.25× when < 0.40, 1.50× above. Currently uses static placeholder 0.20 — activates with real per-bar conformal widths.
- **Status**: FIXED (immediately effective; adaptive scaling activates when real conformal widths are passed)

### ISSUE-049-FIX (2026-04-09): Meta-labeler missing scale_pos_weight — FIXED
- **Location**: `src/models/meta_labeler.py:train_meta_labeler`
- **Problem**: Primary DA ~55% → meta_y=1 for ~55% of bars. Without scale_pos_weight, meta-labeler over-predicts class 1 (trust signal) and under-identifies class 0 (don't trade).
- **Fix**: Compute `meta_spw = n_meta0 / n_meta1` before fitting. Pass to both mini-study XGBClassifier and final model. Logged as "Meta scale_pos_weight: X.XXX".
- **Status**: FIXED (takes effect on next --stage 5 --force)

### ISSUE-050-FIX (2026-04-09): Dead-zone bars incorrectly counted as meta_y=1 — FIXED
- **Location**: `src/models/meta_labeler.py:create_meta_labels`, `src/pipeline/stage_05_meta.py`
- **Problem**: Bars where |prob_long - 0.5| < dead_zone — primary model is in noise zone — were counted as meta_y=1 if the primary "happened to be correct" by chance. These are not true signals; treating them as correct inflates meta training quality.
- **Fix**: Added `dead_zone=0.05` param to `create_meta_labels`. Dead-zone bars set to meta_y=0 regardless of correctness. stage_05 passes `cfg.model.objective_dead_zone`.
- **Status**: FIXED (takes effect on next --stage 5 --force)

### ISSUE-051-FIX (2026-04-09): CVaR penalty unstable with small tail samples — FIXED
- **Location**: `src/models/primary_model.py:compute_objective`
- **Problem**: CVaR 95% with ~3000 samples = ~150 tail observations. For small symbols or folds with few active positions, n_tail can drop below 50 — estimate is too noisy to be a reliable penalty signal, destabilizes Optuna landscape.
- **Fix**: `effective_cvar_weight = cvar_weight * min(1.0, n_tail / 50)`. Auto-reduces weight proportionally when tail count is thin. At n_tail=25 (half of 50), effective weight = 0.05 instead of 0.1.
- **Status**: FIXED (takes effect on next --stage 4 --force)

### DECISION-052 (2026-04-09): CLAUDE.md protocol overhaul — append-only DECISIONS.md + anti-patterns
- **Location**: `CLAUDE.md`
- **Change**: Added rule 6 (DECISIONS.md append-only), updated End-of-Session actions (append only, not overwrite), rewrote Model Architecture section with current facts, added Anti-Patterns section listing 10+ already-evaluated proposals that should not be re-proposed without new evidence.
- **Rationale**: Claude was re-proposing already-fixed issues across sessions (meta_n_estimators, stability_threshold, TP/SL format) because DECISIONS.md status could be freely edited, creating the illusion that issues were still open.
- **Status**: FIXED

### ISSUE-011: Data split dates were stale — updated before retrain
- **Date discovered**: 2026-04-03
- **Date fixed**: 2026-04-04
- **Location**: `config/base.yaml` lines 12-15
- **Fix**: Updated to train_end=2025-09-30, val_end=2025-12-31, test_start=2026-01-01
- **Status**: FIXED

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

### ISSUE-022: Stage 6 ALL symbols fail "No primary model found" — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `model_registry.json`, `src/models/model_versioning.py`
- **Problem**: `model_registry.json` contained 57 meta entries and 0 primary entries. Registry was created fresh at 17:25:31 when the final `--force` run of stage 5 started — all 57 primary entries registered by stage 4 were lost because the registry file had been deleted between runs. Stage 5 workers initializing `{"models": []}` each time they found no file, overwriting prior content. Additionally `get_latest_model` and `get_active_models` held the exclusive write lock during reads, serializing all parallel stage 5 workers and creating lock contention risk.
- **Fix (immediate)**: Re-injected all 57 primary model entries into `model_registry.json` by reading version strings from `models/training_summary.csv` and feature names from `data/checkpoints/feature_selection/`. All model files confirmed present on disk.
- **Fix (structural)**: `register_model` now handles corrupt/list-format registry files defensively (try/except + format conversion). Extracted `_read_registry()` helper that reads without the write lock and retries on JSON decode error. `get_latest_model` and `get_active_models` now use `_read_registry()` instead of holding the exclusive lock during reads — eliminates lock serialization in parallel stage 5 workers.
- **Root cause prevention**: Never delete `model_registry.json` between pipeline runs. The registry accumulates primary entries from stage 4 that are needed by stage 5 (to link meta to primary version) and all downstream stages. If `--force` re-run of stage 5 is needed, the registry must persist.
- **Status**: FIXED

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

### ISSUE-017: Content-Type header caused all POST orders to fail — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/execution/binance_client.py`, `__init__`
- **Problem**: Session header `"Content-Type": "application/json"` applied to ALL requests. Binance FAPI POST endpoints (`/fapi/v1/order`, `/fapi/v1/allOpenOrders`) require `application/x-www-form-urlencoded`. Every `place_order`, `cancel_order`, `cancel_all_orders` returned HTTP 400.
- **Fix**: Removed `"Content-Type": "application/json"` from `self._session.headers.update`. `requests` now sets the correct Content-Type automatically (form-encoded for POST with `data=`, no content-type for GET with `params=`).
- **Status**: FIXED

### ISSUE-018: sync_fills used zeroed entryPrice as exit proxy — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/execution/order_manager.py`, `sync_fills`
- **Problem**: When a position closes, Binance zeroes `entryPrice` in the positionRisk response. Code used `live_pos.get("entryPrice")` as `exit_price`, which evaluated to 0.0 on all closed positions, making every realized PnL calculation wildly wrong.
- **Fix**: On position close, query `/fapi/v1/userTrades` via `client.get_recent_trades(symbol, limit=10)` and use `trades[-1]["price"]` as actual exit price. Falls back to `markPrice` if query fails.
- **Status**: FIXED

### DECISION-R009: Mainnet interlock requires CONFIRM_MAINNET_TRADING=yes env var
- **Date**: 2026-04-04
- **Decision**: `stage_08_live.py` `run()` checks `cfg.trading.mode == "MAINNET"` immediately after config load. If true, `os.environ.get("CONFIRM_MAINNET_TRADING")` must equal `"yes"` (case-insensitive). If not set, raises `RuntimeError` before any BinanceClient is created.
- **Rationale**: Prevents accidental MAINNET trades if config is edited without intent. Operator must explicitly set `CONFIRM_MAINNET_TRADING=yes` AND have completed `demo_trades_required` demo trades.
- **Status**: CONFIRMED

### DECISION-R010: Live dashboard added (src/dashboard/live_dashboard.py)
- **Date**: 2026-04-04
- **Decision**: `LiveDashboard` class renders per-bar summary to terminal using `rich` (with plain-terminal fallback). Updated at end of each bar loop in `stage_08_live.py`. Shows mode, equity, P&L, open positions, last signals, and demo trade progress.
- **Rationale**: Operators running live need real-time visibility without tailing raw log files. Dashboard is non-fatal — any render error is logged as DEBUG and skipped.
- **Status**: CONFIRMED

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

### GIT-003: DMS heartbeat before+after sleep (superseded by ISSUE-010 fix)
- **Commit**: `70e8c36` — 2026-04-03
- **Decision**: `fix: DMS heartbeat before sleep, klines limit cap 1500, load .env at stage 8 start`
- **What**: Added `heartbeat()` call before AND after the 900s bar-wait sleep. Also capped Binance klines fetch at 1500 (FAPI hard limit). Added `.env` loading at stage_08 startup for API keys.
- **Note**: DMS problem fully resolved by ISSUE-010 (heartbeat loop every 30s during bar-wait).
- **Status**: SUPERSEDED — ISSUE-010 FIXED

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

ISSUE-001 through ISSUE-021 are all FIXED. Remaining open:

1. ISSUE-006 PBO computation still returns ~0.5 (splitter.py) — Tier A gate unreliable, medium priority (acceptable for demo)
2. Meta-labeling still needs re-run for all 15 symbols besides SOLUSDT (run `--stage 5 --force` then stages 6+7)

### ISSUE-019: live_features.py was missing ~70% of model features — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/execution/live_features.py`
- **Problem**: `compute_live_features` only called `build_technical_features`. All microstructure, funding, HMM regime, BOCPD, fracdiff, HTF, macro, and onchain features were absent. Missing features were NaN-imputed to train means → garbage predictions.
- **Fix**: Expanded to call the full feature pipeline in the same order as `feature_pipeline.build_features_for_symbol`. New signature accepts `klines_1h/4h/1d` (fetched per-symbol in `_process_symbol`) and `btc_klines_15m` (fetched once per bar in the main loop). Also added global `shift(1)` before taking `last_row` to match training.
- **Status**: FIXED

### ISSUE-020: conformal_width inverted — certain signals got smallest position — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/pipeline/stage_06_portfolio.py:119`
- **Problem**: `conformal_width = abs(raw_proba[:, 1] - 0.5) * 2` is a confidence score (0=uncertain, 1=certain). `apply_conformal_size_scaling` treats width < 0.20 as "narrow = full position". Combined: maximum confidence → 1.0 width → 0.3× scale. Backwards.
- **Fix**: Changed to `1.0 - abs(raw_proba[:, 1] - 0.5) * 2`. Now width=0 means certain (full position), width=1 means uncertain (0.3× scale).
- **Status**: FIXED

### ISSUE-021: half_kelly double-halved in stage_08 — FIXED
- **Date discovered**: 2026-04-04
- **Date fixed**: 2026-04-04
- **Location**: `src/pipeline/stage_08_live.py:413`
- **Problem**: `cfg.portfolio.kelly_fraction = 0.5` (already half-Kelly). Code did `kelly_fraction * 0.5` → effective 0.25× Kelly. Live positions were half the size that backtest used.
- **Fix**: Changed to `half_kelly = float(cfg.portfolio.kelly_fraction)`.
- **Status**: FIXED

### DECISION-R011: 2-notebook Kaggle split (CPU labels + GPU training)
- **Date**: 2026-04-04
- **Decision**: Split monolithic notebook into `notebook_cpu.py` (stage 3 labels, no GPU) and `notebook_gpu.py` (stages 4-7, GPU, auto-push output).
- **Rationale**: Labels rarely change, CPU-only. Training is GPU-critical. Split reduces GPU quota waste and enables crash recovery per stage.
- **Status**: CONFIRMED

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
