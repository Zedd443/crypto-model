# crypto-model

Automated crypto futures trading system using machine learning. Trains per-symbol XGBoost
classifiers on 15-minute OHLCV data, generates signals via meta-labeling and regime detection,
and executes orders on Binance Demo/Mainnet FAPI.

---

## Architecture

```
Stage 1  ingest        Download OHLCV, on-chain, macro, fear & greed data
Stage 2  features      Technical indicators, regime, cross-sectional ranks (per symbol)
Stage 3  labels        Triple-barrier labeling + sample weights
Stage 4  training      XGBoost + Optuna tuning + stability feature selection
Stage 5  meta_labeling Meta-labeler trained on OOF predictions (signal confidence filter)
Stage 6  portfolio     Pre-compute signals, position sizing, correlation filter
Stage 7  backtest      Walk-forward backtest with realistic costs and circuit breakers
Stage 8  live          Real-time execution loop — Binance Demo or Mainnet FAPI
```

**Model stack:** XGBoost primary → isotonic calibration → meta-labeler → signal_strength = primary_prob × meta_prob

**Risk controls:** Triple-barrier labels · purged time-series CV · growth gate tiers · daily drawdown halt · dead-man-switch

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env — fill in Binance Demo keys from https://demo-fapi.binance.com
```

### 3. Run the full pipeline (stages 1–7)

```bash
# All stages sequentially
python -m src.pipeline.run_pipeline

# Single stage
python -m src.pipeline.run_pipeline --stage 2

# Single stage, single symbol (useful for debugging)
python -m src.pipeline.run_pipeline --stage 4 --symbol BTCUSDT

# Force re-run even if stage is marked done
python -m src.pipeline.run_pipeline --stage 4 --force
```

### 4. Start live demo trading (stage 8)

```bash
python -m src.pipeline.run_pipeline --stage 8
```

The loop wakes at every 15-minute bar boundary, fetches live OHLCV, runs inference,
and places MARKET + bracket (TP limit / SL stop-market) orders on Binance Demo FAPI.
`Ctrl+C` triggers a clean shutdown that cancels all open orders before exit.

---

## GPU Training on Kaggle

Feature files (19 GB) are pre-built locally and uploaded as 4 batch datasets (~4.7 GB each).
Stages 3–7 then run on Kaggle T4/P100 GPU in ~1 hour.

### Upload feature datasets (one-time or after config changes)

```bash
# Upload all 4 batches
python kaggle_notebook/upload_features.py

# Resume a specific batch if interrupted
python kaggle_notebook/upload_features.py --batch 2
```

### Push and run the notebook

```bash
kaggle kernels push -p kaggle_notebook

# Monitor status
kaggle kernels status irfandragneel/crypto-model-training

# Read logs after completion
python kaggle_notebook/fetch_log.py --tail 50
```

### Download and install output

```bash
# Download output zip
kaggle kernels output irfandragneel/crypto-model-training -p C:/kgl_out

# Extract and copy to project
cd C:/kgl_out
unzip crypto_model_output.zip -d extracted
cp -r extracted/checkpoints/* data/checkpoints/
cp -r extracted/models/*      models/
cp extracted/model_registry.json .
cp extracted/project_state.json  .

# Update checkpoints dataset for future Kaggle runs (skip stage 2 next time)
cp project_state.json kaggle_datasets/checkpoints/
kaggle datasets version -p kaggle_datasets/checkpoints -m "post-training update"
```

---

## Stage 2 Modes (Feature Generation)

| Mode | Condition | Duration |
|------|-----------|----------|
| **A — skip** | `crypto-model-features-{1..4}` datasets attached | ~0 min (symlinked) |
| **B — generate** | Only raw data attached, no feature datasets | ~2–3 h CPU |
| **C — upload local** | Stage 2 done locally → run `upload_features.py` → becomes Mode A | ~30 min upload |

---

## Key Configuration (`config/base.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `backtest.daily_halt_dd_pct` | `0.15` | Halt trading if daily drawdown exceeds 15% |
| `portfolio.signal_floor_prob` | `0.55` | Minimum signal strength to enter a trade |
| `portfolio.max_total_margin_pct` | `0.80` | Max 80% of equity used as margin |
| `model.optuna_n_trials` | `50` | Hyperparameter search trials per symbol |
| `trading.mode` | `DEMO` | `DEMO` / `TESTNET` / `MAINNET` |
| `trading.dead_man_switch_seconds` | `60` | Cancel all orders if loop silent for 60s |
| `growth_gate.threshold` | `300.0` | Equity threshold to unlock multi-symbol trading |

> **Changing RR (tp_atr_mult / sl_atr_mult):** requires re-running stages 3–7.
> **Changing signal_floor_prob or sizing:** re-run stages 6–7 only.
> **Changing features/indicators:** re-run stages 2–7 and re-upload feature datasets.

---

## Project Structure

```
config/
  base.yaml              All hyperparameters and trading config
  symbols.yaml           Universe of 59 USDT-margined futures symbols

data/
  raw/                   OHLCV parquets downloaded by stage 1
  features/              Feature parquets generated by stage 2 (19 GB)
  checkpoints/           Scalers, imputers, HMM models, cross-sectional stats
  labels/                Triple-barrier labels per symbol
  processed/             Aligned multi-timeframe data

models/                  Trained XGBoost + meta-labeler files per symbol

results/
  live_trade_log.csv     Live trade history (stage 8)
  *_trades.csv           Per-symbol backtest trades (stage 7)
  *_nav.parquet          NAV curves (stage 7)

src/
  data/                  Data loading, macro/on-chain merging
  features/              Feature engineering (technical, regime, cross-sectional)
  labels/                Triple-barrier labeling, sample weights
  models/                XGBoost training, meta-labeler, model versioning
  portfolio/             Signal generation, position sizing, risk, correlation
  backtest/              Walk-forward engine, metrics, costs
  execution/             Live trading: Binance client, order manager, live features
  pipeline/              Stage runners (stage_01 … stage_08) + run_pipeline entry point
  utils/                 Config loader, logger, state manager, IO utils

kaggle_notebook/
  notebook.py            Kaggle GPU training script (stages 2–7)
  kernel-metadata.json   Notebook config (GPU enabled, datasets attached)
  upload_features.py     Batch upload feature parquets to Kaggle datasets
  fetch_log.py           Fetch kernel log after completion
```

---

## Environment Variables (`.env`)

```
BINANCE_API_KEY          Mainnet FAPI key (data download + live trading)
BINANCE_API_SECRET
BINANCE_DEMO_API_KEY     Demo FAPI key — get from https://demo-fapi.binance.com
BINANCE_DEMO_API_SECRET
BINANCE_DEMO_FAPI_BASE_URL  https://testnet.binancefuture.com
BINANCE_USE_DEMO         true / false
COINMETRICS_API_KEY      On-chain data (optional)
```
