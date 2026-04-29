# Crypto ML Trading System — User Guide

## Table of Contents
1. [Quick Start](#1-quick-start)
2. [Architecture Overview](#2-architecture-overview)
3. [Prerequisites & Installation](#3-prerequisites--installation)
4. [Configuration Reference](#4-configuration-reference)
5. [Pipeline Stages Reference](#5-pipeline-stages-reference)
6. [First Run — Full Pipeline](#6-first-run--full-pipeline)
7. [Live Trading Setup](#7-live-trading-setup)
8. [VPS Deployment](#8-vps-deployment)
9. [Automated Weekly Retrain](#9-automated-weekly-retrain)
10. [Troubleshooting](#10-troubleshooting)
11. [Telegram Notifications Reference](#11-telegram-notifications-reference)

---

## 1. Quick Start

```bash
# 1. Install
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt

# 2. Configure keys
copy .env.example .env   # fill in BINANCE_DEMO_API_KEY + BINANCE_DEMO_API_SECRET

# 3. Run full pipeline (stages 1→7, ~6-10h on CPU)
.venv/Scripts/python.exe -m src.pipeline.run_pipeline

# 4. Start live demo trading
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 8
```

---

## 2. Architecture Overview

```
Local / VPS                          Kaggle (weekly retrain)
────────────────────                 ─────────────────────────
Stage 1: Ingest OHLCV               git clone repo
Stage 2: Feature engineering        Set TRAIN_END_DATE=today-7d
Stage 3: Triple-barrier labels      Run stages 1→5
Stage 4: XGBoost training  ──────→  Upload models to GitHub Release
Stage 5: Meta-labeler               ↓
Stage 6: Signal generation          GitHub Actions
Stage 7: Backtest          ←──────  SSH: vps_hot_swap.sh
Stage 8: Live loop                  Stop stage 8 (graceful)
   ↓                                Swap models
Binance FAPI                        Restart stage 8
(Demo / Mainnet)
```

**Model architecture per symbol:**
- Primary: XGBoost classifier — predicts P(TP hit before SL within N bars)
- Meta: LogisticRegression — predicts P(primary was correct), used as confidence multiplier
- HTF: Independent XGBoost per timeframe (1h/4h/1d) — approval filter only
- Signal = `|P(dir) − 0.5| × 2 × meta_prob` — must exceed `signal_floor_prob`

---

## 3. Prerequisites & Installation

**Requirements:**
- Python 3.10+
- Git
- Binance Demo FAPI account (free): `demo-fapi.binance.com` → API Management
- `gh` CLI (for deploy automation): `https://cli.github.com`

**Install:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

**Environment variables** — copy and fill `.env.example`:
```bash
copy .env.example .env    # Windows
cp .env.example .env      # Linux/Mac
```

| Variable | Required for | Where to get |
|----------|-------------|-------------|
| `BINANCE_DEMO_API_KEY` | Stage 1, Stage 8 demo | demo-fapi.binance.com → API Management |
| `BINANCE_DEMO_API_SECRET` | Stage 1, Stage 8 demo | Same |
| `BINANCE_API_KEY` | Stage 8 mainnet only | binance.com → API Management |
| `BINANCE_API_SECRET` | Stage 8 mainnet only | Same |
| `TELEGRAM_BOT_TOKEN` | Notifications (optional) | @BotFather on Telegram |
| `TELEGRAM_CHAT_ID` | Notifications (optional) | @userinfobot on Telegram |
| `CONFIRM_MAINNET_TRADING` | Mainnet safety gate | Set to `yes` manually |

**Verify install:**
```bash
.venv/Scripts/python.exe -c "from src.utils.config_loader import load_config; load_config(); print('OK')"
```

---

## 4. Configuration Reference

All settings in `config/base.yaml`. Key sections:

### Data splits
```yaml
data:
  train_end: "2025-09-30"    # overridable via TRAIN_END_DATE env var (Kaggle retrain)
  val_start: "2025-10-01"    # overridable via VAL_START_DATE
  val_end:   "2025-12-31"    # overridable via VAL_END_DATE
  test_start: "2026-01-01"   # overridable via TEST_START_DATE
```

### Trading
```yaml
trading:
  mode: "DEMO"               # DEMO or MAINNET
  leverage: 10
  exclude_symbols: []        # permanent skip list — no retrain needed

growth_gate:
  max_open_positions: 2      # max simultaneous trades
  tp_fixed_pct: 0.01         # take-profit 1%
  sl_fixed_pct: 0.05         # stop-loss 5%
  daily_profit_target_pct: 0.04   # stop new entries after +4% day
  daily_loss_limit_pct: 0.05      # hard stop after -5% day

portfolio:
  signal_floor_prob: 0.20    # minimum signal to enter (0–1)
  dead_zone_direction: 0.05  # ignore if |P−0.5| < this
```

### Notifications
```yaml
telegram:
  enabled: true
  notify_entry: true
  notify_exit: true
  notify_daily_summary: true
  notify_heartbeat: true
```

### Config change → retrain matrix

| What changed | Re-run from |
|-------------|-------------|
| `features.*` | `--from-stage 2 --force` |
| `labels.*` (TP, SL, hold bars) | `--from-stage 3 --force` |
| `model.*` (hyperparams, CV) | `--from-stage 4 --force` |
| `portfolio.signal_floor_prob` | `--from-stage 6 --force` |
| `trading.exclude_symbols` | Restart stage 8 only |
| `telegram.*` | Restart stage 8 only |

---

## 5. Pipeline Stages Reference

| Stage | Command flag | Input | Output | Typical duration |
|-------|-------------|-------|--------|-----------------|
| 1 — ingest | `--stage 1` | Binance API | `data/raw/*.parquet` | 30–60 min |
| 2 — features | `--stage 2` | `data/raw/` | `data/features/*.parquet` | 2–4 h |
| 3 — labels | `--stage 3` | `data/raw/` + features | `data/labels/*.parquet` | 10–20 min |
| 4 — training | `--stage 4` | features + labels | `models/*/primary_*` | 2–6 h |
| 4b — htf_train | `--stage 4b` | features (1h/4h/1d) | `models/*/htf_*` | 10–30 min |
| 5 — meta | `--stage 5` | OOF predictions | `models/*/meta_*` | 5–10 min |
| 6 — portfolio | `--stage 6` | features + models | `data/checkpoints/signals/` | 5–10 min |
| 7 — backtest | `--stage 7` | signals | `results/per_symbol_metrics.csv` | 5–10 min |
| 8 — live | `--stage 8` | live OHLCV + models | orders on Binance | continuous |

**Common commands:**
```bash
# Full pipeline (skips completed stages)
.venv/Scripts/python.exe -m src.pipeline.run_pipeline

# Force re-run from stage 4 onwards
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --from-stage 4 --force

# Single stage, single symbol (debugging)
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 4 --symbol BTCUSDT --force

# Check what's done
python -c "import json; s=json.load(open('project_state.json')); [print(k,v['status']) for k,v in s['stages'].items()]"
```

---

## 6. First Run — Full Pipeline

```bash
# Step 1: ingest data (needs internet + Binance Demo keys)
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 1

# Step 2: features (CPU-heavy, ~3h for 59 symbols)
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 2

# Step 3: labels
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 3

# Step 4: train (the longest step — use Kaggle for speed, see section 9)
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 4

# Step 4b: HTF approval models (run after stage 4, before stage 5)
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 4b

# Steps 5-7: meta, portfolio signals, backtest
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --from-stage 5
```

Or run everything at once (resumes from where it left off if interrupted):
```bash
.venv/Scripts/python.exe -m src.pipeline.run_pipeline
```

**Check results after stage 7:**
```bash
python -c "
import pandas as pd
df = pd.read_csv('results/per_symbol_metrics.csv')
print(df.sort_values('sharpe')[['symbol','sharpe','hit_rate','n_trades','edge_val']].to_string())
"
```

---

## 7. Live Trading Setup

### DEMO mode (paper trading)

```bash
# Ensure trading.mode = "DEMO" in config/base.yaml
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 8
```

**What you see every 15 minutes:**
```
BAR 2026-04-23 12:15 UTC  equity=$127.40  open=1  entered=1  floor=32

 Symbol      Dir  P(dir)  P(meta)  Signal  TP reach  Action
 AVAXUSDT    ▲    0.681   0.724    0.421   0.872     ENTERED
 SOLUSDT     ▲    0.634   0.611    0.311   0.654     SKIP_LIMIT
 INJUSDT     ▼    0.338   0.591    0.270   0.521     SKIP_DEAD_ZONE
```

| Action code | Meaning |
|-------------|---------|
| `ENTERED` | Order placed |
| `CANDIDATE` | Passed filters, no slot |
| `HOLD` | Already in position |
| `SKIP_FLOOR` | Signal below `signal_floor_prob` |
| `SKIP_DEAD_ZONE` | Model conviction too low |
| `SKIP_DAILY` | Daily profit/loss cap hit |
| `SKIP_LIMIT` | Max open positions reached |
| `FAILED` | Exchange error |

**Stop:** `Ctrl+C` — bracket orders (TP/SL) remain active on exchange.

### MAINNET mode checklist

Before switching to real money:
- [ ] At least 500 demo trades completed (check `project_state.json → account.demo_trades_completed`)
- [ ] Backtest Sharpe > 0 for symbols you will trade
- [ ] `BINANCE_API_KEY` and `BINANCE_API_SECRET` filled in `.env`
- [ ] `CONFIRM_MAINNET_TRADING=yes` set in environment
- [ ] `trading.mode: MAINNET` in `config/base.yaml`
- [ ] VPS deployed (see section 8) — don't run mainnet on a laptop

```bash
# Windows
set CONFIRM_MAINNET_TRADING=yes
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 8

# Linux
CONFIRM_MAINNET_TRADING=yes .venv/bin/python -m src.pipeline.run_pipeline --stage 8
```

---

## 8. VPS Deployment

### Recommended VPS specs
- 1 vCPU, 1–2 GB RAM (stage 8 uses ~512 MB)
- Ubuntu 22.04 LTS
- ~$4–6/month (DigitalOcean, Hetzner, Vultr)

### One-time VPS setup

```bash
# On VPS — install Python and dependencies
sudo apt update && sudo apt install -y python3.11 python3.11-venv git curl

# Clone repo
git clone https://github.com/YOUR_USERNAME/crypto_model.git ~/crypto_model
cd ~/crypto_model
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Create .env with your keys
cp .env.example .env
nano .env   # fill in keys
```

### systemd service (auto-restart on crash)

```bash
# Create service file
sudo tee /etc/systemd/system/crypto-live.service << EOF
[Unit]
Description=Crypto ML Live Trading
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/crypto_model
EnvironmentFile=$HOME/crypto_model/.env
Environment=CONFIRM_MAINNET_TRADING=yes
ExecStart=$HOME/crypto_model/.venv/bin/python -m src.pipeline.run_pipeline --stage 8
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable crypto-live
sudo systemctl start crypto-live
```

**Monitor:**
```bash
sudo systemctl status crypto-live
journalctl -u crypto-live -f           # live logs
journalctl -u crypto-live -n 100       # last 100 lines
```

### Push code from local to VPS

After making code changes locally:
```bash
python scripts/deploy.py               # push code only
python scripts/deploy.py --retrain     # push + trigger full retrain
python scripts/deploy.py --check       # preflight checks only
```

This uses GitHub Actions to rsync `src/`, `config/`, `scripts/` to VPS automatically.

---

## 9. Automated Weekly Retrain

The system retrains every Sunday 00:00 UTC automatically:
1. GitHub Actions triggers a Kaggle notebook (free compute)
2. Kaggle runs stages 1→5 with `TRAIN_END_DATE` set to today-7d
3. Artifacts uploaded to GitHub Release
4. GitHub Actions SSHs to VPS and runs hot-swap
5. Stage 8 waits for open positions to close, then restarts with new models

**Downtime: 5–10 minutes** (only after all positions close naturally via TP/SL).

### One-time setup

**Step 1 — GitHub secrets** (set once via `gh secret set` or GitHub Settings → Secrets):
```bash
gh secret set VPS_HOST           # your VPS IP or hostname
gh secret set VPS_USER           # your VPS username (e.g. ubuntu)
gh secret set VPS_SSH_KEY        # contents of your VPS private key file
gh secret set KAGGLE_USERNAME
gh secret set KAGGLE_KEY         # from kaggle.com → Account → API → Create Token
gh secret set TELEGRAM_BOT_TOKEN
gh secret set TELEGRAM_CHAT_ID
```

**Step 2 — Kaggle kernel** (create once manually):
1. Go to kaggle.com → Code → New Notebook
2. Name it `crypto-weekly-retrain`
3. Add Kaggle secrets: `GITHUB_TOKEN`, `GITHUB_REPO`, `BINANCE_DEMO_API_KEY`, `BINANCE_DEMO_API_SECRET`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`
4. Upload `scripts/kaggle_retrain.py` as the notebook script

**Step 3 — VPS: install gh CLI and authenticate:**
```bash
# On VPS
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list
sudo apt update && sudo apt install gh -y
gh auth login
```

**Test the full flow manually:**
```bash
python scripts/deploy.py --retrain   # triggers retrain now, don't wait for Sunday
```

### What Telegram alerts you receive

| Event | Message |
|-------|---------|
| Retrain triggered | Kaggle kernel started |
| Retrain complete | Model uploaded, artifact URL |
| Retrain failed | Error details, VPS unchanged |
| Maintenance scheduled | "Retrain flag set, waiting for positions" |
| Waiting for positions | "N positions open, checking next bar" (every 5 min) |
| Stage 8 stopped | "Stopped for model hot-swap" |
| Stage 8 restarted | "Hot-swap complete, running new models" |
| Hot-swap failed | Error details, old models kept |

### Error handling

| Failure | Outcome |
|---------|---------|
| Kaggle OOM/crash | GHA fails → Telegram alert → VPS keeps old models |
| Kaggle timeout (>8h) | Same |
| VPS SSH unreachable | GHA SSH step fails → Telegram alert → no hot-swap |
| Artifact download fails | `vps_hot_swap.sh` exits → no restart → old models kept |
| Positions won't close (>6h) | Hot-swap aborted → Telegram alert → old models kept |
| Stage 8 restart fails | Telegram CRITICAL alert with journalctl snippet |

**Key invariant: VPS always keeps running old models if anything fails.**

---

## 10. Troubleshooting

### Common errors

| Error | Cause | Fix |
|-------|-------|-----|
| `float() argument must be a string or a real number, not 'NAType'` | NaN reached model predict | Fixed in stage 04 and 06 — retrain |
| `OOF predictions not found: ...oof/SYMBOL_15m_oof_proba.npy` | Stage 4 failed for that symbol | `--stage 4 --symbol SYMBOLUSDT --force` |
| `Imputer not found: ...imputer_SYMBOL_15m.pkl` | Stage 4 not run for symbol | Same as above |
| `Primary model prediction failed` | NaN in features after scaler | `np.nan_to_num` applied automatically in stage 06 — retrain if persistent |
| `Cannot connect to Binance` | Wrong keys or mode mismatch | Check `.env` matches `trading.mode` |
| `ACTIVE for trading: 0` | All symbols excluded by filters | Check `per_symbol_metrics.csv` — likely need retrain |
| `Bootstrap N failed: NAType` | NaN in features before stability selection | Fixed — imputer now runs before feature selection |
| `Signal strength always the same` | Expected — model weights are static | Confidence changes with HMM regime and ATR per bar |

### Check pipeline status
```bash
python -c "
import json
s = json.load(open('project_state.json'))
for stage, info in s['stages'].items():
    issues = info.get('issues', [])
    print(f'{stage:15} {info[\"status\"]:12} {len(issues)} issues')
"
```

### Force-retrain a single symbol
```bash
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 4 --symbol SOLUSDT --force
.venv/Scripts/python.exe -m src.pipeline.run_pipeline --stage 5 --symbol SOLUSDT --force
```

### Check model quality for a symbol
```bash
python -c "
import json
reg = json.load(open('model_registry.json'))
sym = 'BTCUSDT'
entries = [e for e in reg if e.get('symbol') == sym and e.get('model_type') == 'primary']
if entries:
    e = sorted(entries, key=lambda x: x['registered_at'])[-1]
    print('tier:', e.get('tier'), '| edge_val:', e.get('metrics', {}).get('edge_val'))
    print('sharpe:', e.get('metrics', {}).get('synthetic_sharpe'))
    print('pbo:', e.get('metrics', {}).get('pbo'))
"
```

### Log locations
| Log | Location |
|-----|----------|
| Pipeline logs | `logs/YYYY-MM-DD_pipeline.log` |
| Live trade log | `results/live_trade_log.csv` |
| Model registry | `model_registry.json` |
| Pipeline state | `project_state.json` |
| VPS service logs | `journalctl -u crypto-live` |

---

## 11. Telegram Notifications Reference

Configure in `config/base.yaml` under `telegram:`.

| Notification | When sent | Configurable |
|-------------|-----------|-------------|
| Entry | Every trade entered | `notify_entry: true/false` |
| Exit | Every trade closed (TP/SL/timeout) | `notify_exit: true/false` |
| Daily summary | Once per UTC day (first bar after 00:00) | `notify_daily_summary: true/false` |
| Heartbeat | Every hour stage 8 is running | `notify_heartbeat: true/false` |
| Alert | Warnings, errors, circuit breakers | Always on if Telegram enabled |
| Maintenance | Hot-swap lifecycle events | Always on if Telegram enabled |

**To disable all notifications:** set `TELEGRAM_BOT_TOKEN=` (empty) in `.env`.

**Setup:**
1. Message `@BotFather` → `/newbot` → get token
2. Message `@userinfobot` → get your chat ID
3. Send at least one message to your bot (`/start`) before stage 8 starts
