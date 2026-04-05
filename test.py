import numpy as np
import pickle
from pathlib import Path
from omegaconf import OmegaConf
from src.models.primary_model import load_model
from src.models.model_versioning import get_latest_model
from src.execution.live_features import compute_live_features
from src.execution.binance_client import BinanceClient

cfg = OmegaConf.load("config/base.yaml")

# ── 1. CALIBRATOR MAPPING ─────────────────────────────────────────────────────
print("=" * 60)
print("1. CALIBRATOR MAPPING CHECK")
print("=" * 60)

entry = get_latest_model("BTCUSDT", "15m", model_type="primary")
model, calibrator = load_model("BTCUSDT", "15m", entry["version"], cfg.data.models_dir)

print(f"Calibrator type   : {type(calibrator).__name__}")
print(f"Model version     : {entry['version']}")
print()

test_inputs = np.linspace(0.1, 0.9, 17)
test_outputs = calibrator.predict(test_inputs)

print(f"{'Raw':>6}  {'Cal':>6}  {'Ratio':>6}  Bar")
print("-" * 50)
all_healthy = True
for i, o in zip(test_inputs, test_outputs):
    ratio = o / i if i > 0 else 0
    bar = "█" * int(o * 40)
    flag = ""
    if abs(i - 0.5) < 0.15 and ratio < 0.6:
        flag = " ← ⚠️  compressed"
        all_healthy = False
    print(f"{i:>6.2f}  {o:>6.4f}  {ratio:>6.2f}  {bar}{flag}")

print()
if all_healthy:
    print("✅ Calibrator shape looks healthy")
else:
    print("⚠️  Calibrator still compressing mid-range — consider refit")

# ── 2. FEATURE COVERAGE ───────────────────────────────────────────────────────
print()
print("=" * 60)
print("2. FEATURE COVERAGE CHECK (BTCUSDT)")
print("=" * 60)

client = BinanceClient(cfg)
klines = client.get_klines("BTCUSDT", "15m", limit=600)
k1h    = client.get_klines("BTCUSDT", "1h",  limit=200)
k4h    = client.get_klines("BTCUSDT", "4h",  limit=100)
k1d    = client.get_klines("BTCUSDT", "1d",  limit=60)

feat = compute_live_features(
    "BTCUSDT", cfg, klines,
    klines_1h=k1h, klines_4h=k4h, klines_1d=k1d,
    btc_klines_15m=klines,
)

nan_cols   = feat[feat.isna()].index.tolist()
rank_nan   = [c for c in nan_cols if c.endswith("_rank")]
nonrank_nan = [c for c in nan_cols if not c.endswith("_rank")]

print(f"Total features    : {len(feat)}")
print(f"NaN count         : {len(nan_cols)}")
print(f"  rank NaN        : {len(rank_nan)}")
print(f"  non-rank NaN    : {len(nonrank_nan)}")

if nan_cols:
    print(f"NaN cols          : {nan_cols[:10]}")
    print("⚠️  Still have missing features" if len(nan_cols) > 1 else "✅ Only 1 NaN col (funding rank — known minor issue)")
else:
    print("✅ Zero NaN features")

# ── 3. END-TO-END SIGNAL ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("3. END-TO-END SIGNAL CHECK")
print("=" * 60)

X = feat.values.reshape(1, -1)
nan_in_x = int(np.isnan(X).sum())
raw_prob  = float(model.predict_proba(X)[0][1])
cal_prob  = float(calibrator.predict(np.array([raw_prob]))[0]) if calibrator else raw_prob
floor     = float(cfg.portfolio.signal_floor_prob)
direction = "long" if raw_prob >= 0.5 else "short"

print(f"NaN in feature vec: {nan_in_x}")
print(f"Raw prob          : {raw_prob:.4f}")
print(f"Calibrated prob   : {cal_prob:.4f}")
print(f"Signal floor      : {floor}")
print(f"Direction         : {direction}")
print(f"Would trade       : {'✅ YES' if cal_prob >= floor else '❌ NO (below floor)'}")

# ── 4. MULTI-SYMBOL SPOT CHECK ────────────────────────────────────────────────
print()
print("=" * 60)
print("4. MULTI-SYMBOL SPOT CHECK (first 5 symbols)")
print("=" * 60)

import yaml
with open("config/symbols.yaml") as f:
    all_symbols = list(yaml.safe_load(f).get("symbols", {}).keys())[:5]

results = []
for sym in all_symbols:
    try:
        sym_entry = get_latest_model(sym, "15m", model_type="primary")
        if sym_entry is None:
            results.append((sym, None, None, None, "no model"))
            continue
        sym_model, sym_cal = load_model(sym, "15m", sym_entry["version"], cfg.data.models_dir)
        sym_klines = client.get_klines(sym, "15m", limit=600)
        sym_feat   = compute_live_features(
            sym, cfg, sym_klines,
            klines_1h=client.get_klines(sym, "1h", limit=200),
            klines_4h=client.get_klines(sym, "4h", limit=100),
            klines_1d=client.get_klines(sym, "1d", limit=60),
            btc_klines_15m=klines,
        )
        sym_X       = sym_feat.values.reshape(1, -1)
        sym_raw     = float(sym_model.predict_proba(sym_X)[0][1])
        sym_cal_p   = float(sym_cal.predict(np.array([sym_raw]))[0]) if sym_cal else sym_raw
        sym_nan     = int(np.isnan(sym_X).sum())
        status      = "✅ PASS" if sym_cal_p >= floor else "— below floor"
        results.append((sym, sym_raw, sym_cal_p, sym_nan, status))
    except Exception as e:
        results.append((sym, None, None, None, f"ERROR: {e}"))

print(f"{'Symbol':<14} {'Raw':>6} {'Cal':>6} {'NaN':>4}  Status")
print("-" * 55)
for sym, raw, cal, nan, status in results:
    if raw is None:
        print(f"{sym:<14} {'—':>6} {'—':>6} {'—':>4}  {status}")
    else:
        print(f"{sym:<14} {raw:>6.3f} {cal:>6.3f} {nan:>4}  {status}")

# ── 5. SUMMARY ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("5. SUMMARY")
print("=" * 60)

tradeable = sum(1 for r in results if r[1] is not None and r[2] is not None and r[2] >= floor)
print(f"Symbols checked   : {len(results)}")
print(f"Tradeable (≥floor): {tradeable}/{len(results)}")
print(f"Signal floor      : {floor}")
print()
print("Next step: run stage 8 live loop and monitor first 5 bars")
print("  python main.py --stage 8")