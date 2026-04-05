import pickle, pandas as pd
from pathlib import Path
from src.execution.live_features import compute_live_features
from src.execution.binance_client import BinanceClient
from omegaconf import OmegaConf

cfg = OmegaConf.load("config/base.yaml")
client = BinanceClient(cfg)

sym = "BTCUSDT"
klines = client.get_klines(sym, "15m", limit=600)
k1h = client.get_klines(sym, "1h", limit=200)
k4h = client.get_klines(sym, "4h", limit=100)
k1d = client.get_klines(sym, "1d", limit=60)
btc  = klines  # BTC is its own reference

feat = compute_live_features(sym, cfg, klines, klines_1h=k1h, klines_4h=k4h, klines_1d=k1d, btc_klines_15m=btc)

nan_cols = feat[feat.isna()].index.tolist()
print(f"Total features : {len(feat)}")
print(f"NaN count      : {len(nan_cols)}")
print(f"NaN cols sample: {nan_cols[:15]}")
print(f"Signal preview : primary_prob will depend on these ^")

import json
from pathlib import Path

p = Path("data/checkpoints/fracdiff/fracdiff_d_BTCUSDT_15m.json")
d = json.loads(p.read_text())
print("Fracdiff d-values (first 10):")
for k,v in list(d.items())[:10]:
    print(f"  {k}: d={v:.3f}")

import numpy as np
from pathlib import Path
from src.models.model_versioning import get_latest_model
from src.models.primary_model import load_model
from src.execution.live_features import compute_live_features
from src.execution.binance_client import BinanceClient
from omegaconf import OmegaConf

cfg = OmegaConf.load("config/base.yaml")
client = BinanceClient(cfg)
sym = "BTCUSDT"

klines = client.get_klines(sym, "15m", limit=600)
k1h = client.get_klines(sym, "1h", limit=200)
k4h = client.get_klines(sym, "4h", limit=100)
k1d = client.get_klines(sym, "1d", limit=60)

feat = compute_live_features(sym, cfg, klines, klines_1h=k1h, klines_4h=k4h, klines_1d=k1d, btc_klines_15m=klines)

entry = get_latest_model(sym, "15m", model_type="primary")
model, calibrator = load_model(sym, "15m", entry["version"], cfg.data.models_dir)

X = feat.values.reshape(1, -1)
raw_prob = float(model.predict_proba(X)[0][1])
if calibrator is not None:
    cal_prob = float(calibrator.predict(np.array([raw_prob]))[0])
else:
    cal_prob = raw_prob

print(f"Raw prob   : {raw_prob:.4f}")
print(f"Cal prob   : {cal_prob:.4f}")
print(f"NaN in X   : {np.isnan(X).sum()}")
print(f"Signal vs floor: {cal_prob:.3f} vs {cfg.portfolio.signal_floor_prob} — {'PASS' if cal_prob >= cfg.portfolio.signal_floor_prob else 'BELOW FLOOR'}")