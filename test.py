import numpy as np
import pickle
from omegaconf import OmegaConf
from src.models.primary_model import load_model
from src.models.model_versioning import get_latest_model

cfg = OmegaConf.load("config/base.yaml")

entry = get_latest_model("BTCUSDT", "15m", model_type="primary")
model, calibrator = load_model("BTCUSDT", "15m", entry["version"], cfg.data.models_dir)

print(f"Calibrator type: {type(calibrator)}")

if calibrator is not None:
    test_inputs = np.linspace(0.1, 0.9, 17)
    test_outputs = calibrator.predict(test_inputs)
    print("\nCalibrator mapping (raw → calibrated):")
    for i, o in zip(test_inputs, test_outputs):
        bar = "█" * int(o * 40)
        print(f"  {i:.2f} → {o:.4f}  {bar}")
else:
    print("No calibrator found")

# Bonus: cek funding rank key mismatch
stats = pickle.load(open("data/checkpoints/cross_sectional_stats.pkl", "rb"))
funding_keys = [k for k in stats.keys() if "funding" in k]
print(f"\nFunding keys in cross_sectional_stats: {funding_keys}")