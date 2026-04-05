python - <<'EOF'
import json
from pathlib import Path

p = Path("data/checkpoints/fracdiff/fracdiff_d_BTCUSDT_15m.json")
d = json.loads(p.read_text())
print("Fracdiff d-values (first 10):")
for k,v in list(d.items())[:10]:
    print(f"  {k}: d={v:.3f}")
EOF