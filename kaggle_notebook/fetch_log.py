"""
Fetch latest Kaggle kernel log and print it.
Usage: python kaggle_notebook/fetch_log.py [--tail N]
"""
import json
import base64
import sys
import urllib.request
from pathlib import Path

KERNEL = "irfandragneel/crypto-model-training"
KAGGLE_JSON = Path.home() / ".kaggle" / "kaggle.json"

cfg = json.loads(KAGGLE_JSON.read_text())
creds = base64.b64encode(f"{cfg['username']}:{cfg['key']}".encode()).decode()

tail = int(sys.argv[sys.argv.index("--tail") + 1]) if "--tail" in sys.argv else 0

url = f"https://www.kaggle.com/api/v1/kernels/{KERNEL}/output?type=log"
req = urllib.request.Request(url, headers={"Authorization": f"Basic {creds}"})

try:
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    # log is newline-delimited JSON: {"stream_name":..., "data":...}
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            lines.append(obj.get("data", "").rstrip("\n"))
        except Exception:
            lines.append(line.decode(errors="replace"))
    output = lines[-tail:] if tail else lines
    print("\n".join(output))
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code}: {e.reason}")
    print(e.read().decode(errors="replace"))
