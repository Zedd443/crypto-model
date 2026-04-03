"""
Fetch latest Kaggle kernel log via CLI and print it.
Usage:
  python kaggle_notebook/fetch_log.py           # all logs
  python kaggle_notebook/fetch_log.py --tail 50 # last N lines
"""
import sys
import subprocess
import json
import tempfile
from pathlib import Path

KERNEL = "irfandragneel/crypto-model-training"
LOG_FILE = "crypto-model-training.log"

tail = int(sys.argv[sys.argv.index("--tail") + 1]) if "--tail" in sys.argv else 0

with tempfile.TemporaryDirectory() as tmp:
    # Find kaggle in same Scripts dir as python, fallback to venv
    for candidate in [
        Path(sys.executable).parent / "kaggle",
        Path(sys.executable).parent / "kaggle.exe",
        Path(".venv/Scripts/kaggle"),
        Path(".venv/Scripts/kaggle.exe"),
    ]:
        if candidate.exists():
            kaggle_exe = str(candidate)
            break
    else:
        kaggle_exe = "kaggle"  # hope it's in PATH
    result = subprocess.run(
        [kaggle_exe, "kernels", "output", KERNEL, "-p", tmp],
        capture_output=True, text=True
    )
    log_path = Path(tmp) / LOG_FILE
    if not log_path.exists():
        # Try alternate name
        logs = list(Path(tmp).glob("*.log"))
        if logs:
            log_path = logs[0]
        else:
            print("No log file found. Output files:")
            for f in Path(tmp).iterdir():
                print(f"  {f.name}")
            sys.exit(1)

    raw = log_path.read_bytes()

# Log is newline-delimited JSON: {"stream_name":..., "time":..., "data":...}
lines = []
for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        text = obj.get("data", "").rstrip("\n")
        if text:
            lines.append(text)
    except Exception:
        lines.append(line.decode(errors="replace"))

output = lines[-tail:] if tail else lines
print("\n".join(output))
