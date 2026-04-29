"""
Kaggle weekly retrain script.

Run inside a Kaggle notebook kernel (Python script attached as dataset).
Required Kaggle secrets (add via notebook settings → secrets):
  GITHUB_TOKEN      — fine-grained PAT with repo read+write + release write
  GITHUB_REPO       — e.g. "irfandragneel/crypto_model"
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID
  BINANCE_API_KEY   — demo-fapi key (for stage 1 ingest)
  BINANCE_API_SECRET

The script:
  1. Clones the repo from GitHub
  2. Sets TRAIN_END_DATE to today-7d, VAL_START/END/TEST_START shifted proportionally
  3. Installs deps
  4. Resets stage statuses 1-5 to pending
  5. Runs pipeline stages 1→5
  6. Zips artifacts and uploads as a GitHub Release
  7. Sends Telegram notification on success or failure
"""

import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _env(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise RuntimeError(f"Required env var {key!r} not set. Add it as a Kaggle secret.")
    return val


def _run(cmd: str, cwd: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, cwd=cwd, check=check, text=True)


def _telegram(token: str, chat_id: str, msg: str) -> None:
    import urllib.request, urllib.parse
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}).encode()
    try:
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=10
        )
    except Exception as exc:
        print(f"Telegram send failed: {exc}", flush=True)


def _compute_date_splits(train_end: datetime) -> dict:
    # Val = 3 months after train_end, test = 1 month after val, all UTC
    val_start  = train_end + timedelta(days=1)
    val_end    = train_end + timedelta(days=90)
    test_start = val_end   + timedelta(days=1)
    return {
        "TRAIN_END_DATE": train_end.strftime("%Y-%m-%d"),
        "VAL_START_DATE": val_start.strftime("%Y-%m-%d"),
        "VAL_END_DATE":   val_end.strftime("%Y-%m-%d"),
        "TEST_START_DATE": test_start.strftime("%Y-%m-%d"),
    }


def _reset_stages(state_path: Path, stages: list[str]) -> None:
    state = json.loads(state_path.read_text())
    for stage in stages:
        if stage in state.get("stages", {}):
            state["stages"][stage]["status"] = "pending"
            state["stages"][stage]["completed_symbols"] = state["stages"][stage].get("completed_symbols", [])
            # Clear completed_symbols so all symbols are retrained
            state["stages"][stage]["completed_symbols"] = []
    state["next_scheduled_retrain"] = (
        datetime.now(timezone.utc) + timedelta(days=7)
    ).isoformat()
    state_path.write_text(json.dumps(state, indent=2))
    print(f"Reset stages: {stages}", flush=True)


def _zip_artifacts(repo_dir: Path, zip_path: Path) -> None:
    dirs_to_zip = ["models", "data/checkpoints", "project_state.json", "model_registry.json"]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in dirs_to_zip:
            p = repo_dir / item
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file():
                        zf.write(f, f.relative_to(repo_dir))
            elif p.is_file():
                zf.write(p, p.relative_to(repo_dir))
    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"Artifact zip: {zip_path} ({size_mb:.1f} MB)", flush=True)


def _upload_release(repo: str, token: str, zip_path: Path, tag: str) -> str:
    import urllib.request, json as _json
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
    }

    # Create or get release
    release_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    req = urllib.request.Request(release_url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            release = _json.loads(r.read())
    except Exception:
        # Create new release
        body = _json.dumps({"tag_name": tag, "name": f"Retrain {tag}", "draft": False, "prerelease": False}).encode()
        create_req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/releases",
            data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(create_req, timeout=30) as r:
            release = _json.loads(r.read())

    # Delete existing asset with same name if present
    upload_url = release["upload_url"].replace("{?name,label}", "")
    asset_name = zip_path.name
    for asset in release.get("assets", []):
        if asset["name"] == asset_name:
            del_req = urllib.request.Request(
                asset["url"], headers=headers, method="DELETE"
            )
            try:
                urllib.request.urlopen(del_req, timeout=15)
            except Exception:
                pass

    # Upload asset
    with open(zip_path, "rb") as f:
        data = f.read()
    upload_headers = dict(headers)
    upload_headers["Content-Type"] = "application/zip"
    upload_req = urllib.request.Request(
        f"{upload_url}?name={asset_name}", data=data, headers=upload_headers, method="POST"
    )
    with urllib.request.urlopen(upload_req, timeout=300) as r:
        asset = _json.loads(r.read())
    return asset["browser_download_url"]


def main() -> None:
    token     = _env("GITHUB_TOKEN")
    repo      = _env("GITHUB_REPO")
    tg_token  = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat   = os.environ.get("TELEGRAM_CHAT_ID", "")
    ts        = datetime.now(timezone.utc)
    tag       = f"retrain-{ts.strftime('%Y%m%d')}"
    work_dir  = Path("/kaggle/working")
    repo_dir  = work_dir / "crypto_model"
    zip_path  = work_dir / f"artifacts_{tag}.zip"

    try:
        # 1. Clone repo
        _run(f"git clone https://x-access-token:{token}@github.com/{repo}.git {repo_dir}", cwd=work_dir)

        # 2. Compute date splits (train_end = today - 7 days)
        train_end = ts - timedelta(days=7)
        splits = _compute_date_splits(train_end)
        for k, v in splits.items():
            os.environ[k] = v
        print(f"Date splits: {splits}", flush=True)

        # 3. Install deps
        _run(f"{sys.executable} -m pip install -r requirements.txt -q", cwd=repo_dir)

        # 4. Reset stages 1-5
        _reset_stages(repo_dir / "project_state.json", ["ingest", "features", "labels", "training", "meta_labeling"])

        # 5. Run pipeline stages 1→5
        env_str = " ".join(f"{k}={v}" for k, v in splits.items())
        _run(f"{env_str} {sys.executable} -m src.pipeline.run_pipeline --from-stage 1", cwd=repo_dir)

        # 6. Zip artifacts
        _zip_artifacts(repo_dir, zip_path)

        # 7. Upload to GitHub Release
        download_url = _upload_release(repo, token, zip_path, tag)
        print(f"Uploaded: {download_url}", flush=True)

        if tg_token and tg_chat:
            _telegram(tg_token, tg_chat,
                f"✅ <b>Kaggle retrain complete</b> [{ts.strftime('%Y-%m-%d %H:%M UTC')}]\n"
                f"Tag: <code>{tag}</code>\n"
                f"train_end: {splits['TRAIN_END_DATE']}\n"
                f"Artifact: {download_url}"
            )

    except Exception as exc:
        print(f"RETRAIN FAILED: {exc}", flush=True)
        if tg_token and tg_chat:
            _telegram(tg_token, tg_chat,
                f"🚨 <b>Kaggle retrain FAILED</b> [{ts.strftime('%Y-%m-%d %H:%M UTC')}]\n{exc}"
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
