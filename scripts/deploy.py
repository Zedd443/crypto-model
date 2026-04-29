"""
Single-command deploy tool.

Usage:
  python scripts/deploy.py                  # push code to GitHub (triggers push_deploy.yml)
  python scripts/deploy.py --retrain        # push + manually trigger weekly_retrain.yml on GitHub Actions
  python scripts/deploy.py --check          # preflight checks only, no push

Prerequisites (checked automatically):
  - git remote 'origin' configured
  - gh CLI installed and authenticated (gh auth status)
  - GitHub secrets set: VPS_HOST, VPS_USER, VPS_SSH_KEY, KAGGLE_USERNAME, KAGGLE_KEY,
                        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""

import argparse
import subprocess
import sys
from pathlib import Path


def _run(cmd: str, capture: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {cmd}", flush=True)
    return subprocess.run(
        cmd, shell=True, check=check,
        capture_output=capture, text=True
    )


def _check_gh() -> bool:
    r = _run("gh auth status", capture=True, check=False)
    return r.returncode == 0


def _check_remote() -> bool:
    r = _run("git remote get-url origin", capture=True, check=False)
    return r.returncode == 0 and r.stdout.strip()


def _check_secrets(required: list[str]) -> list[str]:
    r = _run("gh secret list --json name -q '.[].name'", capture=True, check=False)
    if r.returncode != 0:
        return required  # can't check — assume missing
    existing = set(r.stdout.strip().splitlines())
    return [s for s in required if s not in existing]


def preflight(trigger_retrain: bool) -> bool:
    ok = True
    print("\n=== Preflight checks ===")

    if not _check_remote():
        print("FAIL: git remote 'origin' not configured. Run: git remote add origin <url>")
        ok = False
    else:
        print("OK:   git remote origin")

    if not _check_gh():
        print("FAIL: gh CLI not authenticated. Run: gh auth login")
        ok = False
    else:
        print("OK:   gh CLI authenticated")

    required_secrets = [
        "VPS_HOST", "VPS_USER", "VPS_SSH_KEY",
        "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
    ]
    if trigger_retrain:
        required_secrets += ["KAGGLE_USERNAME", "KAGGLE_KEY"]

    missing = _check_secrets(required_secrets)
    if missing:
        print(f"FAIL: Missing GitHub secrets: {missing}")
        print("      Set them with: gh secret set <NAME>")
        ok = False
    else:
        print(f"OK:   GitHub secrets ({len(required_secrets)} checked)")

    # Check .env.example has required vars documented
    env_example = Path(".env.example")
    if not env_example.exists():
        print("WARN: .env.example not found")

    print()
    return ok


def push_code() -> None:
    print("\n=== Pushing code to GitHub ===")
    _run("git add -A")
    # Commit only if there are staged changes
    r = _run("git diff --cached --quiet", check=False)
    if r.returncode != 0:
        _run('git commit -m "chore: auto-deploy push"')
    _run("git push origin main")
    print("Code pushed. GitHub Actions push_deploy.yml will sync src/ to VPS.")


def trigger_retrain() -> None:
    print("\n=== Triggering weekly retrain on GitHub Actions ===")
    _run("gh workflow run weekly_retrain.yml --ref main")
    print("Retrain workflow triggered. Monitor at: gh run list --workflow=weekly_retrain.yml")
    print("Or watch live: gh run watch")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto model deploy tool")
    parser.add_argument("--retrain", action="store_true", help="Also trigger Kaggle weekly retrain")
    parser.add_argument("--check", action="store_true", help="Preflight checks only, no push")
    args = parser.parse_args()

    passed = preflight(trigger_retrain=args.retrain)
    if not passed:
        print("Preflight failed. Fix issues above before deploying.")
        sys.exit(1)

    if args.check:
        print("Preflight passed.")
        return

    push_code()

    if args.retrain:
        trigger_retrain()
        print("\nDone. The full retrain → hot-swap sequence is now running autonomously.")
        print("You will receive Telegram notifications at each stage.")
    else:
        print("\nDone. Code deployed to VPS via GitHub Actions.")


if __name__ == "__main__":
    main()
