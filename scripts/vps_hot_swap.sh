#!/usr/bin/env bash
# VPS hot-swap script — called by GitHub Actions after Kaggle retrain completes.
# Invariant: if ANY step fails, stage 8 is NOT restarted and old models are kept.
#
# Required env vars (set by GitHub Actions secrets):
#   ARTIFACT_URL        — GitHub Release download URL for the artifact zip
#   TELEGRAM_BOT_TOKEN  — for status notifications
#   TELEGRAM_CHAT_ID
#   PROJECT_DIR         — absolute path to crypto_model on VPS (default: ~/crypto_model)
#   SERVICE_NAME        — systemd service name (default: crypto-live)

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/crypto_model}"
SERVICE_NAME="${SERVICE_NAME:-crypto-live}"
FLAG_FILE="$PROJECT_DIR/.retrain_pending"
ARTIFACT_URL="${ARTIFACT_URL:?ARTIFACT_URL must be set}"
TG_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TG_CHAT="${TELEGRAM_CHAT_ID:-}"
MAX_WAIT_SECONDS=21600  # 6 hours = max_hold_bars × 15m

_tg() {
    local msg="$1"
    if [[ -n "$TG_TOKEN" && -n "$TG_CHAT" ]]; then
        curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
            -d "chat_id=${TG_CHAT}" \
            -d "parse_mode=HTML" \
            --data-urlencode "text=${msg}" > /dev/null 2>&1 || true
    fi
}

_fail() {
    local msg="$1"
    echo "ERROR: $msg" >&2
    _tg "🚨 <b>Hot-swap FAILED</b>: $msg — VPS still running old models."
    exit 1
}

echo "=== VPS Hot-Swap Started ==="

# Step 1: Signal stage 8 to wait for position closure
echo "Touching retrain flag: $FLAG_FILE"
touch "$FLAG_FILE"
_tg "🔧 <b>Maintenance SCHEDULED</b>: Retrain flag set. Waiting for open positions to close before model swap."

# Step 2: Wait for stage 8 to exit (polls every 30s, max MAX_WAIT_SECONDS)
echo "Waiting for stage 8 to stop (max ${MAX_WAIT_SECONDS}s)..."
waited=0
while systemctl is-active --quiet "$SERVICE_NAME" 2>/dev/null; do
    if [[ $waited -ge $MAX_WAIT_SECONDS ]]; then
        rm -f "$FLAG_FILE"
        _fail "Stage 8 did not stop within ${MAX_WAIT_SECONDS}s — aborting hot-swap"
    fi
    sleep 30
    waited=$((waited + 30))
    echo "  Still waiting... ${waited}s elapsed"
done
echo "Stage 8 stopped after ${waited}s."

# Step 3: Download artifact zip
TMP_ZIP=$(mktemp /tmp/artifacts_XXXXXX.zip)
echo "Downloading artifact from: $ARTIFACT_URL"
curl -fsSL -H "Authorization: token ${GITHUB_TOKEN:-}" \
    -L "$ARTIFACT_URL" -o "$TMP_ZIP" \
    || _fail "Artifact download failed from $ARTIFACT_URL"

# Step 4: Backup current models (keeps last 1 backup)
BACKUP_DIR="$PROJECT_DIR/../crypto_model_backup"
echo "Backing up models to $BACKUP_DIR"
rm -rf "$BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp -r "$PROJECT_DIR/models" "$BACKUP_DIR/" 2>/dev/null || true
cp "$PROJECT_DIR/project_state.json" "$BACKUP_DIR/" 2>/dev/null || true

# Step 5: Extract artifact
echo "Extracting artifact to $PROJECT_DIR"
cd "$PROJECT_DIR"
unzip -o "$TMP_ZIP" -d "$PROJECT_DIR" \
    || _fail "Artifact extraction failed"
rm -f "$TMP_ZIP"

# Step 6: Remove flag and restart service
rm -f "$FLAG_FILE"
echo "Restarting $SERVICE_NAME..."
systemctl restart "$SERVICE_NAME" \
    || _fail "systemctl restart $SERVICE_NAME failed — check: journalctl -u $SERVICE_NAME -n 50"

# Step 7: Verify restart
sleep 10
if ! systemctl is-active --quiet "$SERVICE_NAME"; then
    _fail "$SERVICE_NAME failed to start after hot-swap — rolled-back models may still be stale. Check journalctl."
fi

echo "=== Hot-Swap Complete ==="
_tg "✅ <b>Maintenance RESTARTED</b>: Model hot-swap complete. Stage 8 is live with new models."
