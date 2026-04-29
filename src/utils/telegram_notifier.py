import io
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.utils.logger import get_logger

logger = get_logger("telegram")

_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
_CHAT_ID   = os.environ.get("TELEGRAM_CHAT_ID", "")
_API_BASE  = "https://api.telegram.org/bot{token}/{method}"

# Per-event cooldown (seconds) — prevents flooding on repeated bars
_COOLDOWNS: dict[str, float] = {}
_COOLDOWN_LOCK = threading.Lock()
_DEFAULT_COOLDOWN = {
    "entry":       0,     # always send
    "exit":        0,     # always send
    "daily_pnl":  900,    # max 1/15min
    "alert":      300,    # max 1/5min per key
    "heartbeat": 3600,    # max 1/hr
}


def _enabled() -> bool:
    return bool(_BOT_TOKEN and _CHAT_ID)


def _throttled(event_key: str, cooldown_s: float) -> bool:
    now = time.monotonic()
    with _COOLDOWN_LOCK:
        last = _COOLDOWNS.get(event_key, 0.0)
        if now - last < cooldown_s:
            return True
        _COOLDOWNS[event_key] = now
    return False


def _post(method: str, **kwargs) -> bool:
    url = _API_BASE.format(token=_BOT_TOKEN, method=method)
    try:
        r = requests.post(url, timeout=10, **kwargs)
        if not r.ok:
            logger.warning(f"Telegram {method} failed {r.status_code}: {r.text[:200]}")
        return r.ok
    except Exception as exc:
        logger.debug(f"Telegram {method} error: {exc}")
        return False


def send_message(text: str, parse_mode: str = "HTML", disable_preview: bool = True) -> bool:
    if not _enabled():
        return False
    return _post("sendMessage", json={
        "chat_id": _CHAT_ID,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": disable_preview,
    })


def send_photo(image_bytes: bytes, caption: str = "", parse_mode: str = "HTML") -> bool:
    if not _enabled():
        return False
    return _post("sendPhoto",
                 data={"chat_id": _CHAT_ID, "caption": caption, "parse_mode": parse_mode},
                 files={"photo": ("chart.png", image_bytes, "image/png")})


def send_document(file_bytes: bytes, filename: str, caption: str = "") -> bool:
    if not _enabled():
        return False
    return _post("sendDocument",
                 data={"chat_id": _CHAT_ID, "caption": caption},
                 files={"document": (filename, file_bytes, "application/octet-stream")})


# ─── High-level event senders ────────────────────────────────────────────────

def notify_entry(symbol: str, direction: str, entry_price: float, size_usd: float,
                 tp_pct: float, sl_pct: float, signal_strength: float,
                 primary_prob: float, meta_prob: float, leverage: int,
                 regime: str = "", chart_bytes: bytes | None = None) -> None:
    if not _enabled():
        return
    dir_emoji = "📈 LONG" if direction.lower() == "long" else "📉 SHORT"
    tp_price  = entry_price * (1 + tp_pct) if direction.lower() == "long" else entry_price * (1 - tp_pct)
    sl_price  = entry_price * (1 - sl_pct) if direction.lower() == "long" else entry_price * (1 + sl_pct)
    ts = datetime.now(timezone.utc).strftime("%H:%M UTC")

    text = (
        f"<b>{dir_emoji}  {symbol}</b>  [{ts}]\n"
        f"Entry: <code>${entry_price:,.4f}</code>\n"
        f"Size:  <code>${size_usd:,.2f}</code> ({leverage}×)\n"
        f"TP:    <code>${tp_price:,.4f}</code>  (+{tp_pct:.1%})\n"
        f"SL:    <code>${sl_price:,.4f}</code>  (-{sl_pct:.1%})\n"
        f"Signal: <code>{signal_strength:.3f}</code>  "
        f"P(dir)={primary_prob:.3f}  P(meta)={meta_prob:.3f}\n"
        f"Regime: {regime or 'unknown'}"
    )
    if chart_bytes:
        send_photo(chart_bytes, caption=text)
    else:
        send_message(text)


def notify_exit(symbol: str, direction: str, entry_price: float, exit_price: float,
                pnl_pct: float, pnl_usd: float, exit_reason: str,
                bars_held: int, chart_bytes: bytes | None = None) -> None:
    if not _enabled():
        return
    if pnl_pct >= 0:
        emoji = "✅ WIN"
    else:
        emoji = "❌ LOSS"

    sign = "+" if pnl_pct >= 0 else ""
    ts   = datetime.now(timezone.utc).strftime("%H:%M UTC")
    text = (
        f"<b>{emoji}  {symbol}</b>  [{ts}]\n"
        f"Exit ({exit_reason}): <code>${exit_price:,.4f}</code>\n"
        f"P&amp;L: <b>{sign}{pnl_pct:.2%}</b>  ({sign}${pnl_usd:,.2f})\n"
        f"Entry: <code>${entry_price:,.4f}</code>  held {bars_held} bars"
    )
    if chart_bytes:
        send_photo(chart_bytes, caption=text)
    else:
        send_message(text)


def notify_daily_summary(equity: float, daily_pnl_pct: float, daily_pnl_usd: float,
                         n_trades: int, n_wins: int, open_positions: dict) -> None:
    if not _enabled():
        return
    if _throttled("daily_pnl", _DEFAULT_COOLDOWN["daily_pnl"]):
        return

    sign  = "+" if daily_pnl_pct >= 0 else ""
    emoji = "🟢" if daily_pnl_pct >= 0 else "🔴"
    open_txt = ""
    if open_positions:
        lines = []
        for sym, pos in open_positions.items():
            d = pos.get("direction", "?")
            upnl = float(pos.get("unrealized_pnl") or 0)
            lines.append(f"  • {sym} {d.upper()} unrealPnL={upnl:+.2f}")
        open_txt = "\nOpen now:\n" + "\n".join(lines)

    wr = f"{n_wins}/{n_trades}" if n_trades else "—"
    ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    text = (
        f"{emoji} <b>Daily summary</b>  [{ts}]\n"
        f"Wallet: <code>${equity:,.2f}</code>\n"
        f"Day P&amp;L: <b>{sign}{daily_pnl_pct:.2%}</b>  ({sign}${daily_pnl_usd:,.2f})\n"
        f"Trades: {wr} wins"
        + open_txt
    )
    send_message(text)


def notify_alert(key: str, message: str, cooldown_s: float | None = None) -> None:
    if not _enabled():
        return
    cd = cooldown_s if cooldown_s is not None else _DEFAULT_COOLDOWN["alert"]
    if _throttled(f"alert_{key}", cd):
        return
    send_message(f"⚠️ <b>ALERT [{key}]</b>\n{message}")


def notify_maintenance(status: str, detail: str = "") -> None:
    # status: "scheduled" | "waiting_positions" | "stopped" | "restarted" | "failed"
    icons = {
        "scheduled":         "🔧",
        "waiting_positions": "⏳",
        "stopped":           "🛑",
        "restarted":         "✅",
        "failed":            "🚨",
    }
    icon = icons.get(status, "🔧")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = f"{icon} <b>Maintenance [{status.upper()}]</b>  [{ts}]"
    if detail:
        msg += f"\n{detail}"
    notify_alert(f"maintenance_{status}", msg, cooldown_s=0)


def notify_heartbeat(equity: float, mode: str, open_count: int, demo_done: int, demo_req: int) -> None:
    if not _enabled():
        return
    if _throttled("heartbeat", _DEFAULT_COOLDOWN["heartbeat"]):
        return
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    send_message(
        f"💓 <b>Heartbeat</b>  [{ts}]\n"
        f"Mode: {mode}  Wallet: <code>${equity:,.2f}</code>\n"
        f"Open positions: {open_count}  Demo: {demo_done}/{demo_req}"
    )
