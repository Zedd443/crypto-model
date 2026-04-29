"""
Trade chart generator — produces a candlestick chart with EMA 9/21, entry/exit markers,
TP/SL levels, volume bars, and signal strength annotation.
Outputs PNG bytes suitable for Telegram sendPhoto or local file save.
"""
import io
from datetime import timezone

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("trade_chart")

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — safe for threads
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    logger.warning("matplotlib not available — charts disabled")


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _candle_colors(open_: pd.Series, close: pd.Series):
    return ["#26a69a" if c >= o else "#ef5350" for o, c in zip(open_, close)]


def generate_entry_chart(
    klines_df: pd.DataFrame,
    symbol: str,
    direction: str,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    signal_strength: float,
    primary_prob: float,
    meta_prob: float,
    n_bars_context: int = 80,
) -> bytes | None:
    """
    Returns PNG bytes of a candlestick chart (last n_bars_context bars before entry).
    klines_df must have columns: open, high, low, close, volume with UTC DatetimeIndex.
    """
    if not _MPL_OK:
        return None
    try:
        return _render_entry_chart(
            klines_df, symbol, direction, entry_price, tp_price, sl_price,
            signal_strength, primary_prob, meta_prob, n_bars_context
        )
    except Exception as exc:
        logger.warning(f"Chart generation failed for {symbol}: {exc}")
        return None


def generate_exit_chart(
    klines_df: pd.DataFrame,
    symbol: str,
    direction: str,
    entry_price: float,
    exit_price: float,
    entry_time,
    exit_time,
    tp_price: float,
    sl_price: float,
    pnl_pct: float,
    exit_reason: str,
    n_bars_context: int = 80,
) -> bytes | None:
    if not _MPL_OK:
        return None
    try:
        return _render_exit_chart(
            klines_df, symbol, direction, entry_price, exit_price,
            entry_time, exit_time, tp_price, sl_price, pnl_pct, exit_reason, n_bars_context
        )
    except Exception as exc:
        logger.warning(f"Exit chart generation failed for {symbol}: {exc}")
        return None


def generate_equity_curve_chart(
    trade_log_path,
    equity_start: float,
    n_days: int = 7,
) -> bytes | None:
    if not _MPL_OK:
        return None
    try:
        return _render_equity_curve(trade_log_path, equity_start, n_days)
    except Exception as exc:
        logger.warning(f"Equity curve chart failed: {exc}")
        return None


# ─── Internal renderers ───────────────────────────────────────────────────────

_DARK_BG  = "#1a1a2e"
_PANEL_BG = "#16213e"
_GRID_CLR = "#2a2a4a"
_TEXT_CLR = "#e0e0e0"
_LONG_CLR = "#26a69a"   # teal
_SHORT_CLR= "#ef5350"   # red
_TP_CLR   = "#26a69a"
_SL_CLR   = "#ef5350"
_EMA9_CLR = "#ffeb3b"   # yellow
_EMA21_CLR= "#ff9800"   # orange
_VOL_UP   = "#26a69a44"
_VOL_DOWN = "#ef535044"


def _setup_dark_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":  _DARK_BG,
        "axes.facecolor":    _PANEL_BG,
        "axes.edgecolor":    _GRID_CLR,
        "axes.labelcolor":   _TEXT_CLR,
        "xtick.color":       _TEXT_CLR,
        "ytick.color":       _TEXT_CLR,
        "text.color":        _TEXT_CLR,
        "grid.color":        _GRID_CLR,
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "font.size":         8,
    })


def _draw_candles(ax, df: pd.DataFrame, bar_width: float = 0.6):
    colors = _candle_colors(df["open"], df["close"])
    x = np.arange(len(df))
    for i, (idx, row) in enumerate(df.iterrows()):
        c = colors[i]
        # Wick
        ax.plot([i, i], [row["low"], row["high"]], color=c, linewidth=0.8, zorder=2)
        # Body
        body_lo = min(row["open"], row["close"])
        body_hi = max(row["open"], row["close"])
        body_h  = max(body_hi - body_lo, row["close"] * 0.0001)
        ax.bar(i, body_h, bottom=body_lo, width=bar_width,
               color=c, edgecolor=c, linewidth=0, zorder=3)


def _draw_ema(ax, close: pd.Series, span: int, color: str, label: str):
    ema = _ema(close, span).values
    x   = np.arange(len(close))
    ax.plot(x, ema, color=color, linewidth=1.2, label=label, zorder=4)


def _format_xticks(ax, df: pd.DataFrame, n_ticks: int = 6):
    step = max(1, len(df) // n_ticks)
    positions = list(range(0, len(df), step))
    labels = [df.index[i].strftime("%m/%d %H:%M") for i in positions]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_entry_chart(
    klines_df, symbol, direction, entry_price, tp_price, sl_price,
    signal_strength, primary_prob, meta_prob, n_bars_context
) -> bytes:
    _setup_dark_style()

    df = klines_df.tail(n_bars_context).copy()
    close = df["close"]
    volume = df["volume"]
    x_last = len(df) - 1

    fig = plt.figure(figsize=(12, 7), facecolor=_DARK_BG)
    gs  = GridSpec(3, 1, figure=fig, height_ratios=[4, 1, 0.6], hspace=0.08)

    # ── Price panel ──────────────────────────────────────────────────────────
    ax_price = fig.add_subplot(gs[0])
    _draw_candles(ax_price, df)
    _draw_ema(ax_price, close, 9,  _EMA9_CLR,  "EMA 9")
    _draw_ema(ax_price, close, 21, _EMA21_CLR, "EMA 21")

    # Entry marker
    dir_color = _LONG_CLR if direction.lower() == "long" else _SHORT_CLR
    dir_marker = "^" if direction.lower() == "long" else "v"
    ax_price.scatter(x_last, entry_price, marker=dir_marker, color=dir_color,
                     s=120, zorder=6, label=f"Entry {direction.upper()}")

    # TP / SL horizontal lines
    ax_price.axhline(tp_price, color=_TP_CLR, linestyle="--", linewidth=1.0,
                     alpha=0.8, label=f"TP {tp_price:,.4f}")
    ax_price.axhline(sl_price, color=_SL_CLR, linestyle="--", linewidth=1.0,
                     alpha=0.8, label=f"SL {sl_price:,.4f}")

    # TP/SL zone fill
    y_lo = min(sl_price, entry_price)
    y_hi = max(tp_price, entry_price)
    ax_price.fill_between([x_last - 0.5, x_last + 1.5], sl_price, entry_price,
                          color=_SL_CLR, alpha=0.08)
    ax_price.fill_between([x_last - 0.5, x_last + 1.5], entry_price, tp_price,
                          color=_TP_CLR, alpha=0.08)

    ax_price.set_xlim(-1, len(df) + 1)
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left", fontsize=7, framealpha=0.3)
    _format_xticks(ax_price, df)
    ax_price.set_xticklabels([])  # hide — shown on volume panel

    # Title
    ts_str = df.index[-1].strftime("%Y-%m-%d %H:%M UTC")
    ax_price.set_title(
        f"{symbol}  {direction.upper()}  |  Entry ${entry_price:,.4f}  "
        f"TP +{(tp_price/entry_price - 1)*100:.2f}%  SL -{abs(sl_price/entry_price - 1)*100:.2f}%\n"
        f"Signal={signal_strength:.3f}  P(dir)={primary_prob:.3f}  P(meta)={meta_prob:.3f}  [{ts_str}]",
        color=_TEXT_CLR, fontsize=9, pad=6
    )

    # ── Volume panel ─────────────────────────────────────────────────────────
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    vol_colors = [_VOL_UP if c >= o else _VOL_DOWN
                  for o, c in zip(df["open"], df["close"])]
    ax_vol.bar(range(len(df)), volume, color=vol_colors, width=0.8, linewidth=0)
    ax_vol.set_ylabel("Volume", fontsize=7)
    ax_vol.grid(True, alpha=0.2)
    _format_xticks(ax_vol, df)

    # ── Signal strength bar ───────────────────────────────────────────────────
    ax_sig = fig.add_subplot(gs[2])
    bar_color = _LONG_CLR if signal_strength >= 0.55 else "#ffeb3b"
    ax_sig.barh(0, signal_strength, color=bar_color, height=0.5)
    ax_sig.barh(0, 1.0, color=_GRID_CLR, height=0.5, alpha=0.3)
    ax_sig.axvline(0.55, color="#ffeb3b", linewidth=0.8, linestyle=":")
    ax_sig.set_xlim(0, 1)
    ax_sig.set_yticks([])
    ax_sig.set_xlabel("Signal strength", fontsize=7)
    ax_sig.text(signal_strength + 0.01, 0, f"{signal_strength:.3f}", va="center", fontsize=7)

    plt.tight_layout()
    return _fig_to_bytes(fig)


def _render_exit_chart(
    klines_df, symbol, direction, entry_price, exit_price,
    entry_time, exit_time, tp_price, sl_price, pnl_pct, exit_reason, n_bars_context
) -> bytes:
    _setup_dark_style()

    df = klines_df.tail(n_bars_context).copy()
    close = df["close"]
    x_last = len(df) - 1

    # Find entry bar index (approximate)
    entry_x = 0
    if entry_time is not None and hasattr(df.index, "searchsorted"):
        try:
            entry_ts = pd.Timestamp(entry_time, tz="UTC") if not hasattr(entry_time, "tz") else entry_time
            pos = df.index.searchsorted(entry_ts)
            entry_x = max(0, min(pos, len(df) - 1))
        except Exception:
            entry_x = max(0, x_last - 20)

    fig = plt.figure(figsize=(12, 6), facecolor=_DARK_BG)
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[4, 1], hspace=0.08)

    ax_price = fig.add_subplot(gs[0])
    _draw_candles(ax_price, df)
    _draw_ema(ax_price, close, 9,  _EMA9_CLR,  "EMA 9")
    _draw_ema(ax_price, close, 21, _EMA21_CLR, "EMA 21")

    dir_color  = _LONG_CLR if direction.lower() == "long" else _SHORT_CLR
    dir_marker = "^" if direction.lower() == "long" else "v"

    # Entry marker
    ax_price.scatter(entry_x, entry_price, marker=dir_marker, color=dir_color,
                     s=100, zorder=6, label=f"Entry ${entry_price:,.4f}")

    # Exit marker
    exit_color  = _LONG_CLR if pnl_pct >= 0 else _SHORT_CLR
    exit_marker = "D"
    ax_price.scatter(x_last, exit_price, marker=exit_marker, color=exit_color,
                     s=100, zorder=6, label=f"Exit ({exit_reason}) ${exit_price:,.4f}")

    # Connect entry to exit
    ax_price.plot([entry_x, x_last], [entry_price, exit_price],
                  color=exit_color, linewidth=0.8, linestyle=":", alpha=0.6)

    ax_price.axhline(tp_price, color=_TP_CLR, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_price.axhline(sl_price, color=_SL_CLR, linestyle="--", linewidth=0.8, alpha=0.6)

    sign = "+" if pnl_pct >= 0 else ""
    pnl_emoji = "✅" if pnl_pct >= 0 else "❌"
    ax_price.set_title(
        f"{pnl_emoji}  {symbol}  {direction.upper()}  |  {exit_reason}  |  "
        f"P&L {sign}{pnl_pct:.2%}\n"
        f"Entry ${entry_price:,.4f}  →  Exit ${exit_price:,.4f}  "
        f"TP ${tp_price:,.4f}  SL ${sl_price:,.4f}",
        color=exit_color if abs(pnl_pct) > 0.002 else _TEXT_CLR,
        fontsize=9, pad=6
    )
    ax_price.set_xlim(-1, len(df) + 1)
    ax_price.grid(True, alpha=0.3)
    ax_price.legend(loc="upper left", fontsize=7, framealpha=0.3)
    ax_price.set_xticklabels([])

    # Volume
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    vol_colors = [_VOL_UP if c >= o else _VOL_DOWN
                  for o, c in zip(df["open"], df["close"])]
    ax_vol.bar(range(len(df)), df["volume"], color=vol_colors, width=0.8, linewidth=0)
    ax_vol.set_ylabel("Volume", fontsize=7)
    ax_vol.grid(True, alpha=0.2)
    _format_xticks(ax_vol, df)

    plt.tight_layout()
    return _fig_to_bytes(fig)


def _render_equity_curve(trade_log_path, equity_start: float, n_days: int) -> bytes:
    import pandas as pd
    from pathlib import Path

    path = Path(trade_log_path)
    if not path.exists():
        return None

    df = pd.read_csv(path, parse_dates=["timestamp_entry", "timestamp_exit"])
    if df.empty:
        return None

    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=n_days)
    df = df[df["timestamp_exit"] >= cutoff].copy()
    if df.empty:
        return None

    df = df.sort_values("timestamp_exit")
    df["cum_pnl_usd"] = (df["pnl_pct"] * df["size_usd"]).cumsum()
    df["equity_curve"] = equity_start + df["cum_pnl_usd"]

    _setup_dark_style()
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), facecolor=_DARK_BG, gridspec_kw={"height_ratios": [3, 1]})
    fig.subplots_adjust(hspace=0.1)

    ax_eq, ax_pnl = axes

    # Equity curve
    ax_eq.plot(df["timestamp_exit"], df["equity_curve"], color="#64b5f6", linewidth=1.5)
    ax_eq.fill_between(df["timestamp_exit"], equity_start, df["equity_curve"],
                       where=df["equity_curve"] >= equity_start,
                       color=_LONG_CLR, alpha=0.15)
    ax_eq.fill_between(df["timestamp_exit"], equity_start, df["equity_curve"],
                       where=df["equity_curve"] < equity_start,
                       color=_SHORT_CLR, alpha=0.15)
    ax_eq.axhline(equity_start, color=_GRID_CLR, linewidth=0.8, linestyle="--")
    ax_eq.set_ylabel("Equity (USDT)", fontsize=8)
    ax_eq.grid(True, alpha=0.3)
    ax_eq.set_title(f"Equity Curve — last {n_days}d", color=_TEXT_CLR, fontsize=10)
    ax_eq.set_xticklabels([])

    # Per-trade P&L bars
    colors = [_LONG_CLR if p >= 0 else _SHORT_CLR for p in df["pnl_pct"]]
    ax_pnl.bar(range(len(df)), df["pnl_pct"] * 100, color=colors, width=0.8, linewidth=0)
    ax_pnl.axhline(0, color=_GRID_CLR, linewidth=0.8)
    ax_pnl.set_ylabel("P&L %", fontsize=8)
    ax_pnl.set_xlabel("Trade #", fontsize=8)
    ax_pnl.grid(True, alpha=0.2)

    plt.tight_layout()
    return _fig_to_bytes(fig)
