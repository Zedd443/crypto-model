import threading
import time
from datetime import datetime, timezone

from src.utils.logger import get_logger

logger = get_logger("live_dashboard")

try:
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.rule import Rule
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_DIR_STYLE  = {1: ("LONG",  "bold green"), -1: ("SHORT", "bold red"), 0: ("—", "dim")}
_EXIT_STYLE = {"TP": "green", "SL": "red", "TIME": "yellow", "DMS": "magenta"}


class LiveDashboard:
    def __init__(self, cfg):
        self._cfg = cfg
        self._data: dict = {}
        self._next_bar_ts: float = 0.0
        self._lock = threading.Lock()
        self._live: "Live | None" = None
        self._ticker_thread: threading.Thread | None = None
        self._pnl_history: list[float] = []   # last N per-trade pnl_pct values
        self._equity_history: list[float] = []  # sampled equity per bar

        if _RICH_AVAILABLE:
            import sys
            self._console = Console(force_terminal=True, width=110)
        else:
            self._console = None

    def set_next_bar(self, next_bar_epoch: float) -> None:
        with self._lock:
            self._next_bar_ts = next_bar_epoch

    def update(self, bar_data: dict) -> None:
        with self._lock:
            self._data = dict(bar_data)
            eq = bar_data.get("equity")
            if eq:
                self._equity_history.append(float(eq))
                if len(self._equity_history) > 96:   # keep last 24h (96 × 15m)
                    self._equity_history.pop(0)
            for sig in bar_data.get("signals", []):
                if sig.get("action") == "ENTERED":
                    # track signal strength of entered trades for diagnostics
                    pass

    def record_trade_pnl(self, pnl_pct: float) -> None:
        with self._lock:
            self._pnl_history.append(pnl_pct)
            if len(self._pnl_history) > 50:
                self._pnl_history.pop(0)

    def render(self) -> None:
        if _RICH_AVAILABLE:
            self._render_rich()
        else:
            self._render_plain()

    def start_countdown(self, next_bar_epoch: float) -> None:
        self.set_next_bar(next_bar_epoch)
        if not _RICH_AVAILABLE:
            return
        self.stop_countdown()
        self._ticker_thread = threading.Thread(target=self._countdown_loop, daemon=True)
        self._ticker_thread.start()

    def stop_countdown(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    # ─────────────────────────────────────────────────────────────────────────
    # Countdown loop (runs in background thread during bar-wait)
    # ─────────────────────────────────────────────────────────────────────────

    def _countdown_loop(self) -> None:
        if not _RICH_AVAILABLE:
            return
        console = Console(force_terminal=True, width=110)
        try:
            with Live(console=console, refresh_per_second=1, transient=True) as live:
                self._live = live
                while True:
                    now       = time.time()
                    remaining = max(0.0, self._next_bar_ts - now)
                    if remaining <= 0:
                        break
                    with self._lock:
                        data = dict(self._data)
                        eq_hist = list(self._equity_history)
                        pnl_hist = list(self._pnl_history)
                    live.update(self._build_waiting_panel(data, remaining, eq_hist, pnl_hist))
                    time.sleep(1)
        except Exception:
            pass
        finally:
            self._live = None

    # ─────────────────────────────────────────────────────────────────────────
    # Panel builders
    # ─────────────────────────────────────────────────────────────────────────

    def _header_text(self, d: dict, countdown: str | None = None) -> Text:
        mode       = str(d.get("mode", "DEMO")).upper()
        equity     = float(d.get("equity", 0.0))
        daily_pnl  = float(d.get("daily_pnl_pct", 0.0))
        demo_done  = int(d.get("demo_trades_completed", 0))
        demo_req   = int(d.get("demo_trades_required", 500))
        daily_tgt  = float(d.get("daily_target_pct", 0.04))

        mode_style = "bold red on white" if mode == "MAINNET" else "bold cyan"
        pnl_style  = "bold green" if daily_pnl >= 0 else "bold red"
        sign       = "+" if daily_pnl >= 0 else ""

        # Daily PnL progress bar fraction
        pnl_frac   = min(abs(daily_pnl) / max(daily_tgt, 0.01), 1.0)

        t = Text()
        t.append(f" {mode} ", style=mode_style)
        t.append("  Wallet: ", style="white")
        t.append(f"${equity:,.2f}", style="bold white")
        t.append("  Day P&L: ", style="white")
        t.append(f"{sign}{daily_pnl:.2%}", style=pnl_style)
        t.append(f"  Demo: {demo_done}/{demo_req}", style="dim white")
        if countdown is not None:
            t.append("  Next bar: ", style="white")
            t.append(f"{_fmt_countdown(countdown)}", style="bold yellow")
        return t

    def _build_waiting_panel(self, d: dict, remaining: float, eq_hist: list, pnl_hist: list) -> Panel:
        mins, secs = divmod(int(remaining), 60)
        countdown_str = f"{mins:02d}:{secs:02d}"

        header = self._header_text(d, countdown_str)
        pos_table = self._build_positions_table(d.get("open_positions", {}))
        pnl_strip = self._build_pnl_strip(pnl_hist)
        eq_strip  = self._build_equity_mini(eq_hist)

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        return Panel(
            Group(header, Text(""), pos_table, Text(""), pnl_strip, eq_strip),
            title=f"[bold white]LIVE TRADING  [{ts}][/bold white]",
            border_style="bright_blue",
            padding=(0, 1),
        )

    def _build_positions_table(self, open_positions: dict) -> Table:
        t = Table(
            show_header=True, header_style="bold white",
            box=None, padding=(0, 2), expand=False,
        )
        t.add_column("Symbol",     min_width=14, style="bold white")
        t.add_column("Dir",        min_width=7)
        t.add_column("Entry",      min_width=12, justify="right")
        t.add_column("Size (USDT)",min_width=12, justify="right")
        t.add_column("Unreal P&L", min_width=12, justify="right")
        t.add_column("TP →",       min_width=12, justify="right")
        t.add_column("← SL",       min_width=12, justify="right")
        t.add_column("Signal",     min_width=8,  justify="right")

        if not open_positions:
            t.add_row("[dim]— no open positions —[/dim]", "", "", "", "", "", "", "")
            return t

        for sym, pos in open_positions.items():
            direction   = str(pos.get("direction", "")).lower()
            dir_int     = 1 if direction == "long" else -1
            dir_label, dir_style = _DIR_STYLE.get(dir_int, ("—", "dim"))
            entry_p     = float(pos.get("entry_price") or 0)
            size_usd    = float(pos.get("size_usd") or 0)
            tp_price    = float(pos.get("tp_price") or 0)
            sl_price    = float(pos.get("sl_price") or 0)
            upnl        = float(pos.get("unrealized_pnl") or 0)
            signal_str  = float(pos.get("signal_strength") or 0)

            upnl_style  = "green" if upnl >= 0 else "red"
            upnl_sign   = "+" if upnl >= 0 else ""

            # TP/SL distance indicators
            if entry_p > 0 and tp_price > 0:
                tp_pct = abs(tp_price / entry_p - 1)
                tp_txt = f"${tp_price:,.4f} (+{tp_pct:.1%})"
            else:
                tp_txt = f"${tp_price:,.4f}"
            if entry_p > 0 and sl_price > 0:
                sl_pct = abs(sl_price / entry_p - 1)
                sl_txt = f"${sl_price:,.4f} (-{sl_pct:.1%})"
            else:
                sl_txt = f"${sl_price:,.4f}"

            t.add_row(
                sym,
                Text(dir_label, style=dir_style),
                f"${entry_p:,.4f}",
                f"${size_usd:,.2f}",
                Text(f"{upnl_sign}{upnl:.2f}", style=upnl_style),
                Text(tp_txt, style="green"),
                Text(sl_txt, style="red"),
                f"{signal_str:.3f}",
            )
        return t

    def _build_pnl_strip(self, pnl_history: list) -> Text:
        """Mini sparkline of last N trade outcomes using block characters."""
        if not pnl_history:
            return Text("  Recent trades: —", style="dim")

        t = Text("  Recent trades: ")
        for pnl in pnl_history[-20:]:
            if pnl > 0.005:
                t.append("▇", style="bold green")
            elif pnl > 0:
                t.append("▄", style="green")
            elif pnl > -0.005:
                t.append("▄", style="red")
            else:
                t.append("▇", style="bold red")

        wins   = sum(1 for p in pnl_history if p > 0)
        total  = len(pnl_history)
        avg    = sum(pnl_history) / total
        sign   = "+" if avg >= 0 else ""
        wr_style = "green" if wins / total >= 0.5 else "red"
        t.append(f"  WR: ", style="dim white")
        t.append(f"{wins}/{total}", style=wr_style)
        t.append(f"  Avg: {sign}{avg:.2%}", style="green" if avg >= 0 else "red")
        return t

    def _build_equity_mini(self, eq_history: list) -> Text:
        """Tiny equity sparkline using Unicode block chars."""
        if len(eq_history) < 2:
            return Text("")
        lo, hi = min(eq_history), max(eq_history)
        span   = max(hi - lo, 1e-6)
        _BLOCKS = " ▁▂▃▄▅▆▇█"

        t = Text("  Equity 24h: ")
        for eq in eq_history[-40:]:
            frac  = (eq - lo) / span
            idx   = min(int(frac * 8), 8)
            style = "green" if eq >= eq_history[0] else "red"
            t.append(_BLOCKS[idx], style=style)

        last = eq_history[-1]
        first = eq_history[0]
        delta = last - first
        sign  = "+" if delta >= 0 else ""
        t.append(f"  {sign}${delta:,.2f}", style="green" if delta >= 0 else "red")
        return t

    # ─────────────────────────────────────────────────────────────────────────
    # Full bar render
    # ─────────────────────────────────────────────────────────────────────────

    def _render_rich(self) -> None:
        self.stop_countdown()
        with self._lock:
            d        = dict(self._data)
            eq_hist  = list(self._equity_history)
            pnl_hist = list(self._pnl_history)

        open_positions: dict  = d.get("open_positions", {})
        bar_signals: list     = d.get("signals", [])
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        self._console.rule(f"[bold white]BAR COMPLETE  {ts}[/bold white]", style="bright_blue")

        # ── Header ───────────────────────────────────────────────────────────
        self._console.print(Panel(self._header_text(d), border_style="bright_blue", padding=(0, 1)))

        # ── Open positions ────────────────────────────────────────────────────
        pos_table = self._build_positions_table(open_positions)
        n_open = len(open_positions)
        max_pos = int(getattr(getattr(self._cfg, "growth_gate", self._cfg), "max_open_positions", 2))
        self._console.print(Panel(
            pos_table,
            title=f"[bold white]OPEN POSITIONS  {n_open}/{max_pos}[/bold white]",
            border_style="blue",
            padding=(0, 1),
        ))

        # ── This bar signals ──────────────────────────────────────────────────
        entered = [s for s in bar_signals if s.get("action") == "ENTERED"]
        held    = [s for s in bar_signals if s.get("action") == "HOLD"]
        cands   = [s for s in bar_signals if s.get("action") == "CANDIDATE"]
        skipped = [s for s in bar_signals if s.get("action", "").startswith("SKIP")]
        no_sig  = [s for s in bar_signals if s.get("action") == "NO_SIGNAL"]

        sig_table = Table(
            show_header=True, header_style="bold white",
            box=None, padding=(0, 2), expand=False,
        )
        sig_table.add_column("Symbol",   min_width=14, style="bold white")
        sig_table.add_column("Action",   min_width=12)
        sig_table.add_column("Dir",      min_width=7)
        sig_table.add_column("P(dir)",   min_width=8, justify="right")
        sig_table.add_column("P(meta)",  min_width=8, justify="right")
        sig_table.add_column("Signal",   min_width=8, justify="right")
        sig_table.add_column("Regime",   min_width=12)

        for s in entered:
            _add_signal_row(sig_table, s, "ENTERED", "bold green")
        for s in cands:
            _add_signal_row(sig_table, s, "CANDIDATE", "yellow")
        for s in held:
            _add_signal_row(sig_table, s, "HOLD", "white")
        if skipped or no_sig:
            sig_table.add_row(
                Text(f"[dim]{len(skipped)} skipped (low signal)  {len(no_sig)} no signal[/dim]"),
                "", "", "", "", "", "",
            )

        self._console.print(Panel(
            sig_table,
            title=f"[bold white]THIS BAR  {len(entered)} entered · {len(held)} holding · {len(skipped)+len(no_sig)} skipped[/bold white]",
            border_style="blue",
            padding=(0, 1),
        ))

        # ── PnL strip + equity sparkline ──────────────────────────────────────
        bottom = Group(self._build_pnl_strip(pnl_hist), self._build_equity_mini(eq_hist))
        self._console.print(Panel(bottom, border_style="dim blue", padding=(0, 1)))

    # ─────────────────────────────────────────────────────────────────────────
    # Plain fallback
    # ─────────────────────────────────────────────────────────────────────────

    def _render_plain(self) -> None:
        d = self._data
        mode      = str(d.get("mode", "DEMO")).upper()
        equity    = float(d.get("equity", 0.0))
        daily_pnl = float(d.get("daily_pnl_pct", 0.0))
        demo_done = int(d.get("demo_trades_completed", 0))
        demo_req  = int(d.get("demo_trades_required", 500))
        open_pos  = d.get("open_positions", {})
        signals   = d.get("signals", [])

        W = 72
        sep = "═" * W
        ts  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(sep)
        print(f"  [{mode}]  {ts}")
        print(f"  Wallet: ${equity:,.2f}   Day P&L: {daily_pnl:+.2%}   Demo: {demo_done}/{demo_req}")
        print(sep)
        print("  OPEN POSITIONS")
        if open_pos:
            hdr = f"  {'Symbol':<13} {'Dir':<6} {'Entry':>12} {'Size':>10} {'UPnL':>10} {'TP':>12} {'SL':>12}"
            print(hdr)
            print("  " + "─" * (W - 2))
            for sym, pos in open_pos.items():
                direction = str(pos.get("direction", "")).upper()
                entry_p   = float(pos.get("entry_price") or 0)
                size_usd  = float(pos.get("size_usd") or 0)
                tp_price  = float(pos.get("tp_price") or 0)
                sl_price  = float(pos.get("sl_price") or 0)
                upnl      = float(pos.get("unrealized_pnl") or 0)
                print(f"  {sym:<13} {direction:<6} ${entry_p:>10,.4f} ${size_usd:>8,.2f} {upnl:>+9.2f} ${tp_price:>10,.4f} ${sl_price:>10,.4f}")
        else:
            print("  — no open positions —")
        print(sep)
        entered = [s for s in signals if s.get("action") == "ENTERED"]
        held    = [s for s in signals if s.get("action") == "HOLD"]
        skipped = len([s for s in signals if s.get("action","").startswith("SKIP") or s.get("action") == "NO_SIGNAL"])
        print(f"  THIS BAR: {len(entered)} entered · {len(held)} holding · {skipped} skipped")
        for s in entered + held:
            sym    = s.get("symbol", "")
            dir_i  = int(s.get("direction", 0))
            label  = "LONG" if dir_i == 1 else ("SHORT" if dir_i == -1 else "—")
            prob   = float(s.get("primary_prob", 0))
            sig    = float(s.get("signal_strength", 0))
            action = s.get("action", "")
            print(f"  {sym:<13} {label:<6} P={prob:.3f} S={sig:.3f}  [{action}]")
        print(sep)

        # Compact PnL history
        with self._lock:
            pnl_hist = list(self._pnl_history)
        if pnl_hist:
            wins  = sum(1 for p in pnl_hist if p > 0)
            total = len(pnl_hist)
            avg   = sum(pnl_hist) / total
            print(f"  Recent {total} trades: WR={wins}/{total}  Avg={avg:+.2%}")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_countdown(remaining: float) -> str:
    if isinstance(remaining, float):
        mins, secs = divmod(int(remaining), 60)
        return f"{mins:02d}:{secs:02d}"
    return str(remaining)


def _add_signal_row(table: "Table", s: dict, action_label: str, action_style: str) -> None:
    sym      = str(s.get("symbol", ""))
    dir_int  = int(s.get("direction", 0))
    dir_lbl, dir_style = _DIR_STYLE.get(dir_int, ("—", "dim"))
    prob     = float(s.get("primary_prob") or 0)
    meta     = float(s.get("meta_prob") or 0)
    sig      = float(s.get("signal_strength") or 0)
    regime   = str(s.get("regime", "unknown"))

    # Signal strength colour
    if sig >= 0.65:
        sig_style = "bold green"
    elif sig >= 0.50:
        sig_style = "green"
    elif sig >= 0.35:
        sig_style = "yellow"
    else:
        sig_style = "dim"

    table.add_row(
        sym,
        Text(action_label, style=action_style),
        Text(dir_lbl, style=dir_style),
        f"{prob:.3f}",
        f"{meta:.3f}",
        Text(f"{sig:.3f}", style=sig_style),
        Text(regime, style="dim cyan"),
    )
