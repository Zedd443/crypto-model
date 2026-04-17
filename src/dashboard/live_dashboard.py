import threading
import time
from datetime import datetime, timezone

from src.utils.logger import get_logger

logger = get_logger("live_dashboard")

try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False

_ACTION_LABELS = {
    "ENTERED":    "ENTERED",
    "HOLD":       "HOLD",
    "SKIP_FLOOR": "low signal",
    "SKIP_LIMIT": "limit reached",
    "SKIP_DAILY": "daily target hit",
    "NO_SIGNAL":  "—",
}

_DIR_LABELS = {1: "LONG", -1: "SHORT", 0: "—"}


class LiveDashboard:
    def __init__(self, cfg):
        self._cfg = cfg
        self._data: dict = {}
        self._next_bar_ts: float = 0.0  # epoch seconds of next bar boundary
        self._lock = threading.Lock()
        self._live: "Live | None" = None
        self._ticker_thread: threading.Thread | None = None

        if _RICH_AVAILABLE:
            import sys
            self._console = Console(force_terminal=sys.stdout.isatty())
        else:
            self._console = None

    def set_next_bar(self, next_bar_epoch: float) -> None:
        with self._lock:
            self._next_bar_ts = next_bar_epoch

    def update(self, bar_data: dict) -> None:
        with self._lock:
            self._data = dict(bar_data)

    def render(self) -> None:
        if _RICH_AVAILABLE:
            self._render_rich()
        else:
            self._render_plain()

    def start_countdown(self, next_bar_epoch: float) -> None:
        """Start a background thread that reprints the countdown every second."""
        self.set_next_bar(next_bar_epoch)
        if not _RICH_AVAILABLE:
            return
        # Stop any existing ticker
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

    # ------------------------------------------------------------------
    # Countdown loop — runs in background thread during bar-wait
    # ------------------------------------------------------------------

    def _countdown_loop(self) -> None:
        with self._lock:
            data = dict(self._data)
            next_bar_ts = self._next_bar_ts

        if not _RICH_AVAILABLE:
            return

        console = Console()
        try:
            with Live(console=console, refresh_per_second=1, transient=True) as live:
                self._live = live
                while True:
                    now = time.time()
                    remaining = max(0.0, next_bar_ts - now)
                    if remaining <= 0:
                        break
                    mins, secs = divmod(int(remaining), 60)
                    countdown_str = f"{mins:02d}:{secs:02d}"

                    with self._lock:
                        data = dict(self._data)

                    live.update(self._build_waiting_panel(data, countdown_str))
                    time.sleep(1)
        except Exception:
            pass
        finally:
            self._live = None

    def _build_waiting_panel(self, d: dict, countdown: str) -> Panel:
        mode = str(d.get("mode", "DEMO")).upper()
        equity = float(d.get("equity", 0.0))
        daily_pnl_pct = float(d.get("daily_pnl_pct", 0.0))
        open_positions: dict = d.get("open_positions", {})
        demo_done = int(d.get("demo_trades_completed", 0))
        demo_req = int(d.get("demo_trades_required", 500))

        mode_style = "bold red" if mode == "MAINNET" else "cyan"
        pnl_style = "green" if daily_pnl_pct >= 0 else "red"

        header = Text.assemble(
            Text(f"[{mode}]", style=mode_style),
            "  Wallet: ",
            Text(f"${equity:,.2f}", style="bold white"),
            "  Daily P&L: ",
            Text(f"{daily_pnl_pct:+.2%}", style=pnl_style),
            f"  Demo: {demo_done}/{demo_req}",
            "  Next bar: ",
            Text(countdown, style="bold yellow"),
        )

        # Open positions mini-table
        pos_table = Table(show_header=True, header_style="bold white", box=None, padding=(0, 2))
        pos_table.add_column("Symbol", min_width=12)
        pos_table.add_column("Dir", min_width=6)
        pos_table.add_column("Entry (USDT)", min_width=14)
        pos_table.add_column("Volume (USDT)", min_width=14)
        pos_table.add_column("Unreal P&L", min_width=12)
        pos_table.add_column("TP", min_width=10)
        pos_table.add_column("SL", min_width=10)

        if open_positions:
            for sym, pos in open_positions.items():
                direction = str(pos.get("direction", ""))
                dir_style = "green" if direction == "long" else "red"
                entry_p = float(pos.get("entry_price") or 0)
                size_usd = float(pos.get("size_usd") or 0)
                tp_price = float(pos.get("tp_price") or 0)
                sl_price = float(pos.get("sl_price") or 0)
                upnl = float(pos.get("unrealized_pnl") or 0)
                upnl_style = "green" if upnl >= 0 else "red"
                pos_table.add_row(
                    sym,
                    Text(direction.upper(), style=dir_style),
                    f"${entry_p:,.4f}",
                    f"${size_usd:,.2f}",
                    Text(f"{upnl:+.2f}", style=upnl_style),
                    Text(f"${tp_price:,.4f}", style="green"),
                    Text(f"${sl_price:,.4f}", style="red"),
                )
        else:
            pos_table.add_row("[dim]no open positions[/dim]", "", "", "", "", "", "")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        title = Text(f"LIVE TRADING  [{timestamp}]", style="bold white")

        from rich.console import Group
        content = Group(header, Text(""), pos_table)
        return Panel(content, title=title, border_style="bright_blue")

    # ------------------------------------------------------------------
    # Full bar render (after bar processing completes)
    # ------------------------------------------------------------------

    def _render_rich(self) -> None:
        self.stop_countdown()
        d = self._data
        mode = str(d.get("mode", "DEMO")).upper()
        equity = float(d.get("equity", 0.0))
        daily_pnl_pct = float(d.get("daily_pnl_pct", 0.0))
        open_positions: dict = d.get("open_positions", {})
        signals: list = d.get("signals", [])
        demo_done = int(d.get("demo_trades_completed", 0))
        demo_req = int(d.get("demo_trades_required", 500))
        daily_target = float(d.get("daily_target_pct", 0.02))

        mode_style = "bold red" if mode == "MAINNET" else "cyan"
        pnl_style = "green" if daily_pnl_pct >= 0 else "red"
        gate_hit = daily_pnl_pct >= daily_target

        header = Text.assemble(
            Text(f"[{mode}]", style=mode_style),
            "  Wallet: ",
            Text(f"${equity:,.2f}", style="bold white"),
            "  Daily P&L: ",
            Text(f"{daily_pnl_pct:+.2%}", style=pnl_style),
            f"  Demo: {demo_done}/{demo_req}",
            "  Daily target: ",
            Text("HIT" if gate_hit else "open", style="yellow" if gate_hit else "green"),
        )

        # Open positions table
        pos_table = Table(show_header=True, header_style="bold white", box=None, padding=(0, 2))
        pos_table.add_column("Symbol", min_width=12)
        pos_table.add_column("Dir", min_width=6)
        pos_table.add_column("Entry (USDT)", min_width=14)
        pos_table.add_column("Volume (USDT)", min_width=14)
        pos_table.add_column("Unreal P&L", min_width=12)
        pos_table.add_column("TP", min_width=10)
        pos_table.add_column("SL", min_width=10)

        if open_positions:
            for sym, pos in open_positions.items():
                direction = str(pos.get("direction", ""))
                dir_style = "green" if direction == "long" else "red"
                entry_p = float(pos.get("entry_price") or 0)
                size_usd = float(pos.get("size_usd") or 0)
                tp_price = float(pos.get("tp_price") or 0)
                sl_price = float(pos.get("sl_price") or 0)
                upnl = float(pos.get("unrealized_pnl") or 0)
                upnl_style = "green" if upnl >= 0 else "red"
                pos_table.add_row(
                    sym,
                    Text(direction.upper(), style=dir_style),
                    f"${entry_p:,.4f}",
                    f"${size_usd:,.2f}",
                    Text(f"{upnl:+.2f}", style=upnl_style),
                    Text(f"${tp_price:,.4f}", style="green"),
                    Text(f"${sl_price:,.4f}", style="red"),
                )
        else:
            pos_table.add_row("[dim]no open positions[/dim]", "", "", "", "", "", "")

        # Signals table — only show ENTERED and HOLD, collapse the rest into a count
        sig_table = Table(show_header=True, header_style="bold white", box=None, padding=(0, 2))
        sig_table.add_column("Symbol", min_width=12)
        sig_table.add_column("Dir", min_width=6)
        sig_table.add_column("Prob", min_width=7)
        sig_table.add_column("Signal", min_width=7)
        sig_table.add_column("Action", min_width=16)

        entered = [s for s in signals if s.get("action") == "ENTERED"]
        held = [s for s in signals if s.get("action") == "HOLD"]
        skipped = [s for s in signals if s.get("action", "").startswith("SKIP") or s.get("action") == "NO_SIGNAL"]

        for s in entered + held:
            action = str(s.get("action", ""))
            direction_int = int(s.get("direction", 0))
            dir_label = _DIR_LABELS.get(direction_int, "—")
            dir_style = "green" if direction_int == 1 else ("red" if direction_int == -1 else "dim")
            action_style = "bold green" if action == "ENTERED" else "white"
            sig_table.add_row(
                str(s.get("symbol", "")),
                Text(dir_label, style=dir_style),
                f"{float(s.get('primary_prob') or 0):.3f}",
                f"{float(s.get('signal_strength') or 0):.3f}",
                Text(_ACTION_LABELS.get(action, action), style=action_style),
            )

        if skipped:
            sig_table.add_row(
                Text(f"[dim]{len(skipped)} symbols skipped (low signal / no signal)[/dim]"),
                "", "", "", "",
            )

        if not signals:
            sig_table.add_row("[dim]no signals this bar[/dim]", "", "", "", "")

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        title = Text(f"LIVE TRADING — BAR COMPLETE  [{timestamp}]", style="bold white")

        self._console.rule()
        self._console.print(Panel(header, title=title, border_style="bright_blue"))
        self._console.print(Panel(pos_table, title="[bold]OPEN POSITIONS[/bold]", border_style="blue"))
        self._console.print(Panel(sig_table, title="[bold]THIS BAR[/bold]", border_style="blue"))

    # ------------------------------------------------------------------
    # Plain terminal fallback
    # ------------------------------------------------------------------

    def _render_plain(self) -> None:
        d = self._data
        mode = str(d.get("mode", "DEMO")).upper()
        equity = float(d.get("equity", 0.0))
        daily_pnl_pct = float(d.get("daily_pnl_pct", 0.0))
        open_positions: dict = d.get("open_positions", {})
        signals: list = d.get("signals", [])
        demo_done = int(d.get("demo_trades_completed", 0))
        demo_req = int(d.get("demo_trades_required", 500))
        daily_target = float(d.get("daily_target_pct", 0.02))

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        WIDTH = 68
        sep = "=" * WIDTH
        print(sep)
        print(f"  [{mode}] {timestamp}")
        print(f"  Wallet: ${equity:,.2f}  Daily P&L: {daily_pnl_pct:+.2%}  Demo: {demo_done}/{demo_req}")
        print(sep)
        print("  OPEN POSITIONS")
        if open_positions:
            print(f"  {'Symbol':<12} {'Dir':<6} {'Entry':>12} {'Volume':>12} {'Unreal P&L':>12} {'TP':>10} {'SL':>10}")
            for sym, pos in open_positions.items():
                direction = str(pos.get("direction", "")).upper()
                entry_p = float(pos.get("entry_price") or 0)
                size_usd = float(pos.get("size_usd") or 0)
                tp_price = float(pos.get("tp_price") or 0)
                sl_price = float(pos.get("sl_price") or 0)
                upnl = float(pos.get("unrealized_pnl") or 0)
                print(f"  {sym:<12} {direction:<6} ${entry_p:>10,.4f} ${size_usd:>10,.2f} {upnl:>+11.2f} ${tp_price:>8,.4f} ${sl_price:>8,.4f}")
        else:
            print("  (no open positions)")
        print(sep)
        entered = [s for s in signals if s.get("action") == "ENTERED"]
        held = [s for s in signals if s.get("action") == "HOLD"]
        skipped = len([s for s in signals if s.get("action", "").startswith("SKIP") or s.get("action") == "NO_SIGNAL"])
        print(f"  THIS BAR  ({len(entered)} entered, {len(held)} holding, {skipped} skipped)")
        for s in entered + held:
            action = str(s.get("action", ""))
            dir_label = _DIR_LABELS.get(int(s.get("direction", 0)), "—")
            print(f"  {str(s.get('symbol','')):<12} {dir_label:<6} prob={float(s.get('primary_prob',0)):.3f}  sig={float(s.get('signal_strength',0)):.3f}  {action}")
        print(sep)
