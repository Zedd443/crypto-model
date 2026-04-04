from datetime import datetime, timezone

from src.utils.logger import get_logger

logger = get_logger("live_dashboard")

# Try to import rich; fall back to plain terminal if unavailable
try:
    from rich.columns import Columns
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False
    logger.warning("rich not installed — dashboard will use plain terminal fallback")


_ACTION_LABELS = {
    "ENTERED":    "ENTERED",
    "HOLD":       "HOLD (in pos)",
    "SKIP_FLOOR": "SKIP (low sig)",
    "SKIP_LIMIT": "SKIP (limit)",
    "SKIP_DAILY": "SKIP (daily tgt)",
    "NO_SIGNAL":  "no signal",
}

_DIR_LABELS = {1: "LONG", -1: "SHORT", 0: "FLAT"}


class LiveDashboard:
    def __init__(self, cfg):
        self._cfg = cfg
        self._data: dict = {}
        if _RICH_AVAILABLE:
            self._console = Console()
        else:
            self._console = None

    def update(self, bar_data: dict) -> None:
        # Merge new data into internal state
        self._data = dict(bar_data)

    def render(self) -> None:
        if _RICH_AVAILABLE:
            self._render_rich()
        else:
            self._render_plain()

    # ------------------------------------------------------------------
    # Rich renderer
    # ------------------------------------------------------------------

    def _render_rich(self) -> None:
        d = self._data
        mode = str(d.get("mode", "DEMO")).upper()
        equity = float(d.get("equity", 0.0))
        daily_pnl_pct = float(d.get("daily_pnl_pct", 0.0))
        open_positions: dict = d.get("open_positions", {})
        signals: list = d.get("signals", [])
        demo_done = int(d.get("demo_trades_completed", 0))
        demo_req = int(d.get("demo_trades_required", 500))
        daily_target = float(d.get("daily_target_pct", 0.02))

        gate_status = "HIT" if daily_pnl_pct >= daily_target else "OPEN"

        # --- Header text ---
        if mode == "MAINNET":
            mode_text = Text(f"Mode: {mode}", style="bold red")
        else:
            mode_text = Text(f"Mode: {mode}", style="cyan")

        equity_text = Text(f"Equity: ${equity:.2f}")
        pnl_style = "green" if daily_pnl_pct >= 0 else "red"
        pnl_text = Text(f"Daily P&L: {daily_pnl_pct:+.2%}", style=pnl_style)

        header_parts = [mode_text, Text("  |  "), equity_text, Text("  |  "), pnl_text]
        header = Text.assemble(*header_parts)

        # --- Open Positions table ---
        pos_table = Table(show_header=True, header_style="bold white", box=None, padding=(0, 1))
        pos_table.add_column("Symbol", style="white", min_width=10)
        pos_table.add_column("Dir", min_width=6)
        pos_table.add_column("Entry", min_width=10)
        pos_table.add_column("Size$", min_width=8)
        pos_table.add_column("P&L%", min_width=8)
        pos_table.add_column("Regime", min_width=10)

        if open_positions:
            for sym, pos in open_positions.items():
                direction = str(pos.get("direction", ""))
                dir_style = "green" if direction == "long" else "red"
                entry_p = float(pos.get("entry_price", 0))
                size_usd = float(pos.get("size_usd", 0))
                # Unrealized P&L not tracked in positions dict — show placeholder
                pnl_disp = Text("n/a", style="white")
                regime = str(pos.get("regime", ""))
                pos_table.add_row(
                    sym,
                    Text(direction.upper(), style=dir_style),
                    f"{entry_p:.4g}",
                    f"${size_usd:.2f}",
                    pnl_disp,
                    regime,
                )
        else:
            pos_table.add_row("[dim]no open positions[/dim]", "", "", "", "", "")

        # --- Signals table ---
        sig_table = Table(show_header=True, header_style="bold white", box=None, padding=(0, 1))
        sig_table.add_column("Symbol", style="white", min_width=10)
        sig_table.add_column("Prob", min_width=7)
        sig_table.add_column("Signal", min_width=7)
        sig_table.add_column("Direction", min_width=10)
        sig_table.add_column("Action", min_width=18)

        if signals:
            for s in signals:
                sym = str(s.get("symbol", ""))
                prob = float(s.get("primary_prob", 0))
                strength = float(s.get("signal_strength", 0))
                direction_int = int(s.get("direction", 0))
                action = str(s.get("action", "NO_SIGNAL"))

                dir_label = _DIR_LABELS.get(direction_int, "FLAT")
                dir_style = "green" if direction_int == 1 else ("red" if direction_int == -1 else "yellow")
                action_label = _ACTION_LABELS.get(action, action)
                action_style = "green" if action == "ENTERED" else ("dim" if "SKIP" in action else "white")

                sig_table.add_row(
                    sym,
                    f"{prob:.3f}",
                    f"{strength:.3f}",
                    Text(dir_label, style=dir_style),
                    Text(action_label, style=action_style),
                )
        else:
            sig_table.add_row("[dim]no signals this bar[/dim]", "", "", "", "")

        # --- Account summary line ---
        gate_style = "yellow" if gate_status == "HIT" else "green"
        acct_line = (
            f"Equity: ${equity:.2f}  "
            f"Demo trades: {demo_done}/{demo_req}  "
            f"Daily target: {daily_target:.2%}  "
            f"Today P&L: {daily_pnl_pct:+.2%}  "
            f"Gate: "
        )
        acct_text = Text.assemble(acct_line, Text(gate_status, style=gate_style))

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Compose into panels
        title = Text(f"CRYPTO MODEL — LIVE TRADING DASHBOARD  [{timestamp}]", style="bold white")
        pos_panel = Panel(pos_table, title="[bold]OPEN POSITIONS[/bold]", border_style="blue")
        sig_panel = Panel(sig_table, title="[bold]LAST SIGNALS (this bar)[/bold]", border_style="blue")
        acct_panel = Panel(acct_text, title="[bold]ACCOUNT[/bold]", border_style="blue")

        self._console.rule()
        self._console.print(Panel(header, title=title, border_style="bright_blue"))
        self._console.print(pos_panel)
        self._console.print(sig_panel)
        self._console.print(acct_panel)

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
        gate_status = "HIT" if daily_pnl_pct >= daily_target else "OPEN"

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        WIDTH = 62
        sep = "=" * WIDTH

        print(sep)
        print(f"  CRYPTO MODEL — LIVE TRADING DASHBOARD  [{timestamp}]")
        print(f"  Mode: {mode}  |  Equity: ${equity:.2f}  |  Daily P&L: {daily_pnl_pct:+.2%}")
        print(sep)

        print("  OPEN POSITIONS")
        if open_positions:
            print(f"  {'Symbol':<10} {'Dir':<6} {'Entry':<10} {'Size$':<8} {'Regime'}")
            for sym, pos in open_positions.items():
                direction = str(pos.get("direction", "")).upper()
                entry_p = float(pos.get("entry_price", 0))
                size_usd = float(pos.get("size_usd", 0))
                regime = str(pos.get("regime", ""))
                print(f"  {sym:<10} {direction:<6} {entry_p:<10.4g} ${size_usd:<7.2f} {regime}")
        else:
            print("  (no open positions)")

        print(sep)
        print("  LAST SIGNALS (this bar)")
        if signals:
            print(f"  {'Symbol':<10} {'Prob':<7} {'Signal':<7} {'Dir':<6} {'Action'}")
            for s in signals:
                sym = str(s.get("symbol", ""))
                prob = float(s.get("primary_prob", 0))
                strength = float(s.get("signal_strength", 0))
                direction_int = int(s.get("direction", 0))
                action = str(s.get("action", "NO_SIGNAL"))
                dir_label = _DIR_LABELS.get(direction_int, "FLAT")
                action_label = _ACTION_LABELS.get(action, action)
                print(f"  {sym:<10} {prob:<7.3f} {strength:<7.3f} {dir_label:<6} {action_label}")
        else:
            print("  (no signals this bar)")

        print(sep)
        print("  ACCOUNT")
        print(
            f"  Equity: ${equity:.2f}  Demo trades: {demo_done}/{demo_req}  "
            f"Daily target: {daily_target:.2%}  Today P&L: {daily_pnl_pct:+.2%}  Gate: {gate_status}"
        )
        print(sep)

    def print_bar_summary(self, bar_data: dict) -> None:
        # Minimal single-line log summary — used when full render is not desired
        d = bar_data
        mode = str(d.get("mode", "DEMO")).upper()
        equity = float(d.get("equity", 0.0))
        daily_pnl_pct = float(d.get("daily_pnl_pct", 0.0))
        n_open = len(d.get("open_positions", {}))
        n_signals = len(d.get("signals", []))
        logger.info(
            f"[dashboard] mode={mode} equity=${equity:.2f} daily_pnl={daily_pnl_pct:+.2%} "
            f"open={n_open} signals={n_signals}"
        )
