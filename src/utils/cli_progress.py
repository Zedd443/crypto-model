"""
Rich-based CLI progress and status display for the trading pipeline.

Usage in any stage:
    from src.utils.cli_progress import stage_header, symbol_progress, print_summary_table

Usage in run_pipeline:
    from src.utils.cli_progress import PipelineProgress
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box

console = Console(highlight=False)

# ─── Stage metadata ───────────────────────────────────────────────────────────
_STAGE_META = {
    1:    ("1  INGEST",       "Download OHLCV, on-chain, macro data from APIs"),
    2:    ("2  FEATURES",     "Compute technical indicators, regime, cross-sectional ranks"),
    3:    ("3  LABELS",       "Triple-barrier labeling + sample weights"),
    4:    ("4  TRAINING",     "XGBoost + Optuna tuning + stability feature selection"),
    "4b": ("4b HTF TRAIN",    "Train 1h/4h/1d approval-filter models"),
    5:    ("5  META",         "Meta-labeler trained on OOF predictions"),
    6:    ("6  PORTFOLIO",    "Signals + HTF approval + position sizing + correlation filter"),
    7:    ("7  BACKTEST",     "Walk-forward backtest with realistic costs"),
    8:    ("8  LIVE",         "Real-time execution loop — Binance FAPI"),
}


def stage_header(stage_key) -> None:
    label, desc = _STAGE_META.get(stage_key, (f"Stage {stage_key}", ""))
    console.rule(f"[bold cyan]STAGE {label}[/bold cyan]  [dim]{desc}[/dim]", style="cyan")


def stage_done(stage_key, elapsed_s: float, n_ok: int = 0, n_fail: int = 0) -> None:
    label, _ = _STAGE_META.get(stage_key, (f"Stage {stage_key}", ""))
    parts = [f"[bold green]✓ STAGE {label} done[/bold green]  [dim]{elapsed_s:.0f}s[/dim]"]
    if n_ok:
        parts.append(f"[green]{n_ok} ok[/green]")
    if n_fail:
        parts.append(f"[red]{n_fail} failed[/red]")
    console.print("  " + "  ".join(parts))
    console.print()


def stage_failed(stage_key, reason: str) -> None:
    label, _ = _STAGE_META.get(stage_key, (f"Stage {stage_key}", ""))
    console.print(f"  [bold red]✗ STAGE {label} FAILED:[/bold red] {reason}")
    console.print()


# ─── Symbol progress bar (replaces tqdm) ─────────────────────────────────────

def make_symbol_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


@contextmanager
def symbol_progress(symbols: list[str], stage_label: str) -> Generator[Progress, None, None]:
    with make_symbol_progress() as prog:
        task = prog.add_task(stage_label, total=len(symbols))
        prog.symbols_task = task  # type: ignore[attr-defined]
        yield prog


# ─── Summary table after a stage ─────────────────────────────────────────────

def print_summary_table(
    rows: list[dict],
    title: str,
    columns: list[str],
    col_styles: dict[str, str] | None = None,
) -> None:
    if not rows:
        return
    col_styles = col_styles or {}
    tbl = Table(title=title, box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
    for col in columns:
        tbl.add_column(col, style=col_styles.get(col, ""))
    for row in rows:
        tbl.add_row(*[str(row.get(c, "")) for c in columns])
    console.print(tbl)


# ─── Stage 8 live bar status panel ───────────────────────────────────────────

class LiveBarPanel:
    """
    Prints a Rich table after each 15-minute bar in stage_08.

    Usage:
        panel = LiveBarPanel()
        panel.print_bar_result(bar_signals, equity, bar_time)
    """

    def print_bar_result(
        self,
        bar_signals: list[dict],
        equity: float,
        bar_time: datetime,
        open_positions: dict,
    ) -> None:
        entered  = [s for s in bar_signals if s.get("action") == "ENTERED"]
        held     = [s for s in bar_signals if s.get("action") == "HOLD"]
        cands    = [s for s in bar_signals if s.get("action") in ("CANDIDATE", "ENTERED", "SKIP_LIMIT")]
        n_floor  = sum(1 for s in bar_signals if s.get("action") == "SKIP_FLOOR")
        n_dead   = sum(1 for s in bar_signals if s.get("action") == "SKIP_DEAD_ZONE")
        n_daily  = sum(1 for s in bar_signals if s.get("action") == "SKIP_DAILY")
        failed   = [s["symbol"] for s in bar_signals if s.get("action") == "FAILED"]

        ts = bar_time.strftime("%Y-%m-%d %H:%M UTC")

        # ── Header line ──────────────────────────────────────────────────────
        header = (
            f"[bold cyan]BAR {ts}[/bold cyan]  "
            f"equity=[yellow]${equity:.2f}[/yellow]  "
            f"open=[white]{len(open_positions)}[/white]  "
            f"entered=[green]{len(entered)}[/green]  "
            f"floor={n_floor}  dead={n_dead}  daily_cap={n_daily}"
        )
        if failed:
            header += f"  [red]FAILED={failed}[/red]"

        console.print(header)

        # ── Candidates table ─────────────────────────────────────────────────
        if cands:
            tbl = Table(box=box.MINIMAL, show_header=True, header_style="bold dim", padding=(0, 1))
            tbl.add_column("Symbol",       style="cyan",  width=16)
            tbl.add_column("Dir",          justify="center", width=4)
            tbl.add_column("P(dir)",       justify="right", width=7)
            tbl.add_column("P(meta)",      justify="right", width=7)
            tbl.add_column("Signal",       justify="right", width=7)
            tbl.add_column("TP reach",     justify="right", width=8)
            tbl.add_column("TP%",          justify="right", width=6)
            tbl.add_column("SL%",          justify="right", width=6)
            tbl.add_column("Regime",       width=12)
            tbl.add_column("Action",       width=12)

            sorted_cands = sorted(cands, key=lambda x: x.get("composite_score", 0), reverse=True)
            for c in sorted_cands[:10]:
                act = c.get("action", "?")
                dir_str = c.get("direction_str", "?")
                dir_display = Text("▲ LONG", style="green") if dir_str == "long" else Text("▼ SHORT", style="red")
                act_style = {
                    "ENTERED":    "bold green",
                    "CANDIDATE":  "yellow",
                    "SKIP_LIMIT": "dim",
                }.get(act, "dim")
                tbl.add_row(
                    c["symbol"],
                    dir_display,
                    f"{c.get('primary_prob', 0):.3f}",
                    f"{c.get('meta_prob', 0):.3f}",
                    f"{c.get('signal_strength', 0):.3f}",
                    f"{c.get('tp_reach_score', 0):.3f}",
                    f"{c.get('tp_pct', 0):.2%}",
                    f"{c.get('sl_pct', 0):.2%}",
                    c.get("regime", ""),
                    Text(act, style=act_style),
                )
            console.print(tbl)

        # ── Open positions ───────────────────────────────────────────────────
        if open_positions:
            ptbl = Table(box=box.MINIMAL, show_header=True, header_style="bold dim", padding=(0, 1))
            ptbl.add_column("Position",  style="cyan", width=16)
            ptbl.add_column("Dir",       justify="center", width=6)
            ptbl.add_column("Entry $",   justify="right", width=10)
            ptbl.add_column("Size USDT", justify="right", width=10)
            ptbl.add_column("TP%",       justify="right", width=6)
            ptbl.add_column("SL%",       justify="right", width=6)

            for sym, pos in open_positions.items():
                d = pos.get("direction", "long")
                dir_txt = Text("▲", style="green") if d == "long" else Text("▼", style="red")
                ptbl.add_row(
                    sym,
                    dir_txt,
                    f"{pos.get('entry_price', 0):.4f}",
                    f"{pos.get('size_usd', 0):.2f}",
                    f"{pos.get('tp_pct', 0):.2%}",
                    f"{pos.get('sl_pct', 0):.2%}",
                )
            console.print(ptbl)

        console.rule(style="dim")


# ─── Pipeline-level progress (wraps run_pipeline.py) ─────────────────────────

class PipelineProgress:
    """
    Context manager used in run_pipeline.py to wrap each stage with timing and
    Rich output. Replaces the plain logger.info start/end messages.

    with PipelineProgress(stage_key) as p:
        stage_fn(cfg, ...)
    """

    def __init__(self, stage_key):
        self._key = stage_key
        self._start = 0.0

    def __enter__(self):
        stage_header(self._key)
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        if exc_type is None:
            stage_done(self._key, elapsed)
        else:
            stage_failed(self._key, str(exc_val))
        return False  # re-raise exceptions
