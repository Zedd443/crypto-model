import pandas as pd
import numpy as np
from src.backtest.survivorship import check_delistings
from src.backtest.costs import compute_total_trade_cost
from src.utils.logger import get_logger

logger = get_logger("engine")


class BacktestEngine:
    def __init__(self, signals_df: pd.DataFrame, prices_dict: dict, costs_cfg, delisted_dict: dict, cfg):
        # signals_df: pre-computed signals DataFrame indexed by timestamp
        # prices_dict: {symbol: DataFrame with OHLCV}
        # CRITICAL: ALL signals must be pre-computed before this engine runs
        self.signals = signals_df
        self.prices = prices_dict
        self.cfg = cfg
        self.delisted = delisted_dict
        self.positions = {}  # {symbol: {entry_price, size_usd, direction, entry_time, tp, sl}}
        self.nav = []
        self.trade_log = []
        self.equity = float(getattr(getattr(cfg, "account", None), "current_equity", None) or 120.0)
        self.equity_start_of_day = self.equity
        self.ath = self.equity
        self.trading_halted_today = False
        self.circuit_breaker_active = False

    def run(self) -> dict:
        # Collect all timestamps from all price series
        timestamps = sorted(set(
            t for df in self.prices.values() for t in df.index
        ))

        for current_bar_idx, ts in enumerate(timestamps):
            self._process_bar(ts, current_bar_idx)

        nav_series = pd.Series(
            [n["nav"] for n in self.nav],
            index=[n["timestamp"] for n in self.nav],
            name="nav",
        )

        trade_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()

        return {
            "nav_series": nav_series,
            "trade_log": trade_df,
            "final_equity": self.equity,
        }

    def _process_bar(self, timestamp: pd.Timestamp, current_bar_idx: int = 0) -> None:
        # Reset daily state at day boundary
        if len(self.nav) > 0:
            last_ts = self.nav[-1]["timestamp"]
            if last_ts.date() < timestamp.date():
                self.equity_start_of_day = self.equity
                self.trading_halted_today = False
                # Reset circuit breaker after ath_pause_hours (approximated as 1 day)
                if self.circuit_breaker_active:
                    self.circuit_breaker_active = False

        current_prices = {
            sym: df.loc[timestamp, "close"]
            for sym, df in self.prices.items()
            if timestamp in df.index and not pd.isna(df.loc[timestamp, "close"])
        }

        # 1. Check delistings
        forced_closes = check_delistings(self.positions, timestamp, self.delisted, self.cfg)
        for fc in forced_closes:
            self._close_position(fc["symbol"], fc["price"], timestamp, "delisted")

        # 2. Process exits (TP, SL, trailing stop, h4_exit flag)
        exits = self._check_exits(current_prices, timestamp, current_bar_idx)
        for ex in exits:
            self._close_position(ex["symbol"], ex["price"], timestamp, ex["reason"])

        # 3. Check circuit breakers
        unrealized_pnl = self._compute_unrealized_pnl(current_prices)
        daily_dd = (
            (self.equity + unrealized_pnl - self.equity_start_of_day)
            / (self.equity_start_of_day + 1e-9)
        )

        daily_halt_threshold = -float(self.cfg.backtest.daily_halt_dd_pct)
        if daily_dd < daily_halt_threshold:
            self._close_all_positions(current_prices, timestamp, "daily_circuit_breaker")
            self.trading_halted_today = True
            logger.warning(f"Daily circuit breaker triggered at {timestamp}: dd={daily_dd:.3f}")

        ath_pause_threshold = -float(self.cfg.backtest.ath_pause_dd_pct)
        if self.ath > 0 and (self.equity - self.ath) / (self.ath + 1e-9) < ath_pause_threshold:
            self.circuit_breaker_active = True
            logger.warning(f"ATH circuit breaker triggered at {timestamp}")

        # 4. Process new entries (if not halted)
        if not self.trading_halted_today and not self.circuit_breaker_active:
            if timestamp in self.signals.index:
                bar_signals = self.signals.loc[timestamp]
                self._process_entries(bar_signals, current_prices, timestamp, current_bar_idx)

        # 5. Update NAV
        unrealized_pnl = self._compute_unrealized_pnl(current_prices)
        self.nav.append({
            "timestamp": timestamp,
            "nav": self.equity + unrealized_pnl,
            "n_positions": len(self.positions),
        })

        # Update ATH on realized equity
        if self.equity > self.ath:
            self.ath = self.equity

    def _check_exits(self, current_prices: dict, timestamp: pd.Timestamp, current_bar_idx: int = 0) -> list:
        exits = []
        max_hold = getattr(getattr(self.cfg, "labels", None), "max_hold_bars", 16)

        for symbol, pos in list(self.positions.items()):
            if symbol not in current_prices:
                continue
            price = current_prices[symbol]
            direction = pos["direction"]

            # Time-barrier: force close when position has been held >= max_hold_bars
            bars_held = current_bar_idx - pos.get("entry_bar_idx", current_bar_idx)
            if bars_held >= max_hold:
                exits.append({"symbol": symbol, "price": price, "reason": "time_barrier"})
                continue

            if direction == 1:  # long
                if price >= pos["tp"]:
                    exits.append({"symbol": symbol, "price": pos["tp"], "reason": "TP"})
                elif price <= pos["sl"]:
                    exits.append({"symbol": symbol, "price": pos["sl"], "reason": "SL"})
                elif pos.get("trailing_sl") is not None and price <= pos["trailing_sl"]:
                    exits.append({"symbol": symbol, "price": pos["trailing_sl"], "reason": "trailing"})
            else:  # short
                if price <= pos["tp"]:
                    exits.append({"symbol": symbol, "price": pos["tp"], "reason": "TP"})
                elif price >= pos["sl"]:
                    exits.append({"symbol": symbol, "price": pos["sl"], "reason": "SL"})
                elif pos.get("trailing_sl") is not None and price >= pos["trailing_sl"]:
                    exits.append({"symbol": symbol, "price": pos["trailing_sl"], "reason": "trailing"})

            # Update trailing stop for remaining positions (not being exited this bar)
            if symbol not in [ex["symbol"] for ex in exits]:
                self._update_trailing_stop(symbol, price)

        return exits

    def _update_trailing_stop(self, symbol: str, current_price: float) -> None:
        pos = self.positions.get(symbol)
        if not pos:
            return
        entry = pos["entry_price"]
        direction = pos["direction"]
        atr = pos.get("atr", entry * 0.01)
        trail_trigger = float(self.cfg.backtest.trailing_trigger_pct)
        trail_mult = float(self.cfg.backtest.trailing_atr_mult)

        unrealized_pct = (current_price - entry) / (entry + 1e-9) * direction
        if unrealized_pct > trail_trigger:
            if direction == 1:
                new_trail = current_price - trail_mult * atr
                existing = pos.get("trailing_sl")
                pos["trailing_sl"] = max(existing, new_trail) if existing is not None else new_trail
            else:
                new_trail = current_price + trail_mult * atr
                existing = pos.get("trailing_sl")
                pos["trailing_sl"] = min(existing, new_trail) if existing is not None else new_trail

    def _process_entries(self, bar_signals, current_prices: dict, timestamp: pd.Timestamp, current_bar_idx: int = 0) -> None:
        # bar_signals: row from signals_df (could be Series indexed by symbol or dict-like)
        # Signals must be pre-computed — just iterate
        if isinstance(bar_signals, pd.Series):
            # Single-symbol: bar_signals is a Series with signal fields
            if "is_signal" in bar_signals.index:
                # Treat index as symbol names if multi-symbol, else single symbol
                self._try_enter_from_signal_row(bar_signals, current_prices, timestamp, current_bar_idx)
            return

        # If bar_signals is a DataFrame row (per-symbol), iterate columns
        if hasattr(bar_signals, "items"):
            for symbol, sig in bar_signals.items():
                if isinstance(sig, dict):
                    if not sig.get("is_signal"):
                        continue
                    self._enter_position(symbol, sig, current_prices, timestamp, current_bar_idx)

    def _try_enter_from_signal_row(self, sig_row: pd.Series, current_prices: dict, timestamp: pd.Timestamp, current_bar_idx: int = 0) -> None:
        # Single-symbol signal row
        if not sig_row.get("is_signal", 0):
            return
        # Determine symbol from prices dict (single-symbol mode)
        for symbol in current_prices:
            if symbol not in self.positions:
                sig_dict = sig_row.to_dict()
                sig_dict["symbol"] = symbol
                self._enter_position(symbol, sig_dict, current_prices, timestamp, current_bar_idx)
                break

    def _enter_position(self, symbol: str, sig: dict, current_prices: dict, timestamp: pd.Timestamp, current_bar_idx: int = 0) -> None:
        if symbol in self.positions:
            return
        if symbol not in current_prices:
            return

        price = current_prices[symbol]
        direction = int(sig.get("direction", 0))
        if direction == 0:
            return

        size_usd = float(sig.get("position_size_usd", self.equity * 0.05))
        atr = float(sig.get("atr", price * 0.01))

        tp = price + direction * atr * float(self.cfg.backtest.tp_atr_mult)
        sl = price - direction * atr * float(self.cfg.backtest.sl_atr_mult)

        # Adjust for direction: short positions need inverted logic
        if direction == -1:
            tp = price - atr * float(self.cfg.backtest.tp_atr_mult)
            sl = price + atr * float(self.cfg.backtest.sl_atr_mult)

        self.positions[symbol] = {
            "entry_price": price,
            "size_usd": size_usd,
            "direction": direction,
            "entry_time": timestamp,
            "entry_bar_idx": current_bar_idx,
            "tp": tp,
            "sl": sl,
            "atr": atr,
            "trailing_sl": None,
        }
        logger.debug(f"Entry: {symbol} dir={direction} price={price:.4f} tp={tp:.4f} sl={sl:.4f}")

    def _close_position(self, symbol: str, price: float, timestamp: pd.Timestamp, reason: str) -> None:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        size_usd = pos["size_usd"]

        pnl_pct = (price - entry_price) / (entry_price + 1e-9) * direction
        pnl_usd = pnl_pct * size_usd

        hold_bars = 0
        if pos["entry_time"] is not None:
            # Approximate hold duration
            try:
                dt = (timestamp - pos["entry_time"]).total_seconds() / 900  # 15m bars
                hold_bars = int(dt)
            except Exception:
                hold_bars = 0

        # Deduct trade costs: slippage (entry + exit) + commissions + funding
        # adv_usd=0 so sqrt market impact term is zero — market impact already baked into
        # slippage_pct in config.  funding_rate=0 avoids needing per-symbol funding data.
        hold_hours = hold_bars * 0.25  # each bar is 15 minutes = 0.25 hours
        cost = compute_total_trade_cost(
            entry_price=entry_price,
            exit_price=price,
            size_usd=size_usd,
            direction=direction,
            adv_usd=0.0,
            funding_rate=0.0,
            hold_hours=hold_hours,
            cfg=self.cfg,
        )
        pnl_usd -= cost["total_cost_usd"]
        logger.debug(f"Exit: {symbol} reason={reason} pnl_usd={pnl_usd:.2f} cost_usd={cost['total_cost_usd']:.4f}")

        self.equity += pnl_usd

        self.trade_log.append({
            "symbol": symbol,
            "entry_time": pos["entry_time"],
            "exit_time": timestamp,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": price,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "cost_usd": cost["total_cost_usd"],
            "exit_reason": reason,
            "size_usd": size_usd,
            "hold_bars": hold_bars,
        })

    def _close_all_positions(self, current_prices: dict, timestamp: pd.Timestamp, reason: str) -> None:
        for symbol in list(self.positions.keys()):
            price = current_prices.get(symbol, self.positions[symbol]["entry_price"])
            self._close_position(symbol, price, timestamp, reason)

    def _compute_unrealized_pnl(self, current_prices: dict) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                pnl_pct = (
                    (current_prices[symbol] - pos["entry_price"])
                    / (pos["entry_price"] + 1e-9)
                    * pos["direction"]
                )
                total += pnl_pct * pos["size_usd"]
        return total
