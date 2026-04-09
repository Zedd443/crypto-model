import csv
import math
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.execution.binance_client import BinanceClient
from src.utils.logger import get_logger
from src.utils.state_manager import increment_demo_trades

logger = get_logger("order_manager")


def _round_qty(qty: float, step: float) -> float:
    # Floor to step size and round to correct decimal precision — same logic as crypto_ml testnet_trader.py
    if step <= 0:
        return round(qty, 3)
    prec = max(0, -int(math.floor(math.log10(step))))
    return round(math.floor(qty / step) * step, prec)

# CSV columns written for every completed trade
_TRADE_LOG_COLS = [
    "timestamp_entry",
    "timestamp_exit",
    "symbol",
    "direction",
    "entry_price",
    "exit_price",
    "size_usd",
    "pnl_pct",
    "regime",
    "signal_strength",
]


class OrderManager:
    def __init__(self, client: BinanceClient, cfg, trade_log_path: Path, mode: str = "DEMO"):
        self._client = client
        self._cfg = cfg
        self._log_path = Path(trade_log_path)
        self._dms_seconds = float(cfg.trading.dead_man_switch_seconds)
        self._mode = mode.upper()

        # {symbol: {order_id, direction, entry_price, size_usd,
        #           tp_price, sl_price, entry_time,
        #           tp_order_id, sl_order_id, regime, signal_strength}}
        self.positions: dict = {}

        self._heartbeat_lock = threading.Lock()
        self._last_heartbeat = time.monotonic()

        # Ensure CSV file has a header row if it doesn't exist yet
        if not self._log_path.exists():
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_TRADE_LOG_COLS).writeheader()

        # Start dead-man-switch background thread
        self._dms_thread = threading.Thread(target=self._dead_man_switch_loop, daemon=True)
        self._dms_thread.start()
        logger.info(f"OrderManager started — DMS={self._dms_seconds}s trade_log={self._log_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def heartbeat(self) -> None:
        with self._heartbeat_lock:
            self._last_heartbeat = time.monotonic()

    def submit_entry(
        self,
        symbol: str,
        direction: str,
        size_usd: float,
        entry_price: float,
        tp_pct: float,
        sl_pct: float,
        regime: str = "",
        signal_strength: float = 0.0,
    ) -> str | None:
        if symbol in self.positions:
            logger.warning(f"{symbol}: submit_entry called but position already tracked — skipping")
            return None

        # MAINNET: cap size_usd to the notional limit for the active leverage tier.
        # This reflects the real Binance position size limit (e.g. $250k at 2-3x).
        # DEMO: leverageBracket not reliable on testnet — skip this cap.
        leverage = getattr(self._cfg.trading, "leverage", getattr(getattr(self._cfg, "growth_gate", self._cfg), "fixed_leverage", 2))
        if self._mode != "DEMO":
            notional_cap = self._client.get_notional_limit(symbol, int(leverage))
            if size_usd > notional_cap:
                logger.info(f"{symbol}: size_usd {size_usd:.2f} > notional cap {notional_cap:.0f} at {leverage}x — clamped")
                size_usd = notional_cap

        # Convert USD size to contract quantity using current price, rounded to symbol's step size
        qty_raw = size_usd / entry_price
        qty_step = self._client.get_qty_step(symbol)
        qty = _round_qty(qty_raw, qty_step)

        # Cap qty to exchange max lot size.
        # DEMO (demo-fapi): MARKET_LOT_SIZE.maxQty=120 is a real market order limit.
        # MAINNET: LOT_SIZE.maxQty applies (notional_cap above is the primary constraint).
        min_notional = self._client.get_min_notional(symbol)
        max_qty = self._client.get_max_qty(symbol, entry_price)
        if qty > max_qty:
            qty = _round_qty(max_qty, qty_step)
            size_usd = qty * entry_price
            logger.info(f"{symbol}: volume dipotong ke max exchange {size_usd:.2f} USDT")

        # Structural check: if even the max possible order is below min notional, skip
        max_volume_exchange = max_qty * entry_price
        if max_volume_exchange < min_notional:
            logger.warning(f"{symbol}: max volume {max_volume_exchange:.2f} < min order {min_notional:.2f} — skip")
            return None

        # Pastikan volume >= minimum order exchange (berlaku di semua mode)
        effective_notional = qty * entry_price
        if effective_notional < min_notional:
            qty_bumped = _round_qty((min_notional * 1.1) / entry_price, qty_step)
            qty_bumped = min(qty_bumped, max_qty)
            effective_notional = qty_bumped * entry_price
            if effective_notional < min_notional:
                logger.warning(f"{symbol}: bumped notional {effective_notional:.2f} still < min {min_notional:.2f} — skip")
                return None
            logger.info(f"{symbol}: volume dinaikkan ke min order {effective_notional:.2f} USDT")
            qty = qty_bumped

        # Market entry
        entry_side = "BUY" if direction == "long" else "SELL"
        try:
            entry_resp = self._client.place_order(symbol, entry_side, qty, order_type="MARKET")
        except Exception as exc:
            logger.error(f"{symbol}: entry order failed: {exc}")
            return None

        order_id = str(entry_resp.get("orderId", ""))
        # Use actual fill price from avgPrice if available (more accurate than kline close)
        filled_price = float(entry_resp.get("avgPrice") or 0)
        if filled_price > 0:
            entry_price = filled_price

        # Wait for position to settle before placing bracket orders.
        # Without this delay, reduceOnly bracket orders can arrive before the exchange
        # registers the open position and get rejected (-2022 / no position to reduce).
        time.sleep(1)

        # Compute TP / SL prices.
        # DEMO: demo-fapi enforces PERCENT_PRICE ±5% from live mark price (not entry_price).
        # Cap TP/SL to ±4% of entry to stay safely inside that band even if mark drifts.
        # MAINNET: no special cap needed (real stopPrice orders don't have this restriction).
        tick = self._client.get_tick_size(symbol)
        _max_bracket_pct = 0.04 if self._mode == "DEMO" else 1.0
        tp_pct_eff = min(tp_pct, _max_bracket_pct)
        sl_pct_eff = min(sl_pct, _max_bracket_pct)

        import math as _math
        if direction == "long":
            raw_tp = entry_price * (1.0 + tp_pct_eff)
            raw_sl = entry_price * (1.0 - sl_pct_eff)
            tp_price = self._client.round_price(symbol, raw_tp)
            if tp_price <= entry_price:
                tp_price = round(entry_price + tick, 10)
            # For DEMO: floor SL so it never exceeds the ±4% band after rounding
            sl_price = round(_math.floor(raw_sl / tick) * tick, 10) if self._mode == "DEMO" else self._client.round_price(symbol, raw_sl)
            if sl_price >= entry_price:
                sl_price = round(entry_price - tick, 10)
            close_side = "SELL"
        else:
            raw_tp = entry_price * (1.0 - tp_pct_eff)
            raw_sl = entry_price * (1.0 + sl_pct_eff)
            tp_price = self._client.round_price(symbol, raw_tp)
            if tp_price >= entry_price:
                tp_price = round(entry_price - tick, 10)
            # For DEMO: ceil SL so it never exceeds the ±4% band after rounding (short SL is above entry)
            sl_price = round(_math.ceil(raw_sl / tick) * tick, 10) if self._mode == "DEMO" else self._client.round_price(symbol, raw_sl)
            # After ceiling, still check it's within 4%
            if self._mode == "DEMO" and sl_price > entry_price * (1.0 + _max_bracket_pct):
                sl_price = round(_math.floor(entry_price * (1.0 + _max_bracket_pct) / tick) * tick, 10)
            if sl_price <= entry_price:
                sl_price = round(entry_price + tick, 10)
            close_side = "BUY"

        tp_algo_id = None
        sl_algo_id = None

        if self._mode == "DEMO":
            # DEMO: LIMIT orders fill immediately at market price — cannot emulate stop/TP orders.
            # Instead, store tp_price/sl_price in the position dict and let sync_fills check
            # mark price each bar to simulate TP/SL. No bracket orders sent to exchange.
            logger.info(f"{symbol}: DEMO mode — bracket orders skipped, TP/SL monitored via sync_fills (TP={tp_price} SL={sl_price})")
        else:
            # MAINNET: real STOP_MARKET / TAKE_PROFIT_MARKET with closePosition=true
            tp_failed = False
            sl_failed = False

            try:
                tp_resp = self._client.place_order(
                    symbol, close_side, qty, order_type="TAKE_PROFIT_MARKET",
                    stop_price=tp_price, close_position=True,
                )
                tp_algo_id = tp_resp.get("orderId")
            except Exception as exc:
                logger.warning(f"{symbol}: TP order failed: {exc}")
                tp_failed = True

            try:
                sl_resp = self._client.place_order(
                    symbol, close_side, qty, order_type="STOP_MARKET",
                    stop_price=sl_price, close_position=True,
                    working_type="MARK_PRICE",
                )
                sl_algo_id = sl_resp.get("orderId")
            except Exception as exc:
                logger.warning(f"{symbol}: SL order failed: {exc}")
                sl_failed = True

            # Retry any failed bracket once
            if tp_failed or sl_failed:
                time.sleep(2)
                if tp_failed:
                    try:
                        tp_resp = self._client.place_order(
                            symbol, close_side, qty, order_type="TAKE_PROFIT_MARKET",
                            stop_price=tp_price, close_position=True,
                        )
                        tp_algo_id = tp_resp.get("orderId")
                        tp_failed = False
                    except Exception as exc:
                        logger.warning(f"{symbol}: TP retry failed: {exc}")
                if sl_failed:
                    try:
                        sl_resp = self._client.place_order(
                            symbol, close_side, qty, order_type="STOP_MARKET",
                            stop_price=sl_price, close_position=True,
                            working_type="MARK_PRICE",
                        )
                        sl_algo_id = sl_resp.get("orderId")
                        sl_failed = False
                    except Exception as exc:
                        logger.warning(f"{symbol}: SL retry failed: {exc}")

            if tp_failed or sl_failed:
                logger.critical(f"{symbol}: {'TP' if tp_failed else 'SL'} gagal setelah retry — posisi tanpa bracket (DMS akan close)")

        self.positions[symbol] = {
            "order_id": order_id,
            "direction": direction,
            "entry_price": entry_price,
            "size_usd": size_usd,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "tp_order_id": tp_algo_id,
            "sl_order_id": sl_algo_id,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "regime": regime,
            "signal_strength": signal_strength,
        }

        logger.info(
            f"{symbol}: MASUK {direction.upper()} — "
            f"volume={qty * entry_price:.2f} USDT @ {entry_price} "
            f"TP={tp_price} SL={sl_price}"
        )
        return order_id

    def submit_exit(self, symbol: str) -> bool:
        if symbol not in self.positions:
            logger.warning(f"{symbol}: submit_exit called but no tracked position")
            return False

        pos = self.positions[symbol]

        # Cancel bracket orders first to avoid double-fill race
        # TP/SL are regular orders on /fapi/v1/order — cancel via orderId
        for oid_key in ("tp_order_id", "sl_order_id"):
            oid = pos.get(oid_key)
            if oid is not None:
                try:
                    self._client.cancel_order(symbol, oid, is_algo=False)
                except Exception as exc:
                    logger.debug(f"{symbol}: cancel {oid_key} failed: {exc}")

        # Flatten with a market close
        close_side = "SELL" if pos["direction"] == "long" else "BUY"
        qty_raw = pos["size_usd"] / pos["entry_price"]
        qty_step = self._client.get_qty_step(symbol)
        qty = _round_qty(qty_raw, qty_step)
        try:
            self._client.place_order(symbol, close_side, qty, order_type="MARKET", reduce_only=True)
            logger.info(f"{symbol}: market exit placed")
        except Exception as exc:
            logger.error(f"{symbol}: market exit failed: {exc}")
            return False

        del self.positions[symbol]
        return True

    def sync_fills(self, symbol: str) -> dict | None:
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        try:
            live_pos = self._client.get_position(symbol)
        except Exception as exc:
            logger.warning(f"{symbol}: sync_fills get_position failed: {exc}")
            return None

        # Position still open — update unrealized PnL and actual entry price from exchange
        if abs(live_pos["positionAmt"]) > 1e-9:
            pos["unrealized_pnl"] = live_pos["unrealizedProfit"]
            # Sync entry price from exchange if it differs from our stored value (fills at market)
            exchange_entry = live_pos["entryPrice"]
            if exchange_entry > 0:
                pos["entry_price"] = exchange_entry

            # DEMO: exchange doesn't handle bracket orders — simulate TP/SL using mark price
            if self._mode == "DEMO":
                mark_price = float(live_pos.get("markPrice", 0) or live_pos.get("entryPrice", 0))
                if mark_price > 0:
                    tp_price = pos.get("tp_price", 0)
                    sl_price = pos.get("sl_price", 0)
                    direction = pos["direction"]
                    close_side = "SELL" if direction == "long" else "BUY"
                    qty_step = self._client.get_qty_step(symbol)
                    qty = _round_qty(pos["size_usd"] / pos["entry_price"], qty_step)

                    hit = None
                    if direction == "long":
                        if tp_price and mark_price >= tp_price:
                            hit = "TP"
                        elif sl_price and mark_price <= sl_price:
                            hit = "SL"
                    else:  # short
                        if tp_price and mark_price <= tp_price:
                            hit = "TP"
                        elif sl_price and mark_price >= sl_price:
                            hit = "SL"

                    if hit:
                        logger.info(f"{symbol}: DEMO {hit} hit — mark={mark_price} {'≥' if hit=='TP' and direction=='long' else '≤'} {tp_price if hit=='TP' else sl_price} — closing")
                        try:
                            self._client.place_order(symbol, close_side, qty, order_type="MARKET", reduce_only=True)
                        except Exception as exc:
                            logger.error(f"{symbol}: DEMO {hit} market close failed: {exc}")
                        # Record the fill using mark price as exit price
                        entry_price = pos["entry_price"]
                        exit_price = mark_price
                        if direction == "long":
                            pnl_pct = (exit_price - entry_price) / (entry_price + 1e-12)
                        else:
                            pnl_pct = (entry_price - exit_price) / (entry_price + 1e-12)
                        fill = {
                            "timestamp_entry": pos["entry_time"],
                            "timestamp_exit": datetime.now(timezone.utc).isoformat(),
                            "symbol": symbol,
                            "direction": direction,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "size_usd": pos["size_usd"],
                            "pnl_pct": round(pnl_pct, 6),
                            "regime": pos.get("regime", ""),
                            "signal_strength": pos.get("signal_strength", 0.0),
                        }
                        self._write_trade_log(fill)
                        del self.positions[symbol]
                        logger.info(f"{symbol}: DEMO {hit} fill recorded — pnl_pct={pnl_pct:.4%}")
                        return fill

            return None  # still open

        # Fill detected (exchange closed the position — MAINNET bracket hit, or manual close)
        # Query recent trades to get actual exit fill price.
        # Binance zeroes entryPrice when position is closed, so it cannot be used as exit proxy.
        try:
            trades = self._client.get_recent_trades(symbol, limit=10)
            exit_price = float(trades[-1]["price"]) if trades else float(live_pos.get("markPrice", 0))
        except Exception:
            exit_price = float(live_pos.get("markPrice", 0))
        entry_price = pos["entry_price"]

        if pos["direction"] == "long":
            pnl_pct = (exit_price - entry_price) / (entry_price + 1e-12)
        else:
            pnl_pct = (entry_price - exit_price) / (entry_price + 1e-12)

        fill = {
            "timestamp_entry": pos["entry_time"],
            "timestamp_exit": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": pos["direction"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size_usd": pos["size_usd"],
            "pnl_pct": round(pnl_pct, 6),
            "regime": pos.get("regime", ""),
            "signal_strength": pos.get("signal_strength", 0.0),
        }

        self._write_trade_log(fill)
        del self.positions[symbol]
        logger.info(f"{symbol}: fill recorded — pnl_pct={pnl_pct:.4%}")
        return fill

    def cancel_all_open(self, force_close: bool = False) -> None:
        # DEMO mode: do NOT auto-close positions on shutdown/DMS.
        # Positions are managed via sync_fills (TP/SL simulation). Closing early on Ctrl+C
        # causes unnecessary losses — user can manually close in the Binance app if needed.
        if self._mode == "DEMO" and not force_close:
            logger.warning(
                f"cancel_all_open called in DEMO mode — skipping market close to avoid fee losses. "
                f"Open positions ({list(self.positions.keys())}) remain on exchange. "
                f"Close manually in Binance app if needed."
            )
            self.positions.clear()
            return

        logger.warning("cancel_all_open: flattening all tracked positions (dead-man-switch or shutdown)")
        for symbol, pos in list(self.positions.items()):
            if pos is None:
                continue
            # Cancel bracket orders first (MAINNET only — DEMO has no exchange bracket orders)
            if self._mode != "DEMO":
                try:
                    self._client.cancel_all_orders(symbol)
                    logger.info(f"{symbol}: bracket orders cancelled")
                except Exception as exc:
                    logger.warning(f"{symbol}: cancel_all_orders failed (continuing to market close): {exc}")

            # Get actual position qty from exchange — do NOT recalculate from size_usd/entry_price
            # to avoid over/under-close due to rounding or maxQty caps at entry.
            actual_qty = None
            close_side = "SELL" if pos["direction"] == "long" else "BUY"
            try:
                live_pos = self._client.get_position(symbol)
                pos_amt = live_pos["positionAmt"]
                if abs(pos_amt) < 1e-9:
                    logger.info(f"{symbol}: no open position on exchange — skip market close")
                    continue
                actual_qty = abs(pos_amt)
            except Exception:
                pass  # if check fails, fall back to calculated qty

            if actual_qty is None:
                qty_step = self._client.get_qty_step(symbol)
                actual_qty = _round_qty(pos["size_usd"] / pos["entry_price"], qty_step)

            try:
                self._client.place_order(symbol, close_side, actual_qty, order_type="MARKET", reduce_only=True)
                logger.info(f"{symbol}: market close issued by DMS/shutdown qty={actual_qty}")
            except Exception as exc:
                logger.critical(f"{symbol}: DMS market close FAILED — position may remain open on exchange: {exc}")
        self.positions.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_trade_log(self, row: dict) -> None:
        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADE_LOG_COLS)
            writer.writerow({col: row.get(col, "") for col in _TRADE_LOG_COLS})

        # Increment demo trade counter when a fill is recorded in DEMO mode
        if self._mode == "DEMO":
            try:
                count = increment_demo_trades()
                demo_required = int(self._cfg.growth_gate.demo_trades_required)
                logger.info(f"Demo trades: {count}/{demo_required}")
            except Exception as exc:
                logger.warning(f"Could not increment demo_trades_completed: {exc}")

    def _dead_man_switch_loop(self) -> None:
        # Background daemon: cancel everything if heartbeat goes silent
        while True:
            time.sleep(5)
            with self._heartbeat_lock:
                elapsed = time.monotonic() - self._last_heartbeat
            if elapsed > self._dms_seconds:
                logger.critical(
                    f"Dead-man-switch triggered — no heartbeat for {elapsed:.0f}s "
                    f"(limit={self._dms_seconds}s). Cancelling all positions."
                )
                try:
                    # DEMO: force_close=True so DMS actually closes (process is dead, no one monitoring)
                    self.cancel_all_open(force_close=(self._mode == "DEMO"))
                except Exception as exc:
                    logger.error(f"DMS cancel_all_open failed: {exc}")
                # Reset heartbeat so we don't loop-cancel
                with self._heartbeat_lock:
                    self._last_heartbeat = time.monotonic()
