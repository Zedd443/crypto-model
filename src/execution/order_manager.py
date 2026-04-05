import csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from src.execution.binance_client import BinanceClient
from src.utils.logger import get_logger
from src.utils.state_manager import increment_demo_trades

logger = get_logger("order_manager")

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

        # Convert USD size to contract quantity using current price, rounded to symbol's step size
        qty_raw = size_usd / entry_price
        qty_step = self._client.get_qty_step(symbol)
        qty = round(qty_raw / qty_step) * qty_step

        # Cap to exchange max order quantity (some small-cap coins have low max qty)
        max_qty = self._client.get_max_qty(symbol, entry_price)
        if qty > max_qty:
            logger.warning(
                f"{symbol}: qty {qty} exceeds exchange max_qty {max_qty} — capping. "
                f"Effective notional={max_qty * entry_price:.2f} USD (wallet too large for this coin)"
            )
            qty = round(max_qty / qty_step) * qty_step

        # Market entry
        entry_side = "BUY" if direction == "long" else "SELL"
        try:
            entry_resp = self._client.place_order(symbol, entry_side, qty, order_type="MARKET")
        except Exception as exc:
            logger.error(f"{symbol}: entry order failed: {exc}")
            return None

        order_id = str(entry_resp.get("orderId", ""))

        # Compute TP / SL prices
        if direction == "long":
            tp_price = round(entry_price * (1.0 + tp_pct), 8)
            sl_price = round(entry_price * (1.0 - sl_pct), 8)
            close_side = "SELL"
        else:
            tp_price = round(entry_price * (1.0 - tp_pct), 8)
            sl_price = round(entry_price * (1.0 + sl_pct), 8)
            close_side = "BUY"

        tp_order_id = None
        sl_order_id = None
        tp_failed = False
        sl_failed = False

        # Limit TP order
        try:
            tp_resp = self._client.place_order(
                symbol, close_side, qty, order_type="LIMIT",
                price=tp_price, reduce_only=True,
            )
            tp_order_id = tp_resp.get("orderId")
        except Exception as exc:
            logger.warning(f"{symbol}: TP bracket order failed: {exc}")
            tp_failed = True

        # Stop-market SL order
        try:
            sl_resp = self._client.place_order(
                symbol, close_side, qty, order_type="STOP_MARKET",
                stop_price=sl_price, reduce_only=True,
            )
            sl_order_id = sl_resp.get("orderId")
        except Exception as exc:
            logger.warning(f"{symbol}: SL bracket order failed: {exc}")
            sl_failed = True

        # If BOTH bracket orders failed, position is unprotected — close immediately
        if tp_failed and sl_failed:
            logger.critical(
                f"{symbol}: BOTH TP and SL bracket orders failed — closing position immediately to avoid naked exposure"
            )
            try:
                self._client.place_order(symbol, close_side, qty, order_type="MARKET", reduce_only=True)
                logger.info(f"{symbol}: emergency market close placed after double bracket failure")
            except Exception as close_exc:
                logger.critical(f"{symbol}: emergency market close ALSO failed: {close_exc}")
            return None

        # If only one bracket failed, retry once after 2 seconds
        if tp_failed or sl_failed:
            time.sleep(2)
            if tp_failed:
                try:
                    tp_resp = self._client.place_order(
                        symbol, close_side, qty, order_type="LIMIT",
                        price=tp_price, reduce_only=True,
                    )
                    tp_order_id = tp_resp.get("orderId")
                    tp_failed = False
                    logger.info(f"{symbol}: TP bracket retry succeeded — orderId={tp_order_id}")
                except Exception as exc:
                    logger.warning(f"{symbol}: TP bracket retry also failed: {exc}")
            if sl_failed:
                try:
                    sl_resp = self._client.place_order(
                        symbol, close_side, qty, order_type="STOP_MARKET",
                        stop_price=sl_price, reduce_only=True,
                    )
                    sl_order_id = sl_resp.get("orderId")
                    sl_failed = False
                    logger.info(f"{symbol}: SL bracket retry succeeded — orderId={sl_order_id}")
                except Exception as exc:
                    logger.warning(f"{symbol}: SL bracket retry also failed: {exc}")

        # After retry: if either bracket still missing, position is unprotected — close immediately
        if tp_failed or sl_failed:
            missing = ("TP" if tp_failed else "") + (" SL" if sl_failed else "")
            logger.critical(
                f"{symbol}: {missing.strip()} bracket failed after retry — closing position to avoid naked exposure"
            )
            try:
                self._client.place_order(symbol, close_side, qty, order_type="MARKET", reduce_only=True)
                logger.info(f"{symbol}: emergency market close placed after single bracket failure")
            except Exception as close_exc:
                logger.critical(f"{symbol}: emergency market close ALSO failed: {close_exc}")
            return None

        self.positions[symbol] = {
            "order_id": order_id,
            "direction": direction,
            "entry_price": entry_price,
            "size_usd": size_usd,
            "tp_price": tp_price,
            "sl_price": sl_price,
            "tp_order_id": tp_order_id,
            "sl_order_id": sl_order_id,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "regime": regime,
            "signal_strength": signal_strength,
        }

        logger.info(
            f"{symbol}: entry submitted — dir={direction} qty={qty} "
            f"entry~{entry_price} tp={tp_price} sl={sl_price} orderId={order_id}"
        )
        return order_id

    def submit_exit(self, symbol: str) -> bool:
        if symbol not in self.positions:
            logger.warning(f"{symbol}: submit_exit called but no tracked position")
            return False

        pos = self.positions[symbol]

        # Cancel bracket orders first to avoid double-fill race
        for oid_key in ("tp_order_id", "sl_order_id"):
            oid = pos.get(oid_key)
            if oid is not None:
                try:
                    self._client.cancel_order(symbol, oid)
                except Exception as exc:
                    logger.warning(f"{symbol}: cancel bracket {oid_key}={oid} failed: {exc}")

        # Flatten with a market close
        close_side = "SELL" if pos["direction"] == "long" else "BUY"
        qty_raw = pos["size_usd"] / pos["entry_price"]
        qty_step = self._client.get_qty_step(symbol)
        qty = round(qty_raw / qty_step) * qty_step
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

        # Position fully closed if exchange reports zero positionAmt
        if abs(live_pos["positionAmt"]) > 1e-9:
            return None  # still open

        # Fill detected — query recent trades to get actual exit fill price.
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

    def cancel_all_open(self) -> None:
        logger.warning("cancel_all_open: flattening all tracked positions (dead-man-switch or shutdown)")
        for symbol, pos in list(self.positions.items()):
            if pos is None:
                continue
            # Cancel bracket orders first to prevent double-fill race
            try:
                self._client.cancel_all_orders(symbol)
                logger.info(f"{symbol}: bracket orders cancelled")
            except Exception as exc:
                logger.warning(f"{symbol}: cancel_all_orders failed (continuing to market close): {exc}")
            # Issue market close to actually flatten the exchange position
            close_side = "SELL" if pos["direction"] == "long" else "BUY"
            qty_raw = pos["size_usd"] / pos["entry_price"]
            qty_step = self._client.get_qty_step(symbol)
            qty = round(qty_raw / qty_step) * qty_step
            try:
                self._client.place_order(symbol, close_side, qty, order_type="MARKET", reduce_only=True)
                logger.info(f"{symbol}: market close issued by DMS/shutdown")
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
                    self.cancel_all_open()
                except Exception as exc:
                    logger.error(f"DMS cancel_all_open failed: {exc}")
                # Reset heartbeat so we don't loop-cancel
                with self._heartbeat_lock:
                    self._last_heartbeat = time.monotonic()
