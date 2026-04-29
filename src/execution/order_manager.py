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
    "pnl_pct_net",       # pnl after round-trip fee deduction
    "fee_pct",           # round-trip fee actually charged (taker × 2)
    "regime",
    "signal_strength",
    # Drift-detection columns — let offline analysis compare live vs trained expectations
    "tp_pct_used",            # actual TP distance used at entry (live R:R numerator)
    "sl_pct_used",            # actual SL distance used at entry (live R:R denominator)
    "atr_pct_at_entry",       # ATR/price at entry — regime vol proxy vs training ATR dist
    "bars_held",              # bars between entry and exit (vs labels.max_hold_bars=32)
    "exit_reason",            # TP / SL / TIME / MANUAL / DMS / UNKNOWN
    "primary_prob_at_entry",  # raw primary model prob before meta gating
    "meta_prob_at_entry",     # meta-labeler prob (0.5 if no meta model)
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

        # Per-symbol commission rate cache — fetched from API on first trade, fallback to config
        self._commission_cache: dict[str, float] = {}
        # Default taker fee from config (used as fallback before API fetch)
        self._default_taker_fee = float(getattr(getattr(cfg, "backtest", cfg), "commission_pct", 0.001))

        self._heartbeat_lock = threading.Lock()
        self._last_heartbeat = time.monotonic()

        # Ensure CSV file has a header row if it doesn't exist yet, or migrate schema if columns were added
        if not self._log_path.exists():
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_TRADE_LOG_COLS).writeheader()
        else:
            self._migrate_trade_log_schema()

        # Start dead-man-switch background thread
        self._dms_thread = threading.Thread(target=self._dead_man_switch_loop, daemon=True)
        self._dms_thread.start()
        logger.info(f"OrderManager started — DMS={self._dms_seconds}s trade_log={self._log_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _drift_fields(self, pos: dict, exit_reason: str) -> dict:
        # Drift-detection fields: R:R, ATR regime, hold duration, prob decomposition.
        # Read at session start by monitoring tooling to compare live vs training dist.
        bar_seconds = float(getattr(self._cfg.trading, "bar_seconds", 900))  # default 15m
        entry_epoch = float(pos.get("entry_epoch", 0.0) or 0.0)
        bars_held = int((time.time() - entry_epoch) / bar_seconds) if entry_epoch > 0 else -1
        return {
            "tp_pct_used":            round(float(pos.get("tp_pct_used", 0.0)), 6),
            "sl_pct_used":            round(float(pos.get("sl_pct_used", 0.0)), 6),
            "atr_pct_at_entry":       round(float(pos.get("atr_pct_at_entry", 0.0)), 6),
            "bars_held":              bars_held,
            "exit_reason":            exit_reason,
            "primary_prob_at_entry":  round(float(pos.get("primary_prob_at_entry", 0.5)), 4),
            "meta_prob_at_entry":     round(float(pos.get("meta_prob_at_entry", 0.5)), 4),
        }

    def _get_taker_fee(self, symbol: str) -> float:
        # Returns taker fee rate for symbol — fetched from API once, cached per symbol
        if symbol not in self._commission_cache:
            try:
                rates = self._client.get_commission_rate(symbol)
                self._commission_cache[symbol] = rates["taker"]
            except Exception:
                self._commission_cache[symbol] = self._default_taker_fee
        return self._commission_cache[symbol]

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
        atr_pct: float = 0.0,
        primary_prob: float = 0.5,
        meta_prob: float = 0.5,
    ) -> str | None:
        if symbol in self.positions:
            logger.warning(f"{symbol}: submit_entry called but position already tracked — skipping")
            return None

        leverage = int(getattr(self._cfg.trading, "leverage",
                               getattr(getattr(self._cfg, "growth_gate", self._cfg), "fixed_leverage", 2)))
        qty_step = self._client.get_qty_step(symbol)
        min_notional = self._client.get_min_notional(symbol)

        # --- Step 1: determine effective max qty ---
        # Start with exchange lot size limit (both DEMO and MAINNET)
        exchange_max_qty = self._client.get_max_qty(symbol, entry_price)

        # MAINNET: also cap by leverage-bracket notional limit
        if self._mode != "DEMO":
            notional_cap = self._client.get_notional_limit(symbol, leverage)
            bracket_max_qty = notional_cap / entry_price
            max_qty_eff = min(exchange_max_qty, bracket_max_qty)
            if bracket_max_qty < exchange_max_qty:
                logger.debug(f"{symbol}: bracket notional cap {notional_cap:.0f} → max_qty {bracket_max_qty:.4f}")
        else:
            max_qty_eff = exchange_max_qty

        # --- Step 2: raw qty from size_usd ---
        qty_raw = size_usd / entry_price
        logger.debug(f"{symbol}: raw qty={qty_raw:.6f} (size_usd={size_usd:.2f} / price={entry_price:.6f}) max_qty_eff={max_qty_eff:.4f}")

        # --- Step 3: clamp to [min_qty, max_qty_eff] with smart bump ---
        # 3a. Cap to exchange/bracket max
        qty = _round_qty(min(qty_raw, max_qty_eff), qty_step)
        size_usd = qty * entry_price
        if qty < qty_raw:
            logger.info(f"{symbol}: qty capped {qty_raw:.4f}→{qty:.4f} (notional={size_usd:.2f} USDT)")

        # 3b. Bump up if below min_notional — price may have moved since filter
        if size_usd < min_notional:
            qty_bumped = _round_qty((min_notional * 1.05) / entry_price, qty_step)
            # Re-apply max cap after bump
            qty_bumped = _round_qty(min(qty_bumped, max_qty_eff), qty_step)
            bumped_notional = qty_bumped * entry_price
            if bumped_notional < min_notional:
                # Even at max allowed qty we can't reach min_notional — skip this bar,
                # but do NOT permanently exclude: price may rise on the next bar.
                logger.warning(
                    f"{symbol}: max tradeable notional {bumped_notional:.2f} < min_notional {min_notional:.2f} "
                    f"(max_qty_eff={max_qty_eff:.4f} × price={entry_price:.6f}) — skipping this bar"
                )
                return None
            logger.info(f"{symbol}: qty bumped {qty:.4f}→{qty_bumped:.4f} to meet min_notional (notional={bumped_notional:.2f})")
            qty = qty_bumped
            size_usd = bumped_notional

        # 3c. DEMO precision guard: if step from exchange is suspiciously small (< 0.001,
        # e.g. 0.0001 for symbols where DEMO actually enforces integer qty) and the qty has
        # sub-integer decimals, floor to integer to avoid -1111 precision errors.
        # Threshold is < 0.001 (strict): stepSize=0.001 symbols like ETHUSDT are genuinely
        # decimal and must NOT be floored — only truly fine-grained steps (0.0001, 0.00001)
        # indicate a DEMO quirk where the reported precision is tighter than what's accepted.
        if self._mode == "DEMO" and qty_step < 0.001 and qty > 1.0 and qty != math.floor(qty):
            qty_int = math.floor(qty)
            notional_int = qty_int * entry_price
            if notional_int >= min_notional:
                logger.debug(f"{symbol}: DEMO pre-rounding {qty:.4f}→{qty_int} (integer guard for -1111)")
                qty = float(qty_int)
                size_usd = notional_int

        logger.info(f"{symbol}: final qty={qty} notional={qty * entry_price:.2f} USDT min_notional={min_notional:.2f}")

        # --- Step 4: place market entry order ---
        entry_side = "BUY" if direction == "long" else "SELL"
        try:
            entry_resp = self._client.place_order(symbol, entry_side, qty, order_type="MARKET")
        except Exception as exc:
            exc_str = str(exc)
            # -1111 fallback: exchange rejected our precision — reduce to integer qty and retry once
            if "-1111" in exc_str and qty != math.floor(qty):
                qty_int = math.floor(qty)
                if qty_int >= 1 and qty_int * entry_price >= min_notional:
                    logger.warning(f"{symbol}: -1111 precision error, integer retry qty={qty_int}")
                    qty = float(qty_int)
                    size_usd = qty * entry_price
                    try:
                        entry_resp = self._client.place_order(symbol, entry_side, qty, order_type="MARKET")
                    except Exception as exc2:
                        logger.error(f"{symbol}: entry failed after integer retry: {exc2}")
                        return None
                else:
                    logger.error(f"{symbol}: -1111 and integer qty {math.floor(qty)} below min_notional — skip: {exc}")
                    return None
            else:
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
        # DEMO: bracket orders are NOT sent to exchange — TP/SL simulated via mark price in sync_fills.
        # No PERCENT_PRICE cap needed for DEMO (cap was only relevant for real exchange stop orders).
        # MAINNET: real STOP_MARKET orders must stay within exchange price bands — no special cap needed
        # because stopPrice is allowed far from mark price on MAINNET FAPI.
        tick = self._client.get_tick_size(symbol)

        if direction == "long":
            raw_tp = entry_price * (1.0 + tp_pct)
            raw_sl = entry_price * (1.0 - sl_pct)
            tp_price = self._client.round_price(symbol, raw_tp)
            if tp_price <= entry_price:
                tp_price = round(entry_price + tick, 10)
            # DEMO: floor SL so rounding never pushes it past the intended level
            sl_price = round(math.floor(raw_sl / tick) * tick, 10) if self._mode == "DEMO" else self._client.round_price(symbol, raw_sl)
            if sl_price >= entry_price:
                sl_price = round(entry_price - tick, 10)
            close_side = "SELL"
        else:
            raw_tp = entry_price * (1.0 - tp_pct)
            raw_sl = entry_price * (1.0 + sl_pct)
            tp_price = self._client.round_price(symbol, raw_tp)
            if tp_price >= entry_price:
                tp_price = round(entry_price - tick, 10)
            # DEMO: ceil SL (above entry for shorts) so rounding never pulls it below the intended level
            sl_price = round(math.ceil(raw_sl / tick) * tick, 10) if self._mode == "DEMO" else self._client.round_price(symbol, raw_sl)
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
            "entry_epoch": time.time(),  # bars_held computed as (exit_epoch - entry_epoch) / bar_seconds
            "regime": regime,
            "signal_strength": signal_strength,
            "tp_pct_used": tp_pct,
            "sl_pct_used": sl_pct,
            "atr_pct_at_entry": atr_pct,
            "primary_prob_at_entry": primary_prob,
            "meta_prob_at_entry": meta_prob,
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
                        fee_pct = self._get_taker_fee(symbol) * 2  # round-trip: entry + exit
                        fill = {
                            "timestamp_entry": pos["entry_time"],
                            "timestamp_exit": datetime.now(timezone.utc).isoformat(),
                            "symbol": symbol,
                            "direction": direction,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "size_usd": pos["size_usd"],
                            "pnl_pct": round(pnl_pct, 6),
                            "pnl_pct_net": round(pnl_pct - fee_pct, 6),
                            "fee_pct": round(fee_pct, 6),
                            "regime": pos.get("regime", ""),
                            "signal_strength": pos.get("signal_strength", 0.0),
                            **self._drift_fields(pos, exit_reason=hit),
                        }
                        self._write_trade_log(fill)
                        del self.positions[symbol]
                        logger.info(f"{symbol}: DEMO {hit} fill recorded — pnl_pct={pnl_pct:.4%} net={pnl_pct - fee_pct:.4%}")
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

        fee_pct = self._get_taker_fee(symbol) * 2  # round-trip: entry + exit

        # Infer exit reason from which bracket price the exit is closer to
        tp_price = pos.get("tp_price") or 0
        sl_price = pos.get("sl_price") or 0
        if tp_price and sl_price:
            exit_reason = "TP" if abs(exit_price - tp_price) <= abs(exit_price - sl_price) else "SL"
        else:
            exit_reason = "UNKNOWN"

        fill = {
            "timestamp_entry": pos["entry_time"],
            "timestamp_exit": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": pos["direction"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size_usd": pos["size_usd"],
            "pnl_pct": round(pnl_pct, 6),
            "pnl_pct_net": round(pnl_pct - fee_pct, 6),
            "fee_pct": round(fee_pct, 6),
            "regime": pos.get("regime", ""),
            "signal_strength": pos.get("signal_strength", 0.0),
            **self._drift_fields(pos, exit_reason=exit_reason),
        }

        self._write_trade_log(fill)
        del self.positions[symbol]
        logger.info(f"{symbol}: fill recorded — pnl_pct={pnl_pct:.4%} net={pnl_pct - fee_pct:.4%}")
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

    def _migrate_trade_log_schema(self) -> None:
        # If the existing CSV header is missing any current column, rewrite the file with the
        # full header, filling blanks for old rows. Safe no-op if schemas match.
        try:
            with open(self._log_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_cols = reader.fieldnames or []
                if set(_TRADE_LOG_COLS).issubset(set(existing_cols)):
                    return  # already up to date
                rows = list(reader)
            with open(self._log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_TRADE_LOG_COLS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({col: row.get(col, "") for col in _TRADE_LOG_COLS})
            logger.info(f"Trade log schema migrated: added {set(_TRADE_LOG_COLS) - set(existing_cols)}")
        except Exception as exc:
            logger.warning(f"Trade log schema migration failed: {exc}")

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
