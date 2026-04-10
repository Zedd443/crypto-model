"""
Unit tests for OrderManager.submit_entry sizing logic.
Focuses on quantity calculation, bumping, max-cap, and -1111 fallback.
All exchange API calls are mocked — no network required.
"""
import math
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


def _make_cfg(mode="DEMO", leverage=2, taker_fee=0.0004):
    cfg = MagicMock()
    cfg.trading.mode = mode
    cfg.trading.leverage = leverage
    cfg.trading.dead_man_switch_seconds = 60.0
    cfg.backtest.commission_pct = taker_fee
    cfg.growth_gate.demo_trades_required = 50
    return cfg


def _make_client(
    mode="DEMO",
    qty_step=1.0,
    max_qty=120.0,
    min_notional=100.0,
    tick=0.01,
    notional_limit=1e9,
    commission={"taker": 0.0004, "maker": 0.0002},
    order_resp=None,
    position_amt=0.0,
):
    client = MagicMock()
    client.get_qty_step.return_value = qty_step
    client.get_max_qty.return_value = max_qty
    client.get_min_notional.return_value = min_notional
    client.get_tick_size.return_value = tick
    client.get_notional_limit.return_value = notional_limit
    client.get_commission_rate.return_value = commission
    client.round_price.side_effect = lambda sym, p: round(p, 2)
    client.place_order.return_value = order_resp or {"orderId": "TEST123", "avgPrice": "0"}
    client.get_position.return_value = {
        "positionAmt": position_amt,
        "entryPrice": 0.0,
        "unrealizedProfit": 0.0,
        "markPrice": 0.0,
    }
    return client


def _make_om(client, cfg, mode="DEMO", log_path=None):
    from src.execution.order_manager import OrderManager
    with patch("threading.Thread"):  # don't start DMS thread
        with patch("builtins.open", unittest.mock.mock_open()):
            with patch("pathlib.Path.exists", return_value=True):
                om = OrderManager.__new__(OrderManager)
                om._client = client
                om._cfg = cfg
                om._log_path = Path(log_path or "results/live_trade_log.csv")
                om._dms_seconds = 60.0
                om._mode = mode.upper()
                om._commission_cache = {}
                om._default_taker_fee = 0.0004
                om.positions = {}
                import threading, time
                om._heartbeat_lock = threading.Lock()
                om._last_heartbeat = time.monotonic()
    return om


class TestSubmitEntrySizing(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1. Normal cap: size_usd >> max_qty * price → capped to max_qty
    # ------------------------------------------------------------------
    def test_cap_to_max_qty(self):
        # SOL at $100, max_qty=120 → raw qty = 9794/100 = 97.94 → should cap to 97 (floor)
        # Actually 97.94 < 120, no cap needed. Use price $10 → raw=979 → capped to 120
        client = _make_client(qty_step=1.0, max_qty=120.0, min_notional=100.0)
        cfg = _make_cfg()
        om = _make_om(client, cfg)

        result = om.submit_entry("TESTUSDT", "long", size_usd=9790.0, entry_price=10.0, tp_pct=0.02, sl_pct=0.01)

        self.assertEqual(result, "TEST123")
        # Should have placed order with qty=120 (capped)
        call_args = client.place_order.call_args_list[0]
        placed_qty = call_args[0][2]
        self.assertEqual(placed_qty, 120.0)

    # ------------------------------------------------------------------
    # 2. Bump up: size_usd < min_notional but max_qty allows it
    # ------------------------------------------------------------------
    def test_bump_to_min_notional(self):
        # ALGO at $0.11, max_qty=1000, min_notional=100
        # size_usd = $50 → qty_raw=454 → qty=454 → notional=$49.94 < $100 → bump
        client = _make_client(qty_step=0.0001, max_qty=1000.0, min_notional=100.0)
        cfg = _make_cfg()
        om = _make_om(client, cfg)

        result = om.submit_entry("ALGOUSDT", "long", size_usd=50.0, entry_price=0.11, tp_pct=0.02, sl_pct=0.01)

        self.assertEqual(result, "TEST123")
        placed_qty = client.place_order.call_args_list[0][0][2]
        placed_notional = placed_qty * 0.11
        self.assertGreaterEqual(placed_notional, 100.0)

    # ------------------------------------------------------------------
    # 3. Skip when even max_qty can't reach min_notional (GALA at $0.003)
    # ------------------------------------------------------------------
    def test_skip_when_max_notional_below_min(self):
        # GALA at $0.003, max_qty=120, min_notional=100 → max notional=$0.36 → skip
        client = _make_client(qty_step=0.0001, max_qty=120.0, min_notional=100.0)
        cfg = _make_cfg()
        om = _make_om(client, cfg)

        result = om.submit_entry("GALAUSDT", "long", size_usd=9000.0, entry_price=0.003, tp_pct=0.02, sl_pct=0.01)

        self.assertIsNone(result)
        client.place_order.assert_not_called()

    # ------------------------------------------------------------------
    # 4. MAINNET leverage-bracket cap is binding
    # ------------------------------------------------------------------
    def test_mainnet_bracket_cap(self):
        # BTC at $72000, max_qty=120, bracket cap=$10000 → max_qty_eff = 10000/72000 = 0.138...
        client = _make_client(qty_step=0.001, max_qty=120.0, min_notional=100.0, notional_limit=10000.0)
        cfg = _make_cfg(mode="MAINNET")
        om = _make_om(client, cfg, mode="MAINNET")

        result = om.submit_entry("BTCUSDT", "long", size_usd=50000.0, entry_price=72000.0, tp_pct=0.02, sl_pct=0.01)

        self.assertEqual(result, "TEST123")
        placed_qty = client.place_order.call_args_list[0][0][2]
        placed_notional = placed_qty * 72000.0
        self.assertLessEqual(placed_notional, 10000.0 * 1.01)  # within 1% of cap

    # ------------------------------------------------------------------
    # 5. DEMO -1111 integer fallback: qty=58.8 → try 58 first (pre-guard)
    # ------------------------------------------------------------------
    def test_demo_integer_guard_pre_order(self):
        # SOL at $83, step=0.0001, max_qty=120, size_usd=4897
        # qty_raw = 59.0, step=0.0001 → qty=59.0 (already integer)
        # Use step=0.001 to trigger the sub-integer guard: qty_raw=58.97 → floor to 58
        client = _make_client(qty_step=0.001, max_qty=120.0, min_notional=100.0)
        cfg = _make_cfg()
        om = _make_om(client, cfg)

        result = om.submit_entry("SOLUSDT", "long", size_usd=4894.51, entry_price=83.0, tp_pct=0.02, sl_pct=0.01)

        self.assertEqual(result, "TEST123")
        placed_qty = client.place_order.call_args_list[0][0][2]
        # qty should be integer (58 or 59 depending on rounding)
        self.assertEqual(placed_qty, math.floor(placed_qty))

    # ------------------------------------------------------------------
    # 6. -1111 exception path: order raises -1111 → retry with floor(qty)
    # This tests the exception fallback for cases where the pre-guard didn't trigger
    # (e.g. qty_step=1.0 so qty is already integer, but exchange rejects with -1111 anyway
    #  for a different reason). We simulate by using a large step that makes qty non-integer
    #  after rounding but bypass the pre-guard threshold.
    # ------------------------------------------------------------------
    def test_demo_1111_exception_retry(self):
        # Use qty_step=10.0 so pre-guard (qty_step <= 0.001) is NOT triggered.
        # qty_raw = 4894 / 83.0 = 58.96 → rounded to step 10 → 50.0 (integer, pre-guard won't help)
        # But we simulate exchange rejecting with -1111 and returning a non-integer qty somehow.
        # Easiest: mock place_order to raise -1111 on first call with a non-integer qty scenario.
        # We need qty != floor(qty) to trigger the retry path.
        # Use step=0.1: qty_raw=58.97 → floor to 58.9 → non-integer
        client = _make_client(qty_step=0.1, max_qty=120.0, min_notional=100.0)
        # pre-guard: qty_step=0.1 > 0.001 → won't trigger pre-guard → qty=58.9 (non-integer)
        client.place_order.side_effect = [Exception("-1111 Precision"), {"orderId": "RETRY123", "avgPrice": "0"}]
        cfg = _make_cfg()
        om = _make_om(client, cfg)

        result = om.submit_entry("SOLUSDT", "long", size_usd=4888.7, entry_price=83.0, tp_pct=0.02, sl_pct=0.01)

        # Should have retried with floor(58.9)=58 and returned RETRY123
        self.assertEqual(result, "RETRY123")
        self.assertEqual(client.place_order.call_count, 2)
        retry_qty = client.place_order.call_args_list[1][0][2]
        self.assertEqual(retry_qty, math.floor(retry_qty))  # must be integer

    # ------------------------------------------------------------------
    # 7. Position already tracked → immediate None
    # ------------------------------------------------------------------
    def test_skip_existing_position(self):
        client = _make_client()
        cfg = _make_cfg()
        om = _make_om(client, cfg)
        om.positions["BTCUSDT"] = {"order_id": "existing"}

        result = om.submit_entry("BTCUSDT", "long", size_usd=5000.0, entry_price=72000.0, tp_pct=0.02, sl_pct=0.01)

        self.assertIsNone(result)
        client.place_order.assert_not_called()

    # ------------------------------------------------------------------
    # 8. Exchange info unavailable → fallback defaults, still attempts order
    # ------------------------------------------------------------------
    def test_graceful_degradation_no_exchange_info(self):
        client = _make_client(qty_step=1.0, max_qty=float("inf"), min_notional=5.0)
        cfg = _make_cfg()
        om = _make_om(client, cfg)

        result = om.submit_entry("UNKNOWNUSDT", "short", size_usd=500.0, entry_price=10.0, tp_pct=0.02, sl_pct=0.01)

        self.assertEqual(result, "TEST123")
        placed_qty = client.place_order.call_args_list[0][0][2]
        # qty_raw = 500/10 = 50, no cap → 50
        self.assertEqual(placed_qty, 50.0)


if __name__ == "__main__":
    unittest.main()
