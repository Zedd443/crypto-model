import hashlib
import hmac
import os
import time
from urllib.parse import urlencode

import pandas as pd
import requests

from src.utils.logger import get_logger

logger = get_logger("binance_client")


class BinanceClient:
    def __init__(self, cfg):
        mode = cfg.trading.mode  # "DEMO" or "MAINNET"
        self._base_url = cfg.trading.endpoints[mode]
        self._backoff_base = float(cfg.trading.rate_limit_backoff_base)
        self._backoff_max = float(cfg.trading.rate_limit_backoff_max)

        # Select keys based on mode
        # DEMO    → BINANCE_DEMO_API_KEY / BINANCE_DEMO_API_SECRET (testnet.binancefuture.com — paper trading)
        # MAINNET → BINANCE_API_KEY / BINANCE_API_SECRET (fapi.binance.com — real money)
        if mode == "DEMO":
            self._api_key = os.environ.get("BINANCE_DEMO_API_KEY", "")
            self._api_secret = os.environ.get("BINANCE_DEMO_API_SECRET", "")
        else:  # MAINNET
            self._api_key = os.environ.get("BINANCE_API_KEY", "")
            self._api_secret = os.environ.get("BINANCE_API_SECRET", "")

        if not self._api_key:
            logger.warning(f"No API key found for mode={mode} — check .env (BINANCE_{mode}_API_KEY)")

        self._session = requests.Session()
        # Only set the API key header — do NOT set Content-Type here.
        # POST endpoints require application/x-www-form-urlencoded; requests sets
        # the correct Content-Type automatically based on whether data= or params= is used.
        self._session.headers.update({
            "X-MBX-APIKEY": self._api_key,
        })
        # Cache for symbol exchange info (step size, max qty, max notional)
        self._qty_step_cache: dict[str, float] = {}
        self._qty_max_cache: dict[str, float] = {}   # max order qty from LOT_SIZE
        self._notional_max_cache: dict[str, float] = {}  # max notional from MARKET_LOT_SIZE / MIN_NOTIONAL
        logger.info(f"BinanceClient initialised — mode={mode} endpoint={self._base_url}")

    def _sign(self, params: dict) -> dict:
        # Append timestamp + recvWindow, then HMAC-SHA256 the query string
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        sig = hmac.new(
            self._api_secret.encode(),
            query.encode(),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = sig
        return params

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False) -> dict | list:
        params = params or {}
        if signed:
            params = self._sign(params)

        url = self._base_url + path
        delay = self._backoff_base
        attempt = 0

        while True:
            try:
                resp = self._session.request(method, url, params=params if method == "GET" else None,
                                             data=params if method != "GET" else None)
            except requests.RequestException as exc:
                logger.error(f"Request error {path}: {exc}")
                raise

            if resp.status_code in (429, 418):
                # Rate-limited — back off exponentially
                retry_after = float(resp.headers.get("Retry-After", delay))
                wait = min(max(delay, retry_after), self._backoff_max)
                logger.warning(f"Rate limited ({resp.status_code}) on {path} — sleeping {wait:.1f}s (attempt {attempt})")
                time.sleep(wait)
                delay = min(delay * 2, self._backoff_max)
                attempt += 1
                continue

            if resp.status_code >= 400:
                logger.error(f"HTTP {resp.status_code} on {path}: {resp.text}")
                resp.raise_for_status()

            return resp.json()

    def get_klines(self, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
        raw = self._request("GET", "/fapi/v1/klines", params={
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        })
        # Each element: [open_time, open, high, low, close, volume, close_time, ...]
        rows = []
        for k in raw:
            rows.append({
                "open_time": pd.Timestamp(k[0], unit="ms", tz="UTC"),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        df = pd.DataFrame(rows).set_index("open_time")
        df.index.name = "timestamp"
        return df

    def get_account(self) -> dict:
        return self._request("GET", "/fapi/v2/account", signed=True)

    def get_position(self, symbol: str) -> dict:
        data = self._request("GET", "/fapi/v2/positionRisk", params={"symbol": symbol}, signed=True)
        # Returns list; grab first match
        if isinstance(data, list):
            for p in data:
                if p.get("symbol") == symbol:
                    return {
                        "positionAmt": float(p.get("positionAmt", 0)),
                        "entryPrice": float(p.get("entryPrice", 0)),
                        "unrealizedProfit": float(p.get("unRealizedProfit", 0)),
                    }
        return {"positionAmt": 0.0, "entryPrice": 0.0, "unrealizedProfit": 0.0}

    def get_open_orders(self, symbol: str) -> list:
        return self._request("GET", "/fapi/v1/openOrders", params={"symbol": symbol}, signed=True)

    def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "MARKET",
        price: float | None = None,
        stop_price: float | None = None,
        reduce_only: bool = False,
    ) -> dict:
        params: dict = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": qty,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if price is not None:
            params["price"] = price
            params["timeInForce"] = "GTC"
        if stop_price is not None:
            params["stopPrice"] = stop_price
        logger.info(f"place_order {symbol} {side} {order_type} qty={qty} price={price} stopPrice={stop_price}")
        return self._request("POST", "/fapi/v1/order", params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        logger.info(f"cancel_order {symbol} orderId={order_id}")
        return self._request("DELETE", "/fapi/v1/order", params={
            "symbol": symbol,
            "orderId": order_id,
        }, signed=True)

    def cancel_all_orders(self, symbol: str) -> list:
        logger.info(f"cancel_all_orders {symbol}")
        return self._request("DELETE", "/fapi/v1/allOpenOrders", params={"symbol": symbol}, signed=True)

    def get_recent_trades(self, symbol: str, limit: int = 10) -> list:
        # GET /fapi/v1/userTrades — signed, returns recent fills for this account/symbol
        return self._request("GET", "/fapi/v1/userTrades", params={"symbol": symbol, "limit": limit}, signed=True)

    def get_server_time(self) -> int:
        data = self._request("GET", "/fapi/v1/time")
        return int(data["serverTime"])

    def _fetch_symbol_info(self, symbol: str) -> None:
        """Fetch and cache all relevant exchange filters for a symbol (one API call)."""
        try:
            data = self._request("GET", "/fapi/v1/exchangeInfo", {"symbol": symbol})
            if "symbols" not in data or len(data["symbols"]) == 0:
                raise ValueError("empty symbols list")
            sym_info = data["symbols"][0]
            for filt in sym_info.get("filters", []):
                ft = filt.get("filterType")
                if ft == "LOT_SIZE":
                    self._qty_step_cache[symbol] = float(filt.get("stepSize", 1.0))
                    max_qty = float(filt.get("maxQty", 0))
                    if max_qty > 0:
                        self._qty_max_cache[symbol] = max_qty
                elif ft == "MARKET_LOT_SIZE":
                    max_qty = float(filt.get("maxQty", 0))
                    if max_qty > 0:
                        # MARKET_LOT_SIZE overrides LOT_SIZE max for market orders
                        self._qty_max_cache[symbol] = max_qty
            logger.debug(
                f"{symbol}: step={self._qty_step_cache.get(symbol)} "
                f"max_qty={self._qty_max_cache.get(symbol)}"
            )
        except Exception as e:
            logger.warning(f"{symbol}: could not fetch exchange info — {e}")
            # Fallbacks
            self._qty_step_cache.setdefault(symbol, 1.0)

    def get_qty_step(self, symbol: str) -> float:
        """Minimum quantity step size for a symbol (lot size precision)."""
        if symbol not in self._qty_step_cache:
            self._fetch_symbol_info(symbol)
        return self._qty_step_cache.get(symbol, 1.0)

    def get_max_qty(self, symbol: str, price: float) -> float:
        """Maximum order quantity for a symbol. Returns inf if no limit found."""
        if symbol not in self._qty_step_cache:
            self._fetch_symbol_info(symbol)
        return self._qty_max_cache.get(symbol, float("inf"))
