import socket
import ssl
import time
import warnings
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger("market_data_fetcher")

# Binance FAPI base URL — free historical endpoints, no auth required
_FAPI_BASE = "https://fapi.binance.com"

# Alternative hostnames for Binance FAPI (fallback when primary is blocked)
_FAPI_FALLBACK_HOSTS = [
    "fapi1.binance.com",
    "fapi2.binance.com",
    "fapi3.binance.com",
    "fapi4.binance.com",
]

# Public DNS resolvers to query directly, bypassing ISP DNS hijacking
_DNS_SERVERS = ["8.8.8.8", "1.1.1.1", "9.9.9.9", "208.67.222.222"]

# DoH fallback resolvers (used if dnspython UDP is blocked)
_DOH_RESOLVERS = [
    "https://1.1.1.1/dns-query",
    "https://8.8.8.8/dns-query",
    "https://9.9.9.9/dns-query",
]

# Cached resolved IPs so we don't re-resolve on every request
_DNS_CACHE: dict[str, str] = {}


def _resolve_via_dnspython(hostname: str) -> str | None:
    """Query DNS directly to a public resolver via UDP, bypassing local ISP DNS."""
    try:
        import dns.resolver
        resolver = dns.resolver.Resolver(configure=False)
        resolver.nameservers = _DNS_SERVERS
        resolver.lifetime = 5.0
        answers = resolver.resolve(hostname, "A")
        for rdata in answers:
            ip = str(rdata)
            if ip and not ip.startswith("0."):
                return ip
    except Exception:
        pass
    return None


def _resolve_via_doh(hostname: str) -> str | None:
    """Resolve hostname via DNS-over-HTTPS (fallback when UDP DNS is also blocked)."""
    try:
        import requests as _req
        for resolver_url in _DOH_RESOLVERS:
            try:
                resp = _req.get(
                    resolver_url,
                    params={"name": hostname, "type": "A"},
                    headers={"Accept": "application/dns-json"},
                    timeout=5,
                    verify=True,
                )
                if resp.status_code == 200:
                    for ans in resp.json().get("Answer", []):
                        if ans.get("type") == 1:
                            ip = ans.get("data", "")
                            if ip and not ip.startswith("0."):
                                return ip
            except Exception:
                continue
    except Exception:
        pass
    return None


def resolve_real_ip(hostname: str) -> str | None:
    """Resolve hostname to its real IP, bypassing ISP DNS hijacking. Results cached."""
    if hostname in _DNS_CACHE:
        return _DNS_CACHE[hostname]
    ip = _resolve_via_dnspython(hostname) or _resolve_via_doh(hostname)
    if ip:
        _DNS_CACHE[hostname] = ip
        logger.debug(f"DNS bypass: {hostname} → {ip}")
    return ip


def _is_intercepted(resp) -> bool:
    """Return True if the response came from an ISP block page (not real Binance)."""
    url = str(getattr(resp, "url", "") or "")
    return "internet-positif" in url or "nawala" in url or "trustpositif" in url

# Max rows per request for 15m period endpoints
_LIMIT = 500

# Endpoint map: name → FAPI path
_ENDPOINTS = {
    "oi": "/futures/data/openInterestHist",
    "global_ls": "/futures/data/globalLongShortAccountRatio",
    "top_position_ls": "/futures/data/topLongShortPositionRatio",
    "top_account_ls": "/futures/data/topLongShortAccountRatio",
    "taker_ratio": "/futures/data/takerlongshortRatio",
}

# Output file naming per data type
_FILE_SUFFIX = {
    "oi": "oi_15m",
    "global_ls": "ls_global_15m",
    "top_position_ls": "ls_top_position_15m",
    "top_account_ls": "ls_top_account_15m",
    "taker_ratio": "taker_ratio_15m",
}


def _make_bypass_session(hostname: str) -> "requests.Session":
    """
    Build a requests.Session that connects to the real Binance IP (resolved via dnspython)
    but sends the correct SNI + Host header so TLS validates normally.
    This bypasses Internet Positif DNS hijacking without disabling SSL verification.
    Falls back to normal session if IP resolve fails (VPN active = DNS not hijacked).
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    import urllib3.connection

    real_ip = resolve_real_ip(hostname)

    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])

    if real_ip:
        _ip = real_ip
        _host = hostname

        class _BypassConn(urllib3.connection.HTTPSConnection):
            def connect(self):
                sock = socket.create_connection((_ip, self.port or 443), timeout=self.timeout or 15)
                ctx = ssl.create_default_context()
                self.sock = ctx.wrap_socket(sock, server_hostname=_host)
                self.is_verified = True

        class _BypassPool(urllib3.HTTPSConnectionPool):
            def _new_conn(self):
                return _BypassConn(host=self.host, port=self.port, timeout=self.timeout)

        class _BypassAdapter(HTTPAdapter):
            def __init__(self, **kw):
                super().__init__(max_retries=retry, **kw)
            def get_connection(self, url, proxies=None):
                from urllib.parse import urlparse
                parsed = urlparse(url)
                return _BypassPool(parsed.hostname, port=parsed.port or 443)

        session.mount(f"https://{hostname}", _BypassAdapter())
        logger.debug(f"Bypass session: {hostname} → {real_ip} (DNS hijack bypass active)")
    else:
        # Normal session — VPN is on or DNS is not hijacked
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

    return session


def _get_session():
    # For backwards compat — used in paginate_endpoint which doesn't need bypass per-call
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _fetch_with_bypass(session, url: str, params: dict, hostname: str, timeout: int = 30):
    """
    Fetch url with ISP bypass strategy:
      1. Bypass session (direct IP + correct SNI) — works when DNS is hijacked
      2. Normal requests fallback — works when VPN is on
    """
    import requests
    warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    # Attempt 1: bypass session (direct IP connection with correct SNI)
    bypass = _make_bypass_session(hostname)
    try:
        resp = bypass.get(url, params=params, timeout=timeout)
        if not _is_intercepted(resp):
            resp.raise_for_status()
            return resp
    except requests.exceptions.HTTPError as e:
        resp_obj = getattr(e, "response", None)
        if resp_obj is not None and resp_obj.status_code not in (403, 301, 302):
            raise
    except Exception:
        pass

    # Attempt 2: plain session (VPN active path)
    try:
        resp = session.get(url, params=params, timeout=timeout)
        if not _is_intercepted(resp):
            resp.raise_for_status()
            return resp
    except Exception:
        pass

    # Attempt 3: fallback Binance hostnames
    for fb_host in _FAPI_FALLBACK_HOSTS:
        fb_url = url.replace(f"https://{hostname}", f"https://{fb_host}")
        try:
            fb_bypass = _make_bypass_session(fb_host)
            resp = fb_bypass.get(fb_url, params=params, timeout=timeout)
            if not _is_intercepted(resp):
                resp.raise_for_status()
                return resp
        except Exception:
            continue

    return None


def _fetch_endpoint(session, endpoint_path: str, symbol: str, period: str = "15m",
                    start_ms: int = None, end_ms: int = None) -> list:
    from urllib.parse import urlparse
    url = f"{_FAPI_BASE}{endpoint_path}"
    params = {"symbol": symbol, "period": period, "limit": _LIMIT}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    hostname = urlparse(url).hostname
    try:
        resp = _fetch_with_bypass(session, url, params, hostname)
        if resp is not None:
            return resp.json()
    except Exception as e:
        logger.warning(f"Fetch failed for {symbol} {endpoint_path}: {e}")
    return []


def _paginate_endpoint(session, endpoint_path: str, symbol: str, period: str,
                       start_ms: int, end_ms: int) -> list:
    # Paginate forward in time using startTime + limit
    all_rows = []
    cur_start = start_ms
    while cur_start < end_ms:
        rows = _fetch_endpoint(session, endpoint_path, symbol, period, cur_start, end_ms)
        if not rows:
            break
        all_rows.extend(rows)
        # Advance past last returned timestamp
        last_ts = rows[-1].get("timestamp", rows[-1].get("timestampStr"))
        if last_ts is None:
            break
        try:
            last_ts = int(last_ts)
        except (TypeError, ValueError):
            break
        if last_ts <= cur_start:
            break
        cur_start = last_ts + 1  # 1ms after last row
        if len(rows) < _LIMIT:
            break
        time.sleep(0.1)  # rate limit courtesy pause
    return all_rows


def _rows_to_df(rows: list, value_cols: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    # Timestamp column may be 'timestamp' (ms int) or 'timestampStr'
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    elif "timestampStr" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestampStr"], utc=True)
    else:
        logger.warning("No timestamp column in response")
        return pd.DataFrame()

    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Keep only numeric value columns that exist
    keep = [c for c in value_cols if c in df.columns]
    if not keep:
        # Fallback: keep all numeric cols
        keep = df.select_dtypes("number").columns.tolist()

    for col in keep:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[keep]


def fetch_symbol_market_data(symbol: str, raw_dir: Path, start_ms: int = None, end_ms: int = None) -> dict:
    """
    Fetch OI, LS ratios, and taker ratio for one symbol from Binance FAPI.
    Saves each dataset as parquet in raw_dir.
    Returns dict of {data_type: DataFrame} for successfully fetched datasets.

    start_ms / end_ms: Unix epoch milliseconds. If None, fetches last _LIMIT bars.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    session = _get_session()
    period = "15m"

    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        # Binance FAPI free OI/LS endpoints only keep ~30 days of history
        # For initial fetch try 30 days; paginator will stop when data runs out
        start_ms = end_ms - int(30 * 24 * 3600 * 1000)

    results = {}

    endpoint_value_cols = {
        "oi": ["sumOpenInterest", "sumOpenInterestValue"],
        "global_ls": ["longShortRatio", "longAccount", "shortAccount"],
        "top_position_ls": ["longShortRatio", "longAccount", "shortAccount"],
        "top_account_ls": ["longShortRatio", "longAccount", "shortAccount"],
        "taker_ratio": ["buySellRatio", "buyVol", "sellVol"],
    }

    for key, endpoint_path in _ENDPOINTS.items():
        out_path = raw_dir / f"{symbol}_{_FILE_SUFFIX[key]}.parquet"
        value_cols = endpoint_value_cols[key]

        # Load existing file to avoid re-fetching already-saved data
        existing_df = None
        existing_end_ms = None
        if out_path.exists():
            try:
                existing_df = pd.read_parquet(out_path)
                if not existing_df.empty and isinstance(existing_df.index, pd.DatetimeIndex):
                    existing_end_ms = int(existing_df.index.max().timestamp() * 1000)
            except Exception:
                existing_df = None

        fetch_start = existing_end_ms + 1 if existing_end_ms else start_ms

        if fetch_start >= end_ms:
            logger.debug(f"{symbol} {key}: already up to date")
            if existing_df is not None:
                results[key] = existing_df
            continue

        logger.info(f"{symbol}: fetching {key} from {pd.Timestamp(fetch_start, unit='ms', tz='UTC').date()}")
        rows = _paginate_endpoint(session, endpoint_path, symbol, period, fetch_start, end_ms)

        if not rows:
            logger.warning(f"{symbol}: no {key} data returned")
            if existing_df is not None:
                results[key] = existing_df
            continue

        new_df = _rows_to_df(rows, value_cols)
        if new_df.empty:
            if existing_df is not None:
                results[key] = existing_df
            continue

        # Merge with existing
        if existing_df is not None and not existing_df.empty:
            combined = pd.concat([existing_df, new_df]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = new_df

        combined.to_parquet(out_path)
        results[key] = combined
        logger.info(f"{symbol}: saved {key} — {len(combined)} rows to {out_path.name}")

    return results


def load_symbol_market_data(symbol: str, raw_dir: Path) -> dict:
    """Load previously fetched market data for a symbol from parquet files."""
    raw_dir = Path(raw_dir)
    results = {}
    for key, suffix in _FILE_SUFFIX.items():
        path = raw_dir / f"{symbol}_{suffix}.parquet"
        if path.exists():
            try:
                df = pd.read_parquet(path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index, utc=True)
                elif df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                else:
                    df.index = df.index.tz_convert("UTC")
                results[key] = df
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")
    return results
