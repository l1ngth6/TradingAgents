"""Binance Spot UTC-daily OHLCV fallback for cryptocurrency market analysis.

Yahoo remains the first configured market-data vendor for compatibility. When
Yahoo has not yet published the exact completed crypto day, this module supplies
the corresponding Binance Spot candle instead of silently substituting an older
day. Binance ``Volume`` is base-asset volume and is kept together with Binance
OHLC prices; aggregate Yahoo/Coin Metrics volume is never mixed into it.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from functools import lru_cache
from typing import Annotated

import pandas as pd
import requests

from .errors import NoMarketDataError, VendorRateLimitError
from .stockstats_utils import (
    _assert_crypto_cutoff_available,
    _assert_ohlcv_not_stale,
    _clean_dataframe,
)
from .symbol_utils import crypto_base, crypto_quote

BINANCE_SPOT_BASES = (
    "https://data-api.binance.vision",
    "https://api.binance.com",
)
REQUEST_TIMEOUT = 20
KLINE_LIMIT = 1000


class BinanceSpotInvalidSymbolError(ValueError):
    """Binance rejected a spot symbol as unknown."""


def _binance_spot_symbol(symbol: str) -> tuple[str, str]:
    """Map supported USD-style inputs to a Binance spot pair and quote asset."""
    base = crypto_base(symbol)
    if not base:
        raise NoMarketDataError(
            symbol,
            symbol,
            "not a supported USD/USDT/USDC cryptocurrency pair",
        )
    # Binance Spot does not expose a generic BTCUSD market. Use the user's
    # stablecoin quote when explicit; otherwise use the liquid USDT proxy.
    requested_quote = crypto_quote(symbol)
    quote = requested_quote if requested_quote in {"USDT", "USDC"} else "USDT"
    return f"{base}{quote}", quote


def _request_klines(params: dict) -> list:
    """Request public klines, trying Binance's market-data-only host first."""
    errors = []
    rate_limited = False
    for base_url in BINANCE_SPOT_BASES:
        try:
            response = requests.get(
                f"{base_url}/api/v3/klines",
                params=params,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code in {418, 429}:
                rate_limited = True
                errors.append(f"{base_url} returned HTTP {response.status_code}")
                continue
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict):
                code = payload.get("code")
                message = payload.get("msg", "unknown Binance error")
                if code == -1121:
                    raise BinanceSpotInvalidSymbolError(message)
                raise ValueError(f"Binance API error {code}: {message}")
            if not isinstance(payload, list):
                raise ValueError("Binance returned an unexpected kline payload")
            return payload
        except BinanceSpotInvalidSymbolError:
            raise
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{base_url}: {exc}")

    if rate_limited and all("HTTP 4" in error for error in errors):
        raise VendorRateLimitError("; ".join(errors))
    raise requests.RequestException("; ".join(errors) or "Binance Spot request failed")


def _utc_day_ms(value: str, *, end: bool = False) -> int:
    day = datetime.strptime(value, "%Y-%m-%d").date()
    moment = datetime.combine(day, time.min, tzinfo=timezone.utc)
    if end:
        return int((moment + timedelta(days=1)).timestamp() * 1000) - 1
    return int(moment.timestamp() * 1000)


@lru_cache(maxsize=64)
def _fetch_kline_rows(binance_symbol: str, start_date: str, end_date: str) -> tuple:
    """Fetch an inclusive UTC daily range with bounded Binance pagination."""
    start_ms = _utc_day_ms(start_date)
    end_ms = _utc_day_ms(end_date, end=True)
    if start_ms > end_ms:
        raise ValueError("start_date must not be after end_date")

    cursor = start_ms
    rows = []
    while cursor <= end_ms:
        payload = _request_klines(
            {
                "symbol": binance_symbol,
                "interval": "1d",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": KLINE_LIMIT,
            }
        )
        if not payload:
            break
        rows.extend(payload)
        try:
            last_open_ms = int(payload[-1][0])
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("Binance returned a malformed kline row") from exc
        next_cursor = last_open_ms + 24 * 60 * 60 * 1000
        if next_cursor <= cursor:
            raise ValueError("Binance kline pagination did not advance")
        cursor = next_cursor
        if len(payload) < KLINE_LIMIT:
            break
    return tuple(tuple(row) for row in rows)


def get_binance_spot_ohlcv_frame(
    symbol: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Return completed Binance Spot UTC daily candles as a fresh DataFrame."""
    binance_symbol, quote = _binance_spot_symbol(symbol)
    try:
        raw_rows = _fetch_kline_rows(binance_symbol, start_date, end_date)
    except BinanceSpotInvalidSymbolError as exc:
        raise NoMarketDataError(
            symbol, binance_symbol, f"Binance Spot does not list this pair ({exc})"
        ) from exc

    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    requested_start_ms = _utc_day_ms(start_date)
    requested_end_ms = _utc_day_ms(end_date, end=True)
    parsed = []
    for row in raw_rows:
        if len(row) < 9:
            continue
        try:
            open_ms = int(row[0])
            close_ms = int(row[6])
            if not (requested_start_ms <= open_ms <= requested_end_ms):
                continue
            if close_ms > now_ms or close_ms > requested_end_ms:
                continue
            parsed.append(
                {
                    "Date": datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).replace(
                        tzinfo=None
                    ),
                    "Open": row[1],
                    "High": row[2],
                    "Low": row[3],
                    "Close": row[4],
                    "Volume": row[5],
                    "Quote Volume": row[7],
                    "Trades": row[8],
                }
            )
        except (TypeError, ValueError, OSError):
            continue

    if not parsed:
        raise NoMarketDataError(
            symbol,
            binance_symbol,
            f"no completed Binance Spot daily candles between {start_date} and {end_date}",
        )

    data = _clean_dataframe(pd.DataFrame(parsed)).sort_values("Date")
    _assert_ohlcv_not_stale(data, end_date, symbol, binance_symbol)
    _assert_crypto_cutoff_available(data, end_date, symbol, binance_symbol)
    data.attrs["market_data_source"] = (
        f"Binance Spot {binance_symbol} UTC daily candles; Volume is base-asset "
        f"volume and Quote Volume is denominated in {quote}"
    )
    data.attrs["binance_symbol"] = binance_symbol
    data.attrs["quote_asset"] = quote
    return data


def load_binance_spot_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Load enough Binance history for long-window indicators and verification."""
    end_day = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date = (end_day - timedelta(days=5 * 366)).strftime("%Y-%m-%d")
    return get_binance_spot_ohlcv_frame(symbol, start_date, curr_date)


def get_binance_spot_data(
    symbol: Annotated[str, "crypto ticker, e.g. BTC-USD or BTC/USDT"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Return Binance Spot UTC daily OHLCV for the inclusive requested range."""
    data = get_binance_spot_ohlcv_frame(symbol, start_date, end_date)
    source = data.attrs["market_data_source"]
    binance_symbol = data.attrs["binance_symbol"]
    return "\n".join(
        [
            f"# Binance Spot data for {binance_symbol} (requested as {symbol})",
            f"# Inclusive UTC window: {start_date} through {end_date}",
            f"# Source/method: {source}",
            "# BTC-USD-style inputs use BTCUSDT as an explicitly disclosed spot proxy; "
            "USDT basis risk means it is not an exact USD market.",
            "",
            data.to_csv(index=False),
        ]
    )


def get_binance_spot_indicators_window(
    symbol: str, indicator: str, curr_date: str, look_back_days: int
) -> str:
    """Calculate stockstats indicators from Binance-only OHLCV."""
    # Import lazily to keep the vendor modules acyclic during interface setup.
    from .y_finance import get_stock_stats_indicators_window

    result = get_stock_stats_indicators_window(
        symbol,
        indicator,
        curr_date,
        look_back_days,
        ohlcv_loader=load_binance_spot_ohlcv,
    )
    binance_symbol, quote = _binance_spot_symbol(symbol)
    return (
        f"# Indicator source: Binance Spot {binance_symbol} UTC daily candles; "
        f"Volume is base-asset volume (quote asset {quote}).\n\n{result}"
    )
