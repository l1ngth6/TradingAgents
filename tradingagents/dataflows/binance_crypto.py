"""Binance USD-M futures enrichment for cryptocurrency analysis.

This module intentionally complements, rather than replaces, the canonical
OHLCV/indicator path.  Yahoo Finance remains the verified price source while
Binance contributes crypto-native derivatives signals that equities do not
have: funding, open interest, account positioning, and taker flow.

All requests are public and keyless.  ``endTime`` is anchored to the requested
analysis date and every returned row is filtered again locally, preventing a
historical run from leaking today's positioning data into the report.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

import requests

from .errors import NoMarketDataError, VendorRateLimitError
from .symbol_utils import crypto_base

BINANCE_FUTURES_BASE = "https://fapi.binance.com"
REQUEST_TIMEOUT = 20
MAX_POSITIONING_DAYS = 30


class BinanceInvalidSymbolError(ValueError):
    """Binance rejected a symbol as unknown."""


def _request(path: str, params: dict) -> list | dict:
    response = requests.get(
        f"{BINANCE_FUTURES_BASE}{path}",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code in {418, 429}:
        raise VendorRateLimitError(
            f"Binance rate limit returned HTTP {response.status_code}."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        response.raise_for_status()
        raise ValueError("Binance returned a non-JSON response.") from exc

    if isinstance(payload, dict) and isinstance(payload.get("code"), int):
        code = payload["code"]
        message = payload.get("msg", "unknown Binance error")
        if code == -1121:
            raise BinanceInvalidSymbolError(message)
        if code < 0:
            raise ValueError(f"Binance API error {code}: {message}")

    response.raise_for_status()
    return payload


def _binance_symbol(symbol: str) -> str:
    """Map the pipeline's Yahoo-style crypto pair to Binance's USDT future."""
    base = crypto_base(symbol)
    if not base:
        raise NoMarketDataError(
            symbol,
            symbol,
            "not a supported USD/USDT/USDC cryptocurrency pair",
        )
    return f"{base}USDT"


def _analysis_end_ms(curr_date: str, analysis_as_of: str = "") -> tuple[int, str]:
    requested = datetime.strptime(curr_date, "%Y-%m-%d").date()
    requested_end = datetime.combine(requested, time.max, tzinfo=timezone.utc)
    supplied_as_of = datetime.now(timezone.utc)
    if analysis_as_of:
        supplied_as_of = datetime.fromisoformat(analysis_as_of.replace("Z", "+00:00"))
        if supplied_as_of.tzinfo is None:
            supplied_as_of = supplied_as_of.replace(tzinfo=timezone.utc)
        supplied_as_of = supplied_as_of.astimezone(timezone.utc)
    actual_end = min(requested_end, supplied_as_of, datetime.now(timezone.utc))
    return int(actual_end.timestamp() * 1000), actual_end.strftime("%Y-%m-%d %H:%M UTC")


def _rows_at_or_before(rows: list, end_ms: int, timestamp_key: str) -> list[dict]:
    usable = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            timestamp = int(row[timestamp_key])
        except (KeyError, TypeError, ValueError):
            continue
        if timestamp <= end_ms:
            usable.append(row)
    return sorted(usable, key=lambda row: int(row[timestamp_key]))


def _utc(timestamp_ms) -> str:
    return datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M"
    )


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _funding_table(rows: list[dict]) -> str:
    lines = ["### Funding-rate history", "", "| Time (UTC) | Funding rate |", "|---|---:|"]
    for row in rows:
        rate = _float(row.get("fundingRate"))
        if rate is not None:
            lines.append(f"| {_utc(row['fundingTime'])} | {rate * 100:+.4f}% |")
    return "\n".join(lines)


def _positioning_table(rows: list[dict]) -> str:
    lines = [
        "### Global long/short account ratio",
        "",
        "| Time (UTC) | Long accounts | Short accounts | L/S ratio |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        long_account = _float(row.get("longAccount"))
        short_account = _float(row.get("shortAccount"))
        ratio = _float(row.get("longShortRatio"))
        if None not in (long_account, short_account, ratio):
            lines.append(
                f"| {_utc(row['timestamp'])} | {long_account * 100:.2f}% | "
                f"{short_account * 100:.2f}% | {ratio:.3f} |"
            )
    return "\n".join(lines)


def _open_interest_table(rows: list[dict]) -> str:
    lines = [
        "### Open interest",
        "",
        "| Time (UTC) | Contracts | Notional (USDT) |",
        "|---|---:|---:|",
    ]
    for row in rows:
        contracts = _float(row.get("sumOpenInterest"))
        notional = _float(row.get("sumOpenInterestValue"))
        if contracts is not None and notional is not None:
            lines.append(
                f"| {_utc(row['timestamp'])} | {contracts:,.4f} | {notional:,.2f} |"
            )
    return "\n".join(lines)


def _taker_table(rows: list[dict]) -> str:
    lines = [
        "### Taker buy/sell flow and derived CVD proxy",
        "",
        "| Time (UTC) | Buy volume | Sell volume | Buy/sell ratio | Delta | Cumulative delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    cumulative_delta = 0.0
    for row in rows:
        buy = _float(row.get("buyVol"))
        sell = _float(row.get("sellVol"))
        ratio = _float(row.get("buySellRatio"))
        if None not in (buy, sell, ratio):
            delta = buy - sell
            cumulative_delta += delta
            lines.append(
                f"| {_utc(row['timestamp'])} | {buy:,.4f} | {sell:,.4f} | {ratio:.3f} |"
                f" {delta:+,.4f} | {cumulative_delta:+,.4f} |"
            )
    lines.extend([
        "",
        "Cumulative delta is a derived proxy from Binance's aggregated taker buy/sell "
        "volume for this window, not tick-level spot CVD or a complete cross-exchange order flow.",
    ])
    return "\n".join(lines)


def _order_book_table(payload: dict) -> str:
    bids = payload.get("bids", []) if isinstance(payload, dict) else []
    asks = payload.get("asks", []) if isinstance(payload, dict) else []
    parsed_bids = [
        (_float(row[0]), _float(row[1]))
        for row in bids
        if isinstance(row, (list, tuple)) and len(row) >= 2
    ]
    parsed_asks = [
        (_float(row[0]), _float(row[1]))
        for row in asks
        if isinstance(row, (list, tuple)) and len(row) >= 2
    ]
    parsed_bids = [(price, qty) for price, qty in parsed_bids if None not in (price, qty)]
    parsed_asks = [(price, qty) for price, qty in parsed_asks if None not in (price, qty)]
    if not parsed_bids or not parsed_asks:
        return "### Live futures order book\n\nDATA_UNAVAILABLE: empty depth snapshot."
    best_bid = max(price for price, _ in parsed_bids)
    best_ask = min(price for price, _ in parsed_asks)
    bid_notional = sum(price * qty for price, qty in parsed_bids)
    ask_notional = sum(price * qty for price, qty in parsed_asks)
    total = bid_notional + ask_notional
    imbalance = (bid_notional - ask_notional) / total if total else 0.0
    spread_bps = ((best_ask - best_bid) / ((best_ask + best_bid) / 2)) * 10_000
    return "\n".join([
        "### Live Binance futures order-book snapshot (top 100 levels)",
        "",
        f"- Retrieved at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "- Nature: observed depth snapshot plus derived notional/imbalance",
        "",
        "| Best bid | Best ask | Spread (bps) | Bid notional | Ask notional | Depth imbalance |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {best_bid:,.2f} | {best_ask:,.2f} | {spread_bps:.2f} | {bid_notional:,.2f} | {ask_notional:,.2f} | {imbalance:+.3f} |",
        "",
        "Depth imbalance is a derived point-in-time Binance USD-M futures measure; "
        "it can change quickly and is neither spot demand nor a durable support/resistance level.",
    ])


def get_crypto_derivatives(
    symbol: str,
    curr_date: str,
    look_back_days: int | None = None,
    analysis_as_of: str = "",
) -> str:
    """Return historical-date-safe Binance futures positioning signals.

    Binance retains the positioning-statistics endpoints for a limited recent
    window.  Older analysis dates may therefore contain funding history but no
    open-interest/ratio rows; the report calls that out instead of substituting
    current data.
    """
    binance_symbol = _binance_symbol(symbol)
    end_ms, actual_end = _analysis_end_ms(curr_date, analysis_as_of)
    days = max(1, min(int(look_back_days or 7), MAX_POSITIONING_DAYS))

    requests_to_make = {
        "funding": (
            "/fapi/v1/fundingRate",
            {"symbol": binance_symbol, "endTime": end_ms, "limit": min(days * 3, 1000)},
            "fundingTime",
        ),
        "positioning": (
            "/futures/data/globalLongShortAccountRatio",
            {"symbol": binance_symbol, "period": "1d", "endTime": end_ms, "limit": days},
            "timestamp",
        ),
        "open_interest": (
            "/futures/data/openInterestHist",
            {"symbol": binance_symbol, "period": "1d", "endTime": end_ms, "limit": days},
            "timestamp",
        ),
        "taker": (
            "/futures/data/takerlongshortRatio",
            {"symbol": binance_symbol, "period": "1d", "endTime": end_ms, "limit": days},
            "timestamp",
        ),
    }

    datasets: dict[str, list[dict]] = {}
    errors: dict[str, str] = {}
    rate_limited = False
    invalid_symbol = False
    live_snapshot = None
    live_order_book = None
    requested_day = datetime.strptime(curr_date, "%Y-%m-%d").date()
    if requested_day == datetime.now(timezone.utc).date():
        try:
            snapshot = _request("/fapi/v1/premiumIndex", {"symbol": binance_symbol})
            if isinstance(snapshot, dict):
                snapshot_time = int(snapshot.get("time") or 0)
                if snapshot_time:
                    live_snapshot = snapshot
        except Exception as exc:  # optional within an already optional source
            errors["live_snapshot"] = str(exc)
        try:
            depth = _request("/fapi/v1/depth", {"symbol": binance_symbol, "limit": 100})
            if isinstance(depth, dict):
                live_order_book = depth
        except Exception as exc:
            errors["live_order_book"] = str(exc)
    for name, (path, params, timestamp_key) in requests_to_make.items():
        try:
            payload = _request(path, params)
            datasets[name] = _rows_at_or_before(payload, end_ms, timestamp_key)
        except VendorRateLimitError as exc:
            rate_limited = True
            datasets[name] = []
            errors[name] = str(exc)
        except BinanceInvalidSymbolError as exc:
            invalid_symbol = True
            datasets[name] = []
            errors[name] = str(exc)
        except (requests.RequestException, ValueError) as exc:
            datasets[name] = []
            errors[name] = str(exc)

    if not any(datasets.values()):
        if invalid_symbol:
            raise NoMarketDataError(symbol, binance_symbol, "Binance does not list this future")
        if rate_limited:
            raise VendorRateLimitError("Binance rate-limited every derivatives request.")
        detail = next(iter(errors.values()), "no derivatives rows returned")
        raise NoMarketDataError(symbol, binance_symbol, detail)

    sections = []
    renderers = {
        "funding": _funding_table,
        "positioning": _positioning_table,
        "open_interest": _open_interest_table,
        "taker": _taker_table,
    }
    labels = {
        "funding": "funding rate",
        "positioning": "global long/short account ratio",
        "open_interest": "open interest",
        "taker": "taker buy/sell flow",
    }
    for name, renderer in renderers.items():
        rows = datasets.get(name) or []
        if rows:
            sections.append(renderer(rows))
        else:
            reason = errors.get(name) or "no rows in Binance's retained historical window"
            sections.append(f"### {labels[name].title()}\n\nDATA_UNAVAILABLE: {reason}.")

    header = (
        f"## Binance USD-M futures context for {binance_symbol}\n\n"
        f"- Requested analysis date: {curr_date}\n"
        f"- Data cutoff: {actual_end}\n"
        f"- Window: up to {days} days (funding may contain up to three observations per day)\n"
        "- Nature: observed public futures data and derived table formatting.\n"
        "- These are derivatives-market positioning signals, not spot holdings or "
        "a replacement for verified OHLCV data. Crowded positioning can be contrarian.\n\n"
    )
    if live_snapshot:
        mark_price = _float(live_snapshot.get("markPrice"))
        index_price = _float(live_snapshot.get("indexPrice"))
        last_funding = _float(live_snapshot.get("lastFundingRate"))
        live_lines = [
            "### Live futures snapshot",
            "",
            "| Observed at (UTC) | Mark price | Index price | Last funding rate |",
            "|---|---:|---:|---:|",
            f"| {_utc(live_snapshot['time'])} | {mark_price if mark_price is not None else 'N/A'} | "
            f"{index_price if index_price is not None else 'N/A'} | "
            f"{f'{last_funding * 100:+.4f}%' if last_funding is not None else 'N/A'} |",
            "",
            "This is a live/point-in-time futures quote, not a completed daily close.",
            "Its own observed-at timestamp is authoritative and may be slightly later than the task-start cutoff because the API call occurs during analysis.",
        ]
        sections.insert(0, "\n".join(live_lines))
    if live_order_book:
        sections.insert(1 if live_snapshot else 0, _order_book_table(live_order_book))
    return header + "\n\n".join(sections)
