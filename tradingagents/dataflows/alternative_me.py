"""Alternative.me Crypto Fear & Greed Index vendor."""

from __future__ import annotations

from datetime import datetime, timezone

import requests

from .errors import NoMarketDataError, VendorRateLimitError
from .symbol_utils import crypto_base

FNG_URL = "https://api.alternative.me/fng/"
REQUEST_TIMEOUT = 20
MAX_HISTORY_ROWS = 4000


def _request(limit: int) -> dict:
    response = requests.get(
        FNG_URL,
        params={"limit": limit, "format": "json"},
        timeout=REQUEST_TIMEOUT,
    )
    if response.status_code == 429:
        raise VendorRateLimitError("Alternative.me rate limit returned HTTP 429.")
    response.raise_for_status()
    payload = response.json()
    error = (payload.get("metadata") or {}).get("error")
    if error:
        raise ValueError(f"Alternative.me API error: {error}")
    return payload


def get_crypto_fear_greed(
    symbol: str,
    curr_date: str,
    look_back_days: int | None = None,
) -> str:
    """Return Fear & Greed observations on or before ``curr_date``.

    The public API returns newest-first and has no date-range parameter, so the
    requested row count includes the distance from today to the historical
    analysis date.  Local filtering is still applied to guarantee no lookahead.
    """
    if not crypto_base(symbol):
        raise NoMarketDataError(symbol, symbol, "Fear & Greed is only used for crypto")

    end_date = datetime.strptime(curr_date, "%Y-%m-%d").date()
    today = datetime.now(timezone.utc).date()
    days = max(1, min(int(look_back_days or 7), 30))
    distance = max(0, (today - end_date).days)
    fetch_limit = min(MAX_HISTORY_ROWS, distance + days + 3)
    payload = _request(fetch_limit)

    observations = []
    for row in payload.get("data") or []:
        try:
            observed = datetime.fromtimestamp(int(row["timestamp"]), tz=timezone.utc)
            value = int(row["value"])
        except (KeyError, TypeError, ValueError, OSError):
            continue
        if observed.date() <= end_date and 0 <= value <= 100:
            observations.append(
                (observed, value, str(row.get("value_classification") or "Unknown"))
            )

    observations.sort(key=lambda item: item[0])
    observations = observations[-days:]
    if not observations:
        raise NoMarketDataError(
            symbol,
            symbol,
            f"no Fear & Greed observation on or before {curr_date}",
        )

    latest_date, latest_value, latest_classification = observations[-1]
    lines = [
        "## Crypto Fear & Greed Index — Alternative.me",
        "",
        f"- Requested analysis date: {curr_date}",
        f"- Latest observation used: {latest_date.strftime('%Y-%m-%d')}",
        f"- Latest value: **{latest_value}/100 ({latest_classification})**",
        "- This public index is Bitcoin-centric and should be treated as a broad "
        "crypto-market sentiment proxy, not coin-specific sentiment.",
        "- Source/attribution: https://alternative.me/crypto/fear-and-greed-index/",
        "",
        "| Date (UTC) | Value | Classification |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| {observed.strftime('%Y-%m-%d')} | {value} | {classification} |"
        for observed, value, classification in observations
    )
    return "\n".join(lines)
