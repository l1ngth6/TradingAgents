"""Market-calendar helpers with an explicit UTC day boundary for crypto.

Cryptocurrency daily candles roll over at 00:00 UTC.  Deriving their current
date from the host's local timezone makes the selected candle depend on where
TradingAgents happens to run (for example, Beijing enters a new local date
eight hours before the UTC candle rolls over).

Other asset types intentionally retain the existing host-local date behaviour
until exchange-specific calendars/timezones are introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone

from tradingagents.dataflows.symbol_utils import crypto_base

CRYPTO_MARKET_TIMEZONE = timezone.utc


@dataclass(frozen=True)
class AnalysisCutoffs:
    """The live-information and completed-candle cutoffs for one task."""

    analysis_as_of: str
    completed_daily_candle_date: str
    is_live_analysis: bool


def _asset_type_value(asset_type: object) -> str:
    """Return a normalized asset-type value, including for string enums."""
    return str(getattr(asset_type, "value", asset_type)).strip().lower()


def uses_utc_market_day(asset_type: object) -> bool:
    """Whether ``asset_type`` uses the 00:00 UTC daily-candle boundary."""
    return _asset_type_value(asset_type) == "crypto"


def current_market_date(
    asset_type: object = "stock", *, now: datetime | None = None
) -> date:
    """Return today's date in the asset's market calendar.

    Crypto is always evaluated in UTC.  Non-crypto assets preserve the
    project's prior host-local behaviour.  ``now`` is an injection point for
    deterministic boundary tests; for crypto it must be timezone-aware so an
    absolute instant cannot be mistaken for local wall-clock time.
    """
    if uses_utc_market_day(asset_type):
        instant = now if now is not None else datetime.now(CRYPTO_MARKET_TIMEZONE)
        if instant.tzinfo is None:
            raise ValueError("now must be timezone-aware for a crypto market date")
        return instant.astimezone(CRYPTO_MARKET_TIMEZONE).date()

    return now.date() if now is not None else date.today()


def validate_analysis_date(
    value: object,
    asset_type: object = "stock",
    *,
    market_today: date | None = None,
) -> date:
    """Parse an analysis date and reject dates beyond the market's today."""
    try:
        requested = datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("analysis date must use YYYY-MM-DD format") from exc

    latest = market_today or current_market_date(asset_type)
    if requested > latest:
        boundary = "UTC market date" if uses_utc_market_day(asset_type) else "current date"
        raise ValueError(
            f"analysis date {requested} is later than the {boundary} {latest}"
        )
    return requested


def analysis_cutoffs(
    analysis_date: object,
    asset_type: object = "stock",
    *,
    now: datetime | None = None,
) -> AnalysisCutoffs:
    """Resolve the two time boundaries used by an analysis.

    Crypto may analyse the latest UTC date and use live information observed at
    task start, but every daily-candle calculation is capped at the most recent
    fully closed 00:00-00:00 UTC candle. Historical dates keep their requested
    date as the completed-candle cutoff. Non-crypto markets retain the existing
    date semantics until exchange-specific calendars are available.
    """
    if uses_utc_market_day(asset_type):
        instant = now or datetime.now(CRYPTO_MARKET_TIMEZONE)
        if instant.tzinfo is None:
            raise ValueError("now must be timezone-aware for crypto analysis cutoffs")
        instant = instant.astimezone(CRYPTO_MARKET_TIMEZONE)
        today = instant.date()
        requested = validate_analysis_date(
            analysis_date, asset_type, market_today=today
        )
        live = requested == today
        as_of = instant if live else datetime.combine(requested, time.max, CRYPTO_MARKET_TIMEZONE)
        completed = requested - timedelta(days=1) if live else requested
        return AnalysisCutoffs(
            analysis_as_of=as_of.isoformat(timespec="seconds"),
            completed_daily_candle_date=completed.isoformat(),
            is_live_analysis=live,
        )

    local_now = now or datetime.now().astimezone()
    requested = validate_analysis_date(
        analysis_date, asset_type, market_today=local_now.date()
    )
    live = requested == local_now.date()
    as_of = local_now if live else datetime.combine(requested, time.max).astimezone()
    return AnalysisCutoffs(
        analysis_as_of=as_of.isoformat(timespec="seconds"),
        completed_daily_candle_date=requested.isoformat(),
        is_live_analysis=live,
    )


def completed_daily_cutoff_for_symbol(value: str, symbol: str) -> str:
    """Defensively clamp a crypto daily-data request to a closed UTC candle.

    This lives in the data path as a backstop for malformed model tool calls:
    even if an agent supplies today's date, current crypto OHLCV cannot become
    an input to SMA/EMA/RSI/MACD/Bollinger/ATR or candle-pattern analysis.
    """
    normalized = str(symbol).strip().upper()
    is_crypto = bool(crypto_base(normalized))
    if not is_crypto:
        return str(value)
    requested = validate_analysis_date(value, "crypto")
    today = current_market_date("crypto")
    if requested >= today:
        return (today - timedelta(days=1)).isoformat()
    return requested.isoformat()


def market_timestamp(asset_type: object = "stock") -> str:
    """Format a retrieval timestamp, explicitly labeling crypto timestamps."""
    if uses_utc_market_day(asset_type):
        return datetime.now(CRYPTO_MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S UTC")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
