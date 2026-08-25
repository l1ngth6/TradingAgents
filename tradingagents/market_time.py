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
    completed_4h_candle_end: str
    completed_1h_candle_end: str
    is_live_analysis: bool


def _floor_utc_hours(instant: datetime, hours: int) -> datetime:
    """Return the UTC boundary at which the last complete interval ended."""
    instant = instant.astimezone(CRYPTO_MARKET_TIMEZONE)
    return instant.replace(
        hour=(instant.hour // hours) * hours,
        minute=0,
        second=0,
        microsecond=0,
    )


def crypto_intraday_cutoffs(
    analysis_as_of: str | datetime,
    *,
    historical_day_complete: bool | None = None,
    now: datetime | None = None,
) -> dict[str, str]:
    """Resolve exclusive end boundaries for completed UTC 1h and 4h candles.

    Live tasks floor the frozen ``analysis_as_of`` timestamp. Historical tasks
    whose cutoff represents end-of-day include every bar of that requested UTC
    day by using the following midnight as the exclusive boundary. A future
    timestamp is clamped to ``now`` so malformed model tool arguments cannot
    introduce look-ahead data.
    """
    if isinstance(analysis_as_of, datetime):
        instant = analysis_as_of
    else:
        try:
            instant = datetime.fromisoformat(str(analysis_as_of).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("analysis_as_of must be an ISO-8601 timestamp") from exc
    if instant.tzinfo is None:
        raise ValueError("analysis_as_of must include a timezone")
    instant = instant.astimezone(CRYPTO_MARKET_TIMEZONE)

    current = now or datetime.now(CRYPTO_MARKET_TIMEZONE)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    current = current.astimezone(CRYPTO_MARKET_TIMEZONE)
    if instant > current:
        instant = current

    if historical_day_complete is None:
        historical_day_complete = (
            instant.date() < current.date()
            and instant.hour == 23
            and instant.minute == 59
            and instant.second == 59
        )
    reference = (
        datetime.combine(
            instant.date() + timedelta(days=1), time.min, CRYPTO_MARKET_TIMEZONE
        )
        if historical_day_complete
        else instant
    )
    return {
        "1h": _floor_utc_hours(reference, 1).isoformat(timespec="seconds"),
        "4h": _floor_utc_hours(reference, 4).isoformat(timespec="seconds"),
    }


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
    """Resolve the information and completed-candle boundaries for an analysis.

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
        as_of = (
            instant
            if live
            else datetime.combine(requested, time.max, CRYPTO_MARKET_TIMEZONE)
        )
        completed = requested - timedelta(days=1) if live else requested
        intraday = crypto_intraday_cutoffs(
            as_of,
            historical_day_complete=not live,
            now=instant,
        )
        return AnalysisCutoffs(
            analysis_as_of=as_of.isoformat(timespec="seconds"),
            completed_daily_candle_date=completed.isoformat(),
            completed_4h_candle_end=intraday["4h"],
            completed_1h_candle_end=intraday["1h"],
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
        completed_4h_candle_end="",
        completed_1h_candle_end="",
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
