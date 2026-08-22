"""Market-calendar helpers with an explicit UTC day boundary for crypto.

Cryptocurrency daily candles roll over at 00:00 UTC.  Deriving their current
date from the host's local timezone makes the selected candle depend on where
TradingAgents happens to run (for example, Beijing enters a new local date
eight hours before the UTC candle rolls over).

Other asset types intentionally retain the existing host-local date behaviour
until exchange-specific calendars/timezones are introduced.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

CRYPTO_MARKET_TIMEZONE = timezone.utc


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


def market_timestamp(asset_type: object = "stock") -> str:
    """Format a retrieval timestamp, explicitly labeling crypto timestamps."""
    if uses_utc_market_day(asset_type):
        return datetime.now(CRYPTO_MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S UTC")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
