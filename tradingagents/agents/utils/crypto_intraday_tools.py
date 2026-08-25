"""Deterministic completed-intraday technical context for crypto assets."""

from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.market_data_validator import (
    build_crypto_intraday_snapshot,
)
from tradingagents.dataflows.symbol_utils import NoMarketDataError


@tool
def get_crypto_intraday_snapshot(
    symbol: Annotated[str, "crypto ticker, e.g. BTC-USD or BTC/USDT"],
    analysis_as_of: Annotated[
        str, "frozen task cutoff as an ISO-8601 timestamp including timezone"
    ],
    decision_horizon: Annotated[str, "weekly, monthly, or strategic"],
) -> str:
    """Return verified completed 4h/1h candles and fixed intraday indicators.

    Weekly tasks receive 4h tactical plus 1h execution context; monthly tasks
    receive daily-primary 4h confirmation plus 1h protective context; strategic
    tasks receive only a restrained 4h anomaly/execution view. Any current spot
    quote is isolated as provisional and never enters indicator calculations.
    """
    try:
        return build_crypto_intraday_snapshot(
            symbol, analysis_as_of, decision_horizon
        )
    except NoMarketDataError as exc:
        return (
            f"CRYPTO_INTRADAY_DATA_UNAVAILABLE: {exc}. Continue with completed "
            "daily evidence; do not estimate intraday candles or indicators."
        )
    except (ValueError, TypeError) as exc:
        return (
            f"CRYPTO_INTRADAY_INPUT_INVALID: {exc}. Use the exact symbol, "
            "analysis_as_of, and decision_horizon supplied by the task."
        )
