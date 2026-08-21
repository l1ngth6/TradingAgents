from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_crypto_derivatives(
    symbol: Annotated[str, "crypto ticker, e.g. BTC-USD or ETH-USD"],
    curr_date: Annotated[str, "analysis date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "trailing daily observations, max 30"] = 7,
) -> str:
    """Retrieve Binance USD-M futures funding, open interest, long/short
    positioning, and taker-flow history for a cryptocurrency. The result is
    capped at the analysis date and uses the configured crypto_derivatives
    vendor. Use only for crypto assets; it is supplemental to verified OHLCV.
    """
    return route_to_vendor(
        "get_crypto_derivatives", symbol, curr_date, look_back_days
    )


@tool
def get_crypto_fear_greed(
    symbol: Annotated[str, "crypto ticker, e.g. BTC-USD or ETH-USD"],
    curr_date: Annotated[str, "analysis date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "trailing daily observations, max 30"] = 7,
) -> str:
    """Retrieve Alternative.me's historical Crypto Fear & Greed Index on or
    before the analysis date. The index is Bitcoin-centric and is a broad
    market-sentiment proxy rather than coin-specific sentiment.
    """
    return route_to_vendor(
        "get_crypto_fear_greed", symbol, curr_date, look_back_days
    )
