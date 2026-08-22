from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_crypto_derivatives(
    symbol: Annotated[str, "crypto ticker, e.g. BTC-USD or ETH-USD"],
    curr_date: Annotated[str, "analysis date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "trailing daily observations, max 30"] = 7,
    analysis_as_of: Annotated[str, "ISO-8601 live-information cutoff"] = "",
) -> str:
    """Retrieve Binance USD-M futures funding, open interest, long/short
    positioning, and taker-flow history for a cryptocurrency. The result is
    capped at the analysis date and uses the configured crypto_derivatives
    vendor. Use only for crypto assets; it is supplemental to verified OHLCV.
    """
    return route_to_vendor(
        "get_crypto_derivatives", symbol, curr_date, look_back_days, analysis_as_of
    )


@tool
def get_crypto_options(
    symbol: Annotated[str, "crypto ticker, e.g. BTC-USD or ETH-USD"],
    analysis_as_of: Annotated[str, "ISO-8601 live-information cutoff"] = "",
) -> str:
    """Retrieve a keyless Deribit options snapshot: ATM IV term structure,
    put/call open interest and volume, and strike/expiry OI concentrations.
    Unsupported 25-delta skew/GEX fields are reported unavailable, never guessed.
    """
    return route_to_vendor("get_crypto_options", symbol, analysis_as_of)


@tool
def get_crypto_onchain(
    symbol: Annotated[str, "crypto ticker, e.g. BTC-USD or ETH-USD"],
    start_date: Annotated[str, "start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "end date in yyyy-mm-dd format"],
) -> str:
    """Retrieve free-first Coin Metrics network/stablecoin metrics, completed
    daily cross-market reported spot-volume activity, and optional user-defined
    Dune on-chain or ETF-flow results. Aggregate activity is auxiliary and must
    never replace exchange OHLCV volume in technical indicators. Missing premium
    exchange-flow/whale data is explicitly reported unavailable.
    """
    return route_to_vendor("get_crypto_onchain", symbol, start_date, end_date)


@tool
def get_crypto_liquidations(
    symbol: Annotated[str, "crypto ticker, e.g. BTC-USD or ETH-USD"],
    start_time: Annotated[str, "ISO-8601 UTC window start"],
    end_time: Annotated[str, "ISO-8601 UTC window end"],
    interval: Annotated[str, "Coinalyze interval such as 1hour"] = "1hour",
) -> str:
    """Retrieve actual long/short liquidation history from optional Coinalyze.
    This is observed liquidation flow, not a predicted liquidation heatmap.
    """
    return route_to_vendor(
        "get_crypto_liquidations", symbol, start_time, end_time, interval
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
