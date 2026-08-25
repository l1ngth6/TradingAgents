"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.binance_spot import (
    get_binance_spot_intraday_frame,
    get_binance_spot_last_price,
    load_binance_spot_ohlcv,
)
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.errors import NoMarketDataError
from tradingagents.dataflows.stockstats_utils import (
    _assert_crypto_cutoff_available,
    load_ohlcv,
)
from tradingagents.dataflows.symbol_utils import crypto_base
from tradingagents.market_time import (
    completed_daily_cutoff_for_symbol,
    crypto_intraday_cutoffs,
)

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)

INTRADAY_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "macd", "macds", "macdh", "atr", "boll", "boll_ub", "boll_lb",
)

INTRADAY_HORIZON_POLICY = {
    "weekly": {
        "intervals": (("4h", 300), ("1h", 240)),
        "authority": (
            "Daily candles define the broader regime; completed 4h candles are the "
            "primary tactical timeframe; completed 1h candles are execution/risk timing only."
        ),
    },
    "monthly": {
        "intervals": (("4h", 400), ("1h", 240)),
        "authority": (
            "Daily candles remain primary; completed 4h candles may adjust confidence, "
            "entry, stops, and risk triggers; completed 1h candles are protective/execution "
            "context and cannot independently reverse the monthly stance."
        ),
    },
    "strategic": {
        "intervals": (("4h", 500),),
        "authority": (
            "Daily/weekly structure remains primary; completed 4h candles only flag tactical "
            "stress or improve execution and cannot independently reverse the strategic stance."
        ),
    },
}


def _verified_rows(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV on or before curr_date, date-sorted. Raises if nothing usable.

    ``load_ohlcv`` already normalizes the Date column and filters out
    look-ahead rows, but we re-apply the cutoff defensively — this is a
    verification path, so it must not trust its input to be pre-filtered.
    """
    curr_date = completed_daily_cutoff_for_symbol(curr_date, symbol)
    try:
        data = load_ohlcv(symbol, curr_date)
    except NoMarketDataError:
        vendor_setting = str(
            get_config().get("data_vendors", {}).get(
                "technical_indicators", "yfinance"
            )
        )
        configured = {item.strip() for item in vendor_setting.split(",")}
        if not crypto_base(symbol) or (
            "binance" not in configured and "default" not in configured
        ):
            raise
        data = load_binance_spot_ohlcv(symbol, curr_date)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {symbol}.")

    market_data_source = str(
        data.attrs.get("market_data_source")
        or "Yahoo Finance completed daily OHLCV"
    )
    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(curr_date)].sort_values("Date")
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {curr_date} for {symbol}.")
    # ``load_ohlcv`` enforces this too, but the verification path rechecks its
    # input defensively. A completed crypto candle may not fall back to an older
    # row the way an equity cutoff can fall on a weekend or exchange holiday.
    _assert_crypto_cutoff_available(df, curr_date, symbol)
    df.attrs["market_data_source"] = market_data_source
    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_verified_market_snapshot(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes."""
    curr_date = completed_daily_cutoff_for_symbol(curr_date, symbol)
    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    df = _verified_rows(symbol, curr_date)
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        try:
            stock_df[name]  # triggers stockstats calculation
            indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    source = str(df.attrs["market_data_source"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested completed-candle cutoff: {curr_date}",
        f"- Latest trading row used: {latest_date}",
        f"- Market-data source: {source}",
        "- This is completed daily-candle data. Rows after this closed-candle cutoff are excluded.",
        "- Verification method: deterministic calculation independent of the LLM; "
        "this is not an independent market-data vendor.",
        "",
        "### Latest verified OHLCV row",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        lines.append(f"| {field} | {_fmt(latest.get(field))} |")

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices.",
    ]
    return "\n".join(lines)


def _fmt_intraday_timestamp(value) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return "N/A"
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def _safe_pct_change(current, previous) -> str:
    try:
        current_value = float(current)
        previous_value = float(previous)
        if previous_value == 0:
            return "N/A"
        return f"{((current_value / previous_value) - 1) * 100:+.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def _render_intraday_interval(frame: pd.DataFrame, interval: str) -> list[str]:
    """Render compact deterministic OHLCV, indicators, and recent closed bars."""
    stock_df = wrap(frame.copy())
    indicators = {}
    for name in INTRADAY_SNAPSHOT_INDICATORS:
        try:
            stock_df[name]
            indicators[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 - preserve the rest of the snapshot
            indicators[name] = f"N/A ({type(exc).__name__})"

    latest = frame.iloc[-1]
    previous = frame.iloc[-2]
    recent_window = frame.tail(min(len(frame), 24))
    recent_high = pd.to_numeric(recent_window["High"], errors="coerce").max()
    recent_low = pd.to_numeric(recent_window["Low"], errors="coerce").min()
    source = str(frame.attrs.get("market_data_source", "Binance Spot"))

    lines = [
        f"## Completed {interval} technical snapshot",
        "",
        f"- Required exclusive candle-end boundary: {frame.attrs.get('completed_end', 'unknown')}",
        f"- Latest candle: {_fmt_intraday_timestamp(latest.get('Date'))} through "
        f"{_fmt_intraday_timestamp(latest.get('Close Time'))}",
        f"- Source/method: {source}",
        f"- Latest one-bar close change: {_safe_pct_change(latest.get('Close'), previous.get('Close'))}",
        f"- High/low across the latest {len(recent_window)} completed bars: "
        f"{_fmt(recent_high)} / {_fmt(recent_low)}",
        "",
        "### Latest closed OHLCV bar",
        "",
        "| Open | High | Low | Close | Base volume | Quote volume | Trades |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(
            _fmt(latest.get(field))
            for field in (
                "Open", "High", "Low", "Close", "Volume", "Quote Volume", "Trades"
            )
        ) + " |",
        "",
        "### Indicators on completed bars only",
        "",
        "| Indicator | Value |",
        "|---|---:|",
    ]
    for name, value in indicators.items():
        lines.append(f"| {name} | {value} |")

    recent = frame.tail(8)
    lines += [
        "",
        "### Recent completed bars",
        "",
        "| Candle open | Candle end | Open | High | Low | Close | Base volume |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in recent.iterrows():
        lines.append(
            f"| {_fmt_intraday_timestamp(row.get('Date'))} | "
            f"{_fmt_intraday_timestamp(row.get('Close Time'))} | "
            + " | ".join(
                _fmt(row.get(field))
                for field in ("Open", "High", "Low", "Close", "Volume")
            )
            + " |"
        )
    return lines


def build_crypto_intraday_snapshot(
    symbol: str,
    analysis_as_of: str,
    decision_horizon: str = "monthly",
) -> str:
    """Build a multi-timeframe snapshot without using any unfinished candle.

    The current spot price is an explicitly isolated provisional observation on
    live tasks. It is never appended to the OHLCV frames used for indicators.
    """
    if not crypto_base(symbol):
        raise NoMarketDataError(symbol, symbol, "intraday snapshot is crypto-only")
    horizon = str(decision_horizon).strip().lower()
    if horizon not in INTRADAY_HORIZON_POLICY:
        raise ValueError("decision_horizon must be weekly, monthly, or strategic")

    cutoffs = crypto_intraday_cutoffs(analysis_as_of)
    policy = INTRADAY_HORIZON_POLICY[horizon]
    lines = [
        f"# Verified crypto intraday snapshot for {symbol.upper()}",
        "",
        f"- Frozen task information cutoff: {analysis_as_of}",
        f"- Decision horizon: {horizon}",
        f"- Timeframe authority: {policy['authority']}",
        f"- Last completed 4h candle ends at: {cutoffs['4h']}",
        f"- Last completed 1h candle ends at: {cutoffs['1h']}",
        "- Every indicator and candle-pattern claim below uses completed bars only.",
        "- Never merge this Binance intraday volume with Yahoo daily OHLCV or "
        "Coin Metrics cross-market activity.",
    ]

    latest_closed_price = None
    unavailable = []
    for interval, bars in policy["intervals"]:
        try:
            frame = get_binance_spot_intraday_frame(
                symbol,
                interval,
                cutoffs[interval],
                lookback_bars=bars,
            )
            if interval == "1h" or latest_closed_price is None:
                latest_closed_price = float(frame.iloc[-1]["Close"])
            lines += [""] + _render_intraday_interval(frame, interval)
        except Exception as exc:  # noqa: BLE001 - one timeframe may degrade independently
            unavailable.append(f"{interval}: {type(exc).__name__}: {exc}")

    try:
        as_of = datetime.fromisoformat(str(analysis_as_of).replace("Z", "+00:00"))
        if as_of.tzinfo is None:
            raise ValueError("analysis_as_of must include a timezone")
        live = as_of.astimezone(timezone.utc).date() == datetime.now(timezone.utc).date()
    except ValueError:
        live = False

    lines += ["", "## Provisional live state (excluded from all indicators)", ""]
    if live:
        try:
            quote = get_binance_spot_last_price(symbol)
            delta = (
                _safe_pct_change(float(quote["price"]), latest_closed_price)
                if latest_closed_price is not None
                else "N/A"
            )
            lines += [
                f"- Binance Spot {quote['symbol']} last price: {quote['price']} "
                f"{quote['quote_asset']}",
                f"- Observed at: {quote['observed_at']}",
                f"- Change versus the latest completed intraday close: {delta}",
                "- This observation may be slightly later than task initialization. It is "
                "provisional context only—not a close, breakout confirmation, volume "
                "confirmation, candlestick pattern, or input to RSI/MACD/ATR/Bollinger values.",
            ]
        except Exception as exc:  # noqa: BLE001 - live quote is optional
            lines.append(f"- Live spot price unavailable: {type(exc).__name__}: {exc}")
    else:
        lines.append(
            "- Omitted for a historical task; a present-day quote would be look-ahead data."
        )

    if unavailable:
        lines += ["", "## Unavailable intraday sections", ""]
        lines.extend(f"- {item}" for item in unavailable)
    if len(unavailable) == len(policy["intervals"]):
        raise NoMarketDataError(
            symbol,
            symbol,
            "all requested Binance Spot intraday timeframes were unavailable: "
            + "; ".join(unavailable),
        )
    return "\n".join(lines)
