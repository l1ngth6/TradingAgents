import functools
import logging
from collections.abc import Mapping
from typing import Any

import yfinance as yf
from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import get_stock_data
from tradingagents.agents.utils.crypto_data_tools import (
    get_crypto_derivatives,
    get_crypto_fear_greed,
    get_crypto_liquidations,
    get_crypto_onchain,
    get_crypto_options,
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
)
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators
from tradingagents.agents.utils.market_data_validation_tools import get_verified_market_snapshot
from tradingagents.agents.utils.news_data_tools import (
    get_global_news,
    get_insider_transactions,
    get_news,
)
from tradingagents.agents.utils.prediction_markets_tools import get_prediction_markets
from tradingagents.agents.utils.technical_indicators_tools import get_indicators

# Public surface: the data tools are imported here so agents and the graph
# import them from one place, plus the instrument/language helpers defined below.
__all__ = [
    "get_stock_data",
    "get_crypto_derivatives",
    "get_crypto_fear_greed",
    "get_crypto_options",
    "get_crypto_onchain",
    "get_crypto_liquidations",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
    "get_macro_indicators",
    "get_prediction_markets",
    "get_verified_market_snapshot",
    "build_instrument_context",
    "resolve_instrument_identity",
    "get_instrument_context_from_state",
    "get_crypto_advisory_context",
    "get_decision_context_from_state",
    "get_language_instruction",
    "create_msg_delete",
]

logger = logging.getLogger(__name__)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def _clean_identity_value(value: Any) -> str | None:
    """Return a trimmed string, or None for empty / placeholder-ish values."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in {"none", "n/a", "nan", "null"}:
        return None
    return cleaned


@functools.lru_cache(maxsize=256)
def resolve_instrument_identity(ticker: str) -> dict:
    """Resolve deterministic identity metadata (company name, sector, …) for a ticker.

    This exists to stop the pipeline from hallucinating a *different* company
    when a chart pattern suggests a different industry than the real one
    (#814): without a ground-truth name, the market analyst would pattern-match
    the price action to a narrative and invent an identity that then cascaded
    through every downstream agent.

    Best-effort by design: if yfinance is unavailable, rate-limited, or doesn't
    recognise the ticker, we return ``{}`` and the caller falls back to
    ticker-only context rather than failing before analysis starts. Cached so
    the lookup happens at most once per ticker per process.

    The symbol is normalized first (e.g. ``XAUUSD`` -> ``GC=F``) so identity
    resolves for the same instrument the price path actually fetches (#983).
    """
    from tradingagents.dataflows.symbol_utils import normalize_symbol

    try:
        info = yf.Ticker(normalize_symbol(ticker)).info or {}
    except Exception as exc:  # noqa: BLE001 — fail open, never block the run
        logger.debug("Could not resolve instrument identity for %s: %s", ticker, exc)
        return {}

    identity: dict[str, str] = {}
    company_name = _clean_identity_value(info.get("longName")) or _clean_identity_value(
        info.get("shortName")
    )
    if company_name:
        identity["company_name"] = company_name
    for source_key, target_key in (
        ("sector", "sector"),
        ("industry", "industry"),
        ("exchange", "exchange"),
        ("quoteType", "quote_type"),
    ):
        value = _clean_identity_value(info.get(source_key))
        if value:
            identity[target_key] = value
    return identity


def build_instrument_context(
    ticker: str,
    asset_type: str = "stock",
    identity: Mapping[str, str] | None = None,
) -> str:
    """Describe the exact instrument so agents preserve identity and ticker.

    When ``identity`` is provided (resolved deterministically via
    :func:`resolve_instrument_identity`), the company name and business
    classification are injected so agents anchor to the real company rather
    than pattern-matching the price chart to a wrong one (#814).
    """
    is_crypto = asset_type == "crypto"
    instrument_label = "asset" if is_crypto else "instrument"
    context = (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
    )

    details = []
    if identity:
        name = identity.get("company_name") or identity.get("name")
        if name:
            details.append(f"{'Name' if is_crypto else 'Company'}: {name}")
        sector, industry = identity.get("sector"), identity.get("industry")
        if sector and industry:
            details.append(f"Business classification: {sector} / {industry}")
        elif sector:
            details.append(f"Sector: {sector}")
        elif industry:
            details.append(f"Industry: {industry}")
        if identity.get("exchange"):
            details.append(f"Exchange: {identity['exchange']}")

    if details:
        context += (
            f" Resolved identity: {'; '.join(details)}. "
            "Do not substitute a different company or ticker unless a tool "
            "result explicitly disproves this resolved identity."
        )

    if is_crypto:
        context += (
            " Treat it as a crypto asset rather than a company, and do not "
            "assume company fundamentals are available."
        )
    return context


def get_instrument_context_from_state(state: Mapping[str, Any]) -> str:
    """Return the instrument context for the current run.

    Prefers the identity-resolved context computed once at run start and
    stored on the state (see ``TradingAgentsGraph.resolve_instrument_context``).
    Falls back to a ticker-only context — with no network lookup — when the
    state was constructed without it (bare programmatic states, tests), so a
    consumer is never forced to make a yfinance call mid-graph.
    """
    context = state.get("instrument_context")
    if not isinstance(context, str) or not context.strip():
        context = build_instrument_context(
            str(state["company_of_interest"]),
            state.get("asset_type", "stock"),
        )
    return f"{context} {get_decision_context_from_state(state)}".strip()


_HORIZON_LABELS = {
    "weekly": "Weekly (3-7 calendar days)",
    "monthly": "Monthly (2-4 weeks)",
    "strategic": "Strategic (1-3 months)",
}


def get_decision_context_from_state(state: Mapping[str, Any]) -> str:
    """Return invariant horizon and market-time rules shared by every agent."""
    horizon = str(state.get("decision_horizon", "monthly")).strip().lower()
    label = _HORIZON_LABELS.get(horizon, _HORIZON_LABELS["monthly"])
    as_of = state.get("analysis_as_of") or state.get("trade_date", "unknown")
    completed = state.get("completed_daily_candle_date") or state.get("trade_date", "unknown")
    text = (
        f"The selected decision horizon is {label}; keep every thesis, catalyst, "
        f"target, risk trigger, and outcome window within that horizon. Information "
        f"may be observed through {as_of}. Completed daily OHLCV and all candle-based "
        f"technical indicators are capped at {completed}. Never call a live price a "
        "daily close, and never treat an unfinished candle as a confirmed close, "
        "breakout, volume confirmation, or candlestick pattern."
    )
    portfolio = state.get("portfolio_context")
    if isinstance(portfolio, Mapping) and portfolio:
        values = "; ".join(f"{key}={value}" for key, value in portfolio.items() if value not in (None, ""))
        if values:
            text += f" User-supplied portfolio context: {values}."
    else:
        text += (
            " No current position, cost basis, leverage, or risk budget was supplied. "
            "State assumptions explicitly and do not invent an existing position or an "
            "arbitrary fraction to buy or sell."
        )
    return text


def get_crypto_advisory_context(state: Mapping[str, Any], max_chars: int = 2400) -> str:
    """Expose only a restrained crypto cross-check in Advisory mode.

    Shadow reports remain persisted for the user but are deliberately invisible
    to downstream decision agents.
    """
    if state.get("crypto_intelligence_mode") != "advisory":
        return ""
    report = str(state.get("crypto_intelligence_report") or "").strip()
    if not report:
        return ""
    summary = report.split("## Detailed evidence", 1)[0].strip()[:max_chars]
    return (
        "\nAuxiliary crypto-native cross-validation (never a standalone direction "
        "signal and insufficient by itself to change the rating):\n" + summary
    )


def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add a context-anchored placeholder.

        The placeholder must not be a bare ``"Continue"``: some
        OpenAI-compatible providers interpret that literally as the user task
        and produce output about the word "continue" instead of analysing the
        instrument (#888). Anchoring it to the resolved instrument context and
        date keeps the next analyst on-task even if the provider treats the
        placeholder as a standalone request.
        """
        messages = state["messages"]
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        instrument_context = get_instrument_context_from_state(state)
        trade_date = state.get("trade_date", "the requested date")
        placeholder = HumanMessage(
            content=(
                f"Proceed with your assigned analysis for this workflow. "
                f"{instrument_context} The analysis date is {trade_date}."
            )
        )
        return {"messages": removal_operations + [placeholder]}

    return delete_messages
