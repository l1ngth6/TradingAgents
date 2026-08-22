"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class PositionAction(str, Enum):
    """Position-aware action kept separate from the directional market rating."""

    OPEN = "Open"
    ADD = "Add"
    MAINTAIN = "Maintain"
    REDUCE = "Reduce"
    EXIT = "Exit"
    AVOID = "Avoid"
    CONDITIONAL = "Conditional"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "A position-independent market plan for the trader: entry conditions, "
            "invalidation levels, and execution considerations consistent with the "
            "rating. Do not assume a current holding or prescribe a fraction of an "
            "unknown portfolio; the Trader will apply portfolio context later."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional unconditional hard protective stop in quote currency. Do not "
            "use this field for a close-confirmed or volume-confirmed risk trigger."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description=(
            "Optional sizing guidance. If no portfolio context is supplied, make it "
            "conditional instead of inventing an existing position or sale fraction."
        ),
    )
    risk_trigger: str | None = Field(
        default=None,
        description="Optional conditional risk trigger, separate from the hard stop.",
    )
    action_on_trigger: str | None = Field(
        default=None,
        description="Action to take only if risk_trigger occurs; state assumptions.",
    )
    confirmation_interval: str | None = Field(
        default=None,
        description="Confirmation interval for risk_trigger, e.g. 1h or 4h close.",
    )
    target_exposure: str | None = Field(
        default=None,
        description="Conditional target exposure after execution, when context supports it.",
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    if proposal.target_exposure:
        parts.extend(["", f"**Target Exposure**: {proposal.target_exposure}"])
    if proposal.risk_trigger:
        parts.extend(["", f"**Risk Trigger**: {proposal.risk_trigger}"])
    if proposal.confirmation_interval:
        parts.extend(["", f"**Confirmation Interval**: {proposal.confirmation_interval}"])
    if proposal.action_on_trigger:
        parts.extend(["", f"**Action on Trigger**: {proposal.action_on_trigger}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final market stance. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell. It must not silently encode whether the user is "
            "currently flat or holding."
        ),
    )
    position_action: PositionAction = Field(
        description=(
            "The action for the supplied position context: Open / Add / Maintain / "
            "Reduce / Exit / Avoid. Use Conditional when the position is unknown."
        ),
    )
    action_if_flat: str | None = Field(
        default=None,
        description=(
            "Required only when current position is unknown: concrete action for a "
            "user who is flat, including entry conditions or a decision to avoid."
        ),
    )
    action_if_holding: str | None = Field(
        default=None,
        description=(
            "Required only when current position is unknown: concrete action for a "
            "user already holding, without inventing cost basis or position size. "
            "Distinguish long and short holdings if the direction is also unknown "
            "and would materially change the action."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Must match the task-selected decision horizon; do not choose a new horizon.",
    )
    current_position_assumption: str | None = Field(
        default=None,
        description="State the supplied current position or explicitly say it is unknown.",
    )
    target_allocation: str | None = Field(
        default=None,
        description="Conditional target allocation; do not invent account values.",
    )
    hard_stop: float | None = Field(
        default=None,
        description="Optional unconditional hard protective stop, distinct from risk_trigger.",
    )
    risk_trigger: str | None = Field(
        default=None,
        description="Conditional risk threshold based only on completed bars or explicit intraday rules.",
    )
    action_on_trigger: str | None = Field(
        default=None,
        description="Concrete action if the conditional risk trigger fires.",
    )
    confirmation_interval: str | None = Field(
        default=None,
        description="Time interval required to confirm the conditional trigger.",
    )

    @field_validator("price_target", "hard_stop", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_pm_decision(decision: PortfolioDecision, forced_horizon: str | None = None) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Position Action**: {decision.position_action.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.current_position_assumption:
        parts.extend(["", f"**Current Position Assumption**: {decision.current_position_assumption}"])
    if decision.action_if_flat:
        parts.extend(["", f"**Action if Flat**: {decision.action_if_flat}"])
    if decision.action_if_holding:
        parts.extend(["", f"**Action if Holding**: {decision.action_if_holding}"])
    if decision.target_allocation:
        parts.extend(["", f"**Target Allocation**: {decision.target_allocation}"])
    if decision.hard_stop is not None:
        parts.extend(["", f"**Hard Stop**: {decision.hard_stop}"])
    if decision.risk_trigger:
        parts.extend(["", f"**Risk Trigger**: {decision.risk_trigger}"])
    if decision.confirmation_interval:
        parts.extend(["", f"**Confirmation Interval**: {decision.confirmation_interval}"])
    if decision.action_on_trigger:
        parts.extend(["", f"**Action on Trigger**: {decision.action_on_trigger}"])
    horizon = forced_horizon or decision.time_horizon
    if horizon:
        parts.extend(["", f"**Time Horizon**: {horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
