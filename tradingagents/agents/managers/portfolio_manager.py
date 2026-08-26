"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

from tradingagents.agents.schemas import (
    CryptoPortfolioDecision,
    PortfolioDecision,
    render_crypto_pm_decision,
    render_pm_decision,
)
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_portfolio_context_from_state,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")
    crypto_structured_llm = bind_structured(
        llm, CryptoPortfolioDecision, "Portfolio Manager (crypto)"
    )

    def portfolio_manager_node(state) -> dict:
        is_crypto = state.get("asset_type") == "crypto"
        instrument_context = get_instrument_context_from_state(state)
        portfolio_context = get_portfolio_context_from_state(state)
        portfolio_status = state.get("portfolio_context", {}).get("status", "unknown")

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )
        horizon_labels = {
            "weekly": "Weekly (3-7 calendar days)",
            "monthly": "Monthly (2-4 weeks)",
            "strategic": "Strategic (1-3 months)",
        }
        horizon = horizon_labels.get(
            state.get("decision_horizon", "monthly"),
            horizon_labels["monthly"],
        )

        rating_scale = (
            """**Crypto Market Outlook Scale** (use exactly one):
- **Strong Bullish**: Strongly favorable directional outlook
- **Bullish**: Constructive directional outlook
- **Neutral**: Balanced or directionally inconclusive outlook
- **Bearish**: Cautious directional outlook
- **Strong Bearish**: Strongly unfavorable directional outlook

Use directional crypto-market language only. Do not use Buy, Overweight, Hold,
Underweight, or Sell as the market outlook. In a free-text response, label this
field exactly **Market Outlook**."""
            if is_crypto
            else """**Rating Scale** (use exactly one):
- **Buy**: Strongly favorable market stance
- **Overweight**: Constructive market stance
- **Hold**: Neutral or balanced market stance
- **Underweight**: Cautious market stance
- **Sell**: Strongly unfavorable market stance"""
        )
        position_guidance = {
            "unknown": (
                "The current position is unknown. Use Conditional as the position "
                "action, do not issue an unconditional buy/sell/add/reduce instruction, "
                "and populate both action_if_flat and action_if_holding as separate "
                "scenario perspectives."
            ),
            "flat": (
                "The user explicitly reported no current position. Use only Open or "
                "Avoid as the position action, and leave action_if_flat and "
                "action_if_holding empty."
            ),
            "holding": (
                "The user explicitly reported an existing position. Use Add, Maintain, "
                "Reduce, or Exit as appropriate, and leave action_if_flat and "
                "action_if_holding empty."
            ),
        }.get(
            portfolio_status,
            "Treat the current position as unknown and use scenario mode.",
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.

{instrument_context}

{portfolio_context}

---

{rating_scale}

The market view is independent from assumptions about the user's position.
Keep the separate position action conditional on the supplied portfolio status.
{position_guidance}

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.
The final time horizon must be exactly `{horizon}`. Keep a hard stop distinct
from a conditional risk trigger, its confirmation interval, and the action on
trigger. The normalized portfolio status for this run is `{portfolio_status}`.

{NO_EXTERNAL_TOOLS}{get_language_instruction()}"""

        render_decision = (
            render_crypto_pm_decision if is_crypto else render_pm_decision
        )
        final_trade_decision = invoke_structured_or_freetext(
            crypto_structured_llm if is_crypto else structured_llm,
            llm,
            prompt,
            lambda decision: render_decision(
                decision,
                forced_horizon=horizon,
                portfolio_status=portfolio_status,
            ),
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
