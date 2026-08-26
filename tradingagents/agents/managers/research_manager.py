"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import (
    CryptoResearchPlan,
    ResearchPlan,
    render_crypto_research_plan,
    render_research_plan,
)
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")
    crypto_structured_llm = bind_structured(
        llm, CryptoResearchPlan, "Research Manager (crypto)"
    )

    def research_manager_node(state) -> dict:
        is_crypto = state.get("asset_type") == "crypto"
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        stance_scale = (
            """**Crypto Market Outlook Scale** (use exactly one):
- **Strong Bullish**: Strong conviction in material upside
- **Bullish**: Constructive directional view with favorable risk/reward
- **Neutral**: Balanced or directionally inconclusive view
- **Bearish**: Cautious directional view with unfavorable risk/reward
- **Strong Bearish**: Strong conviction in material downside

Use directional crypto-market language only. Do not use Buy, Overweight, Hold,
Underweight, or Sell as the market outlook, because those labels can imply a
transaction or a benchmark allocation. In a free-text response, label this
field exactly **Market Outlook**."""
            if is_crypto
            else """**Market Stance Scale** (use exactly one):
- **Buy**: Strong conviction in the bull thesis
- **Overweight**: Constructive view with favorable risk/reward
- **Hold**: Balanced or neutral view
- **Underweight**: Cautious view with unfavorable risk/reward
- **Sell**: Strong conviction in the bear thesis"""
        )

        prompt = f"""As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.

{instrument_context}

---

{stance_scale}

Commit to a clear stance whenever the debate's strongest arguments warrant one;
reserve {"Neutral" if is_crypto else "Hold"} for situations where the evidence
on both sides is genuinely balanced.
Your plan must remain position-independent. Define market conditions, entry or
invalidation levels, and execution considerations, but do not assume whether the
user is flat or holding; the Trader and risk team receive that context later.

---

**Debate History:**
{history}

{NO_EXTERNAL_TOOLS}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            crypto_structured_llm if is_crypto else structured_llm,
            llm,
            prompt,
            render_crypto_research_plan if is_crypto else render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
