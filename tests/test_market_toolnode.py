"""The market analyst is bound (and prompt-instructed) to call
get_verified_market_snapshot; if the executor ToolNode doesn't register it, the
call fails and the model reports the tool "unavailable" and skips verification.

Regression guard for that wiring gap (snapshot bound to the LLM but missing from
the market ToolNode).
"""
import pytest

from tradingagents.agents.analysts.market_analyst import _tools_for_asset_type
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.unit
def test_market_toolnode_can_execute_verified_snapshot():
    # _create_tool_nodes does not use self -> call unbound (avoids building LLMs).
    nodes = TradingAgentsGraph._create_tool_nodes(None)
    market_tools = set(nodes["market"].tools_by_name)
    assert "get_verified_market_snapshot" in market_tools, (
        "get_verified_market_snapshot is bound to the market analyst but not "
        "registered in the market ToolNode, so the model's call fails."
    )
    # the other core market tools must remain too
    assert {"get_stock_data", "get_indicators"} <= market_tools
    assert "get_crypto_intraday_snapshot" in market_tools
    assert "get_crypto_derivatives" not in market_tools

    crypto_tools = set(nodes["crypto_intelligence"].tools_by_name)
    assert {
        "get_crypto_derivatives",
        "get_crypto_options",
        "get_crypto_onchain",
        "get_crypto_liquidations",
    } <= crypto_tools


@pytest.mark.unit
def test_stock_market_analyst_cannot_see_crypto_tools():
    tool_names = {tool.name for tool in _tools_for_asset_type("stock")}
    assert tool_names == {
        "get_stock_data",
        "get_indicators",
        "get_verified_market_snapshot",
    }
    assert "get_crypto_derivatives" not in tool_names


@pytest.mark.unit
def test_crypto_market_analyst_adds_only_intraday_technical_snapshot():
    tool_names = {tool.name for tool in _tools_for_asset_type("crypto")}
    assert tool_names == {
        "get_stock_data",
        "get_indicators",
        "get_verified_market_snapshot",
        "get_crypto_intraday_snapshot",
    }
