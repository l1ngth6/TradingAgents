"""Optional per-agent model and thinking-level overrides."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.llm_clients.agent_overrides import (
    AGENT_LLM_ROLE_DEFAULTS,
    build_agent_llm_overrides,
    parse_agent_llm_overrides,
)


def _config(**updates):
    config = {
        "agent_llm_overrides_enabled": True,
        "agent_llm_overrides": {},
        "llm_provider": "openai_compatible",
        "backend_url": "http://192.168.2.221:8080/v1",
        "quick_think_llm": "deepseek-v4-flash",
        "deep_think_llm": "gpt-5.6-sol",
    }
    config.update(updates)
    return config


@pytest.mark.unit
def test_feature_disabled_is_inert_even_with_invalid_payload(monkeypatch):
    create_client = MagicMock()
    monkeypatch.setattr(
        "tradingagents.llm_clients.agent_overrides.create_llm_client", create_client
    )

    result = build_agent_llm_overrides(
        _config(agent_llm_overrides_enabled=False, agent_llm_overrides="not-json"),
        {"reasoning_effort": "medium"},
    )

    assert result == {}
    create_client.assert_not_called()


@pytest.mark.unit
def test_json_config_builds_role_models_and_maps_thinking_level(monkeypatch):
    calls = []

    def fake_create_llm_client(**kwargs):
        calls.append(kwargs)
        client = MagicMock()
        client.get_llm.return_value = f"llm-{kwargs['model']}-{kwargs.get('reasoning_effort')}"
        return client

    monkeypatch.setattr(
        "tradingagents.llm_clients.agent_overrides.create_llm_client",
        fake_create_llm_client,
    )
    overrides = (
        '{"market_analyst":{"model":"deepseek-v4-flash","thinking_level":"low"},'
        '"research_manager":{"model":"gpt-5.6-sol","thinking_level":"high"}}'
    )

    result = build_agent_llm_overrides(
        _config(agent_llm_overrides=overrides),
        {"reasoning_effort": "medium", "max_retries": 6, "callbacks": ["stats"]},
    )

    assert result == {
        "market_analyst": "llm-deepseek-v4-flash-low",
        "research_manager": "llm-gpt-5.6-sol-high",
    }
    assert [call["reasoning_effort"] for call in calls] == ["low", "high"]
    assert all(call["base_url"] == "http://192.168.2.221:8080/v1" for call in calls)
    assert all(call["max_retries"] == 6 for call in calls)
    assert all(call["callbacks"] == ["stats"] for call in calls)


@pytest.mark.unit
def test_missing_fields_inherit_quick_deep_defaults(monkeypatch):
    calls = []

    def fake_create_llm_client(**kwargs):
        calls.append(kwargs)
        client = MagicMock()
        client.get_llm.return_value = kwargs["model"]
        return client

    monkeypatch.setattr(
        "tradingagents.llm_clients.agent_overrides.create_llm_client",
        fake_create_llm_client,
    )

    build_agent_llm_overrides(
        _config(
            agent_llm_overrides={
                "trader": {"thinking_level": "low"},
                "portfolio_manager": {"model": "gpt-5.6-sol"},
            }
        ),
        {"reasoning_effort": "medium"},
    )

    assert calls[0]["model"] == "deepseek-v4-flash"
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[1]["model"] == "gpt-5.6-sol"
    assert calls[1]["reasoning_effort"] == "medium"


@pytest.mark.unit
def test_null_thinking_level_removes_inherited_setting(monkeypatch):
    captured = {}

    def fake_create_llm_client(**kwargs):
        captured.update(kwargs)
        client = MagicMock()
        client.get_llm.return_value = "portfolio-llm"
        return client

    monkeypatch.setattr(
        "tradingagents.llm_clients.agent_overrides.create_llm_client",
        fake_create_llm_client,
    )

    result = build_agent_llm_overrides(
        _config(agent_llm_overrides={"portfolio_manager": {"thinking_level": None}}),
        {"reasoning_effort": "high", "max_retries": 4},
    )

    assert result == {"portfolio_manager": "portfolio-llm"}
    assert "reasoning_effort" not in captured
    assert captured["max_retries"] == 4


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider,expected_kwarg",
    [("google", "thinking_level"), ("anthropic", "effort")],
)
def test_thinking_level_maps_to_native_provider_kwarg(monkeypatch, provider, expected_kwarg):
    captured = {}

    def fake_create_llm_client(**kwargs):
        captured.update(kwargs)
        client = MagicMock()
        client.get_llm.return_value = "role-llm"
        return client

    monkeypatch.setattr(
        "tradingagents.llm_clients.agent_overrides.create_llm_client",
        fake_create_llm_client,
    )

    build_agent_llm_overrides(
        _config(
            llm_provider=provider,
            agent_llm_overrides={"market_analyst": {"thinking_level": "high"}},
        ),
        {},
    )

    assert captured[expected_kwarg] == "high"


@pytest.mark.unit
def test_all_documented_roles_are_accepted():
    overrides = {role: {"model": f"model-{role}"} for role in AGENT_LLM_ROLE_DEFAULTS}
    assert set(parse_agent_llm_overrides(overrides)) == set(AGENT_LLM_ROLE_DEFAULTS)


@pytest.mark.unit
@pytest.mark.parametrize(
    "payload,match",
    [
        ("not-json", "valid JSON"),
        ({"unknown_role": {"model": "m"}}, "Unknown agent LLM override role"),
        ({"trader": {"unknown": "x"}}, "Unknown keys"),
        ({"trader": {}}, "must set model and/or thinking_level"),
        ({"trader": {"model": ""}}, "non-empty string"),
        ({"trader": {"thinking_level": 3}}, "non-empty string or null"),
    ],
)
def test_invalid_override_config_fails_loudly(payload, match):
    with pytest.raises(ValueError, match=match):
        parse_agent_llm_overrides(payload)


@pytest.mark.unit
def test_graph_setup_uses_override_or_legacy_default():
    from tradingagents.graph.setup import GraphSetup

    setup = GraphSetup(
        quick_thinking_llm="quick",
        deep_thinking_llm="deep",
        tool_nodes={},
        conditional_logic=MagicMock(),
        agent_llms={"trader": "trader-override"},
    )

    assert setup._llm_for("trader", setup.quick_thinking_llm) == "trader-override"
    assert setup._llm_for("market_analyst", setup.quick_thinking_llm) == "quick"
    assert setup._llm_for("research_manager", setup.deep_thinking_llm) == "deep"


@pytest.mark.unit
def test_graph_setup_routes_every_agent_factory_to_its_role_override(monkeypatch):
    import tradingagents.graph.setup as setup_module
    from tradingagents.graph.setup import GraphSetup

    class FakeWorkflow:
        def __init__(self, _state_type):
            self.nodes = {}

        def add_node(self, name, node):
            self.nodes[name] = node

        def add_edge(self, *_args):
            pass

        def add_conditional_edges(self, *_args):
            pass

    role_by_factory = {
        "create_market_analyst": "market_analyst",
        "create_sentiment_analyst": "sentiment_analyst",
        "create_news_analyst": "news_analyst",
        "create_fundamentals_analyst": "fundamentals_analyst",
        "create_bull_researcher": "bull_researcher",
        "create_bear_researcher": "bear_researcher",
        "create_research_manager": "research_manager",
        "create_trader": "trader",
        "create_aggressive_debator": "aggressive_analyst",
        "create_neutral_debator": "neutral_analyst",
        "create_conservative_debator": "conservative_analyst",
        "create_portfolio_manager": "portfolio_manager",
    }
    for factory_name, role in role_by_factory.items():
        monkeypatch.setattr(
            setup_module,
            factory_name,
            lambda llm, role=role: (role, llm),
        )
    monkeypatch.setattr(setup_module, "create_msg_delete", lambda: "clear")
    monkeypatch.setattr(setup_module, "StateGraph", FakeWorkflow)

    role_llms = {role: f"llm-for-{role}" for role in AGENT_LLM_ROLE_DEFAULTS}
    setup = GraphSetup(
        quick_thinking_llm="quick",
        deep_thinking_llm="deep",
        tool_nodes={key: f"tools-{key}" for key in ("market", "social", "news", "fundamentals")},
        conditional_logic=MagicMock(),
        agent_llms=role_llms,
    )
    workflow = setup.setup_graph(("market", "social", "news", "fundamentals"))

    expected_nodes = {
        "Market Analyst": "market_analyst",
        "Sentiment Analyst": "sentiment_analyst",
        "News Analyst": "news_analyst",
        "Fundamentals Analyst": "fundamentals_analyst",
        "Bull Researcher": "bull_researcher",
        "Bear Researcher": "bear_researcher",
        "Research Manager": "research_manager",
        "Trader": "trader",
        "Aggressive Analyst": "aggressive_analyst",
        "Neutral Analyst": "neutral_analyst",
        "Conservative Analyst": "conservative_analyst",
        "Portfolio Manager": "portfolio_manager",
    }
    for node_name, role in expected_nodes.items():
        assert workflow.nodes[node_name] == (role, f"llm-for-{role}")
