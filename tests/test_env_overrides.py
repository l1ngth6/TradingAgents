"""Tests for TRADINGAGENTS_* env-var overlay onto DEFAULT_CONFIG."""

from __future__ import annotations

import importlib

import pytest

import tradingagents.default_config as default_config_module


def _reload_with_env(monkeypatch, **overrides):
    """Set/clear env vars then reload default_config to re-evaluate DEFAULT_CONFIG."""
    for key in list(default_config_module._ENV_OVERRIDES):
        monkeypatch.delenv(key, raising=False)
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    return importlib.reload(default_config_module)


def test_no_env_uses_built_in_defaults(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gpt-5.5"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gpt-5.4-mini"
    assert dc.DEFAULT_CONFIG["backend_url"] is None
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is False


def test_string_overrides(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="google",
        TRADINGAGENTS_DEEP_THINK_LLM="gemini-3-pro-preview",
        TRADINGAGENTS_QUICK_THINK_LLM="gemini-3-flash-preview",
        TRADINGAGENTS_LLM_BACKEND_URL="https://example.invalid/v1",
        TRADINGAGENTS_OUTPUT_LANGUAGE="Chinese",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "google"
    assert dc.DEFAULT_CONFIG["deep_think_llm"] == "gemini-3-pro-preview"
    assert dc.DEFAULT_CONFIG["quick_think_llm"] == "gemini-3-flash-preview"
    assert dc.DEFAULT_CONFIG["backend_url"] == "https://example.invalid/v1"
    assert dc.DEFAULT_CONFIG["output_language"] == "Chinese"


def test_int_coercion(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="3",
        TRADINGAGENTS_MAX_RISK_ROUNDS="2",
    )
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 3
    assert isinstance(dc.DEFAULT_CONFIG["max_debate_rounds"], int)
    assert dc.DEFAULT_CONFIG["max_risk_discuss_rounds"] == 2
    assert isinstance(dc.DEFAULT_CONFIG["max_risk_discuss_rounds"], int)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True), ("True", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("False", False), ("0", False), ("no", False), ("off", False),
    ],
)
def test_bool_coercion(monkeypatch, raw, expected):
    dc = _reload_with_env(monkeypatch, TRADINGAGENTS_CHECKPOINT_ENABLED=raw)
    assert dc.DEFAULT_CONFIG["checkpoint_enabled"] is expected


def test_reasoning_thinking_overrides(monkeypatch):
    """The provider reasoning/thinking knobs are env-configurable (non-interactive runs)."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_OPENAI_REASONING_EFFORT="high",
        TRADINGAGENTS_GOOGLE_THINKING_LEVEL="minimal",
        TRADINGAGENTS_ANTHROPIC_EFFORT="low",
    )
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] == "high"
    assert dc.DEFAULT_CONFIG["google_thinking_level"] == "minimal"
    assert dc.DEFAULT_CONFIG["anthropic_effort"] == "low"


def test_reasoning_effort_defaults_to_none(monkeypatch):
    """Unset reasoning/thinking knobs stay None so each provider uses its own default."""
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["openai_reasoning_effort"] is None
    assert dc.DEFAULT_CONFIG["google_thinking_level"] is None
    assert dc.DEFAULT_CONFIG["anthropic_effort"] is None


def test_agent_llm_override_env_config(monkeypatch):
    payload = '{"trader":{"model":"gpt-5.6-sol","thinking_level":"high"}}'
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_AGENT_LLM_OVERRIDES_ENABLED="true",
        TRADINGAGENTS_AGENT_LLM_OVERRIDES=payload,
    )
    assert dc.DEFAULT_CONFIG["agent_llm_overrides_enabled"] is True
    # Parsing is intentionally deferred until the feature is enabled at graph
    # construction, so a disabled optional payload remains completely inert.
    assert dc.DEFAULT_CONFIG["agent_llm_overrides"] == payload


def test_agent_llm_overrides_disabled_by_default(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["agent_llm_overrides_enabled"] is False
    assert dc.DEFAULT_CONFIG["agent_llm_overrides"] == ""


def test_x_search_env_config(monkeypatch):
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_X_SEARCH_ENABLED="true",
        TRADINGAGENTS_X_SEARCH_PROVIDER="openai_compatible",
        TRADINGAGENTS_X_SEARCH_BASE_URL="http://x-search.example/v1",
        TRADINGAGENTS_X_SEARCH_MODEL="grok-test",
        TRADINGAGENTS_X_SEARCH_THINKING_LEVEL="low",
        TRADINGAGENTS_X_SEARCH_TIMEOUT="45",
        TRADINGAGENTS_X_SEARCH_RETRY_ENABLED="true",
        TRADINGAGENTS_X_SEARCH_MAX_RETRIES="4",
        TRADINGAGENTS_X_SEARCH_RETRY_INTERVAL="1.5",
        TRADINGAGENTS_X_SEARCH_MAX_OUTPUT_TOKENS="9000",
    )
    assert dc.DEFAULT_CONFIG["x_search_enabled"] is True
    assert dc.DEFAULT_CONFIG["x_search_provider"] == "openai_compatible"
    assert dc.DEFAULT_CONFIG["x_search_base_url"] == "http://x-search.example/v1"
    assert dc.DEFAULT_CONFIG["x_search_model"] == "grok-test"
    assert dc.DEFAULT_CONFIG["x_search_thinking_level"] == "low"
    assert dc.DEFAULT_CONFIG["x_search_timeout"] == 45
    assert dc.DEFAULT_CONFIG["x_search_retry_enabled"] is True
    assert dc.DEFAULT_CONFIG["x_search_max_retries"] == 4
    assert dc.DEFAULT_CONFIG["x_search_retry_interval"] == 1.5
    assert dc.DEFAULT_CONFIG["x_search_max_output_tokens"] == 9000


def test_x_search_disabled_by_default(monkeypatch):
    dc = _reload_with_env(monkeypatch)
    assert dc.DEFAULT_CONFIG["x_search_enabled"] is False
    assert dc.DEFAULT_CONFIG["x_search_provider"] == "xai"
    assert dc.DEFAULT_CONFIG["x_search_base_url"] is None
    assert dc.DEFAULT_CONFIG["x_search_model"] == "grok-4.6"
    assert dc.DEFAULT_CONFIG["x_search_thinking_level"] == "medium"
    assert dc.DEFAULT_CONFIG["x_search_timeout"] == 60
    assert dc.DEFAULT_CONFIG["x_search_retry_enabled"] is False
    assert dc.DEFAULT_CONFIG["x_search_max_retries"] == 2
    assert dc.DEFAULT_CONFIG["x_search_retry_interval"] == 2.0
    assert dc.DEFAULT_CONFIG["x_search_max_output_tokens"] == 8000


def test_empty_env_value_is_passthrough(monkeypatch):
    """Empty TRADINGAGENTS_* values must not clobber the built-in default."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_LLM_PROVIDER="",
        TRADINGAGENTS_MAX_DEBATE_ROUNDS="",
    )
    assert dc.DEFAULT_CONFIG["llm_provider"] == "openai"
    assert dc.DEFAULT_CONFIG["max_debate_rounds"] == 1


def test_invalid_int_raises(monkeypatch):
    """Garbage int values should surface a ValueError at import, not silently misconfigure."""
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "not-a-number")
    with pytest.raises(ValueError, match="TRADINGAGENTS_MAX_DEBATE_ROUNDS"):
        importlib.reload(default_config_module)
    # Restore module state for subsequent tests in this process
    monkeypatch.delenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", raising=False)
    importlib.reload(default_config_module)


@pytest.mark.parametrize("bad", ["treu", "flase", "maybe", "2", "enabled"])
def test_invalid_bool_raises(monkeypatch, bad):
    """A misspelled boolean must fail loudly (like ints) instead of silently False."""
    monkeypatch.setenv("TRADINGAGENTS_CHECKPOINT_ENABLED", bad)
    with pytest.raises(ValueError, match="TRADINGAGENTS_CHECKPOINT_ENABLED"):
        importlib.reload(default_config_module)
    monkeypatch.delenv("TRADINGAGENTS_CHECKPOINT_ENABLED", raising=False)
    importlib.reload(default_config_module)


def test_unknown_env_var_is_ignored(monkeypatch):
    """Env vars outside _ENV_OVERRIDES must not bleed into DEFAULT_CONFIG."""
    dc = _reload_with_env(
        monkeypatch,
        TRADINGAGENTS_NONEXISTENT_KEY="oops",
    )
    assert "nonexistent_key" not in dc.DEFAULT_CONFIG
