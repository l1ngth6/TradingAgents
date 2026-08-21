"""Optional xAI X Search sentiment source."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from tradingagents.dataflows import x_search
from tradingagents.dataflows.config import set_config


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


@pytest.mark.unit
def test_disabled_is_inert(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    set_config({"x_search_enabled": False})
    with patch.object(x_search, "urlopen") as mocked:
        result = x_search.fetch_x_sentiment("NVDA", "2026-01-08", "2026-01-15")
    assert result == "<x_search disabled>"
    mocked.assert_not_called()


@pytest.mark.unit
def test_enabled_requires_xai_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_X_SEARCH_API_KEY", raising=False)
    set_config({"x_search_enabled": True})
    with patch.object(x_search, "urlopen") as mocked:
        result = x_search.fetch_x_sentiment("NVDA", "2026-01-08", "2026-01-15")
    assert "XAI_API_KEY is not set" in result
    mocked.assert_not_called()


@pytest.mark.unit
def test_request_uses_x_search_dates_and_engagement_filter(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_X_SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "secret-test-key")
    set_config({
        "x_search_enabled": True,
        "x_search_model": "grok-test",
        "x_search_thinking_level": "low",
        "x_search_timeout": 17,
        "x_search_max_output_tokens": 9000,
    })
    payload = {
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": "High-engagement discussion was mixed.",
                "annotations": [{
                    "type": "url_citation",
                    "title": "Example post",
                    "url": "https://x.com/example/status/1",
                }],
            }],
        }],
    }
    seen = {}

    def fake_urlopen(req, timeout):
        seen["request"] = req
        seen["timeout"] = timeout
        return _Response(payload)

    with patch.object(x_search, "urlopen", side_effect=fake_urlopen):
        result = x_search.fetch_x_sentiment("NVDA", "2026-01-08", "2026-01-15")

    request_body = json.loads(seen["request"].data)
    assert seen["request"].full_url == "https://api.x.ai/v1/responses"
    assert seen["request"].get_header("Authorization") == "Bearer secret-test-key"
    assert request_body["model"] == "grok-test"
    assert request_body["reasoning"] == {"effort": "low"}
    assert request_body["tools"] == [{
        "type": "x_search", "from_date": "2026-01-08", "to_date": "2026-01-15",
    }]
    assert "omit isolated low-impact posts" in request_body["input"][0]["content"]
    assert request_body["max_output_tokens"] == 9000
    assert seen["timeout"] == 17
    assert "High-engagement discussion was mixed." in result
    assert "https://x.com/example/status/1" in result


@pytest.mark.unit
def test_openai_compatible_reuses_custom_endpoint_and_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("TRADINGAGENTS_X_SEARCH_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "custom-endpoint-key")
    set_config({
        "x_search_enabled": True,
        "x_search_provider": "openai_compatible",
        "backend_url": "http://192.168.2.221:8080/v1/",
        "x_search_model": "grok-build-0.1",
        "x_search_thinking_level": "medium",
    })
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["authorization"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return _Response({"output_text": "Subscription-backed X evidence."})

    with patch.object(x_search, "urlopen", side_effect=fake_urlopen):
        result = x_search.fetch_x_sentiment("NVDA", "2026-01-08", "2026-01-15")

    assert seen["url"] == "http://192.168.2.221:8080/v1/responses"
    assert seen["authorization"] == "Bearer custom-endpoint-key"
    assert seen["body"]["model"] == "grok-build-0.1"
    assert seen["body"]["reasoning"] == {"effort": "medium"}
    assert seen["body"]["tools"][0]["type"] == "x_search"
    assert result == "Subscription-backed X evidence."


@pytest.mark.unit
def test_x_search_dedicated_endpoint_and_key_override_global_settings(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_X_SEARCH_API_KEY", "dedicated-x-key")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "main-llm-key")
    set_config({
        "x_search_enabled": True,
        "x_search_provider": "openai_compatible",
        "x_search_base_url": "http://x-search.example/v1/responses",
        "backend_url": "http://main-llm.example/v1",
    })
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["authorization"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return _Response({"output_text": "Dedicated transport works."})

    with patch.object(x_search, "urlopen", side_effect=fake_urlopen):
        result = x_search.fetch_x_sentiment("NVDA", "2026-01-08", "2026-01-15")

    assert seen["url"] == "http://x-search.example/v1/responses"
    assert seen["authorization"] == "Bearer dedicated-x-key"
    assert seen["body"]["input"][0]["role"] == "user"
    assert result == "Dedicated transport works."


@pytest.mark.unit
def test_xai_mode_can_use_dedicated_endpoint_and_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.setenv("TRADINGAGENTS_X_SEARCH_API_KEY", "gateway-key")
    set_config({
        "x_search_enabled": True,
        "x_search_provider": "xai",
        "x_search_base_url": "https://grok-gateway.example/v1",
    })
    seen = {}

    def fake_urlopen(req, timeout):
        seen["url"] = req.full_url
        seen["authorization"] = req.get_header("Authorization")
        return _Response({"output_text": "Native-compatible gateway works."})

    with patch.object(x_search, "urlopen", side_effect=fake_urlopen):
        result = x_search.fetch_x_sentiment("NVDA", "2026-01-08", "2026-01-15")

    assert seen["url"] == "https://grok-gateway.example/v1/responses"
    assert seen["authorization"] == "Bearer gateway-key"
    assert result == "Native-compatible gateway works."


@pytest.mark.unit
def test_openai_compatible_requires_backend_url(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "custom-endpoint-key")
    set_config({
        "x_search_enabled": True,
        "x_search_provider": "openai_compatible",
        "backend_url": None,
    })
    with patch.object(x_search, "urlopen") as mocked:
        result = x_search.fetch_x_sentiment("NVDA", "2026-01-08", "2026-01-15")
    assert "TRADINGAGENTS_LLM_BACKEND_URL is not set" in result
    mocked.assert_not_called()


@pytest.mark.unit
def test_empty_response_degrades_gracefully(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "secret-test-key")
    set_config({"x_search_enabled": True})
    with patch.object(x_search, "urlopen", return_value=_Response({"output": []})):
        result = x_search.fetch_x_sentiment("NVDA", "2026-01-08", "2026-01-15")
    assert result == "<x_search unavailable: empty response>"
