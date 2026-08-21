"""Optional per-agent model and thinking-level overrides."""

from __future__ import annotations

import json
from typing import Any

from .factory import create_llm_client


# Public, stable role IDs accepted by ``agent_llm_overrides``. The value is the
# legacy model tier inherited when an override changes only ``thinking_level``.
AGENT_LLM_ROLE_DEFAULTS: dict[str, str] = {
    "market_analyst": "quick",
    "sentiment_analyst": "quick",
    "news_analyst": "quick",
    "fundamentals_analyst": "quick",
    "bull_researcher": "quick",
    "bear_researcher": "quick",
    "research_manager": "deep",
    "trader": "quick",
    "aggressive_analyst": "quick",
    "neutral_analyst": "quick",
    "conservative_analyst": "quick",
    "portfolio_manager": "deep",
}

_ALLOWED_OVERRIDE_KEYS = frozenset({"model", "thinking_level"})


def parse_agent_llm_overrides(value: Any) -> dict[str, dict[str, Any]]:
    """Parse and validate a programmatic dict or JSON environment value."""
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"agent_llm_overrides must be valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ValueError("agent_llm_overrides must be a JSON object keyed by agent role")

    normalized = {}
    for role, override in value.items():
        if role not in AGENT_LLM_ROLE_DEFAULTS:
            valid = ", ".join(AGENT_LLM_ROLE_DEFAULTS)
            raise ValueError(f"Unknown agent LLM override role {role!r}; valid roles: {valid}")
        if not isinstance(override, dict):
            raise ValueError(f"Override for {role!r} must be a JSON object")

        unknown_keys = set(override) - _ALLOWED_OVERRIDE_KEYS
        if unknown_keys:
            unknown = ", ".join(sorted(unknown_keys))
            raise ValueError(f"Unknown keys for agent role {role!r}: {unknown}")
        if not override:
            raise ValueError(f"Override for {role!r} must set model and/or thinking_level")

        model = override.get("model")
        if "model" in override and (not isinstance(model, str) or not model.strip()):
            raise ValueError(f"model for agent role {role!r} must be a non-empty string")
        thinking_level = override.get("thinking_level")
        if "thinking_level" in override and thinking_level is not None and (
            not isinstance(thinking_level, str) or not thinking_level.strip()
        ):
            raise ValueError(
                f"thinking_level for agent role {role!r} must be a non-empty string or null"
            )

        normalized_override = dict(override)
        if isinstance(model, str):
            normalized_override["model"] = model.strip()
        if isinstance(thinking_level, str):
            normalized_override["thinking_level"] = thinking_level.strip()
        normalized[role] = normalized_override

    return normalized


def _thinking_kwarg_for_provider(provider: str) -> str | None:
    """Translate the generic config field to the selected provider's kwarg."""
    provider = provider.lower()
    if provider == "google":
        return "thinking_level"
    if provider == "anthropic":
        return "effort"
    if provider == "azure":
        return "reasoning_effort"

    # Import lazily so parsing/validation does not import the OpenAI SDK.
    from .openai_client import is_openai_compatible

    if is_openai_compatible(provider):
        return "reasoning_effort"
    return None


def build_agent_llm_overrides(
    config: dict[str, Any],
    shared_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Build only the explicitly configured per-role LLM instances.

    When the feature flag is false, the override payload is intentionally not
    parsed or validated: the feature is inert and every role keeps the existing
    Quick/Deep clients. Missing fields within an enabled role override inherit
    the corresponding global model tier and provider kwargs. An explicit JSON
    ``null`` thinking level removes the inherited provider-wide setting.
    """
    if not config.get("agent_llm_overrides_enabled", False):
        return {}

    overrides = parse_agent_llm_overrides(config.get("agent_llm_overrides"))
    provider = str(config.get("llm_provider", "")).strip().lower()
    thinking_kwarg = (
        _thinking_kwarg_for_provider(provider)
        if any("thinking_level" in override for override in overrides.values())
        else None
    )
    llms = {}

    for role, override in overrides.items():
        default_tier = AGENT_LLM_ROLE_DEFAULTS[role]
        default_model_key = f"{default_tier}_think_llm"
        model = override.get("model") or config[default_model_key]
        kwargs = dict(shared_kwargs)

        if "thinking_level" in override:
            value = override["thinking_level"]
            if thinking_kwarg is None and value is not None:
                raise ValueError(
                    f"Provider {provider!r} does not expose a supported thinking-level "
                    f"parameter for agent role {role!r}"
                )
            if thinking_kwarg is not None:
                if value is None:
                    kwargs.pop(thinking_kwarg, None)
                else:
                    kwargs[thinking_kwarg] = value

        client = create_llm_client(
            provider=provider,
            model=model,
            base_url=config.get("backend_url"),
            **kwargs,
        )
        llms[role] = client.get_llm()

    return llms
