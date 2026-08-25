"""Optional X Search source for sentiment analysis.

This module deliberately keeps X retrieval separate from the analyst LLM.  The
configured Grok model acts only as a bounded search worker through a Responses
API. The transport can use xAI directly or an explicitly selected
OpenAI-compatible gateway, with an endpoint and credential independent from the
main LLM; its evidence digest is then added to the Yahoo Finance, StockTwits,
and asset-specific Reddit prompt blocks as a formal source when enabled.

The feature is disabled by default and degrades to a plaintext placeholder on
missing credentials, transport errors, or malformed responses.  No caller has
to special-case failures, and enabling it never replaces another data source.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .config import get_config

logger = logging.getLogger(__name__)

_RESPONSES_API = "https://api.x.ai/v1/responses"
_SUPPORTED_PROVIDERS = frozenset({"xai", "openai_compatible"})
_SEARCH_INSTRUCTIONS = (
    "You are a bounded X Search worker for financial sentiment analysis. "
    "Use X Search results as evidence, respect the requested date range, "
    "include exact source URLs, and never invent posts or engagement metrics."
)


def _responses_url(base_url: str) -> str:
    """Resolve a Responses endpoint from an OpenAI-compatible base URL."""
    normalized = base_url.strip().rstrip("/")
    return normalized if normalized.endswith("/responses") else normalized + "/responses"


def _transport(config: dict) -> tuple[str, str, str] | tuple[None, None, str]:
    """Resolve request URL and credential for the selected X Search provider."""
    provider = str(config.get("x_search_provider") or "xai").strip().lower()
    if provider not in _SUPPORTED_PROVIDERS:
        valid = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        return None, None, f"unsupported provider {provider!r}; expected one of: {valid}"

    # Explicit X Search transport settings always win. This permits a dedicated
    # gateway, account, or subscription group without changing the main
    # openai_compatible endpoint and key. Provider-specific/global settings are
    # retained solely as convenient backward-compatible fallbacks.
    base_url = str(config.get("x_search_base_url") or "").strip()
    api_key = os.getenv("TRADINGAGENTS_X_SEARCH_API_KEY", "").strip()

    if provider == "xai":
        api_key = api_key or os.getenv("XAI_API_KEY", "").strip()
        if not api_key:
            return None, None, (
                "TRADINGAGENTS_X_SEARCH_API_KEY or XAI_API_KEY is not set"
            )
        return _responses_url(base_url) if base_url else _RESPONSES_API, api_key, provider

    base_url = base_url or str(config.get("backend_url") or "").strip()
    if not base_url:
        return None, None, (
            "TRADINGAGENTS_X_SEARCH_BASE_URL or "
            "TRADINGAGENTS_LLM_BACKEND_URL is not set for openai_compatible"
        )
    # Match the main openai_compatible client's support for keyless local
    # servers, but prefer the dedicated X Search credential when supplied.
    api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY", "").strip() or "EMPTY"
    return _responses_url(base_url), api_key, provider


def _search_prompt(ticker: str, start_date: str, end_date: str) -> str:
    """Build a bounded, evidence-first request for X market sentiment."""
    return f"""Search X for market sentiment about {ticker} from {start_date} through {end_date}, inclusive.

Return a concise evidence digest for a downstream financial sentiment analyst.

Selection requirements:
- Include only posts with meaningful reach or discussion relative to the normal X activity for this ticker. Prioritize posts with substantial views, replies, reposts, quotes, or likes; omit isolated low-impact posts.
- Exclude a post when every available engagement metric (views, replies, reposts, quotes, and likes) is zero. Do not use completely zero-interaction posts as sentiment evidence.
- Prefer primary sources, recognized market participants, subject-matter experts, and posts that triggered visible discussion. Do not treat follower count alone as proof of relevance.
- Give the exact X post URL, timestamp, author, and available engagement metrics for every cited post. If engagement cannot be verified, label it unavailable; include it only when it is a relevant primary-source factual announcement, and do not count it as demonstrated crowd sentiment.
- Separate factual announcements from opinions, speculation, memes, coordinated promotion, and possible bot activity.
- Cover both bullish and bearish evidence. Do not infer a broad consensus from a small or one-sided sample.
- Do not include information posted outside the requested dates.

Summarize the dominant narratives, direction of sentiment, contrary evidence, and data limitations. Never invent posts, metrics, URLs, or quotations."""


def _response_text(payload: object) -> str:
    """Extract assistant text and deduplicated citation URLs from Responses JSON."""
    if not isinstance(payload, dict):
        return ""

    # Some compatible Responses implementations expose the convenience field.
    top_level_text = payload.get("output_text")
    text_parts = [top_level_text.strip()] if isinstance(top_level_text, str) and top_level_text.strip() else []
    citations: list[tuple[str, str]] = []

    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "output_text":
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip() and text.strip() not in text_parts:
                    text_parts.append(text.strip())
                annotations = part.get("annotations")
                if not isinstance(annotations, list):
                    continue
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        continue
                    url = annotation.get("url")
                    if not isinstance(url, str) or not url.startswith(("https://", "http://")):
                        continue
                    title = annotation.get("title")
                    citations.append((
                        title.strip() if isinstance(title, str) and title.strip() else "X source",
                        url,
                    ))

    if not text_parts:
        return ""

    seen_urls: set[str] = set()
    source_lines = []
    for title, url in citations:
        if url in seen_urls:
            continue
        seen_urls.add(url)
        source_lines.append(f"- {title}: {url}")

    result = "\n\n".join(text_parts)
    if source_lines:
        result += "\n\nCited X sources:\n" + "\n".join(source_lines)
    return result


def fetch_x_sentiment(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    timeout: float | None = None,
) -> str:
    """Fetch a bounded X sentiment evidence digest using Grok X Search.

    The global ``x_search_enabled`` switch is checked before credentials or
    network access.  This guarantees that the optional feature is completely
    inert unless a user explicitly enables it.
    """
    config = get_config()
    if not config.get("x_search_enabled", False):
        return "<x_search disabled>"

    responses_url, api_key, provider_or_error = _transport(config)
    if responses_url is None or api_key is None:
        logger.warning("X Search is enabled but unavailable: %s", provider_or_error)
        return f"<x_search unavailable: {provider_or_error}>"
    provider = provider_or_error

    model = str(config.get("x_search_model") or "grok-4.6").strip()
    thinking_level = str(config.get("x_search_thinking_level") or "medium").strip()
    request_timeout = timeout if timeout is not None else float(config.get("x_search_timeout", 60))
    prompt = _search_prompt(ticker, start_date, end_date)
    search_tool = {"type": "x_search"}
    body = {
        "model": model,
        # Some otherwise Responses-compatible gateways require a non-empty
        # top-level instructions field before they will forward the request.
        "instructions": _SEARCH_INSTRUCTIONS,
        "reasoning": {"effort": thinking_level},
        # The string form is the smallest Responses-compatible input shape and
        # avoids gateways that accept an input array or reasoning separately
        # but fail when both appear in the same request.
        "input": prompt,
        "tools": [search_tool],
    }
    if provider == "xai":
        # Native xAI accepts the complete documented X Search shape. Generic
        # compatible gateways may reject these optional fields even though the
        # date range remains explicit in the prompt, so use the common subset
        # for openai_compatible transports.
        body["input"] = [{"role": "user", "content": prompt}]
        search_tool.update({"from_date": start_date, "to_date": end_date})
        body["max_output_tokens"] = int(config.get("x_search_max_output_tokens", 8000))
    req = Request(
        responses_url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "tradingagents/0.3 (+https://github.com/TauricResearch/TradingAgents)",
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=request_timeout) as resp:
            payload = json.loads(resp.read())
    except HTTPError as exc:
        error_body = exc.read(4096).decode("utf-8", errors="replace").strip()
        request_id = exc.headers.get("X-Request-Id") if exc.headers else None
        logger.warning(
            "X Search via %s failed for %s: HTTP %s %s "
            "(request_id=%s, body=%r)",
            provider,
            ticker,
            exc.code,
            exc.reason,
            request_id or "unknown",
            error_body or "<empty>",
        )
        return f"<x_search unavailable: HTTP {exc.code}>"
    except (
        OSError,
        http.client.HTTPException,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        logger.warning("X Search via %s failed for %s: %s", provider, ticker, exc)
        return f"<x_search unavailable: {type(exc).__name__}>"

    text = _response_text(payload)
    if not text:
        logger.warning(
            "X Search via %s returned no usable text for %s", provider, ticker
        )
        return "<x_search unavailable: empty response>"
    return text
