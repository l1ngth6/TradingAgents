"""Optional xAI X Search source for sentiment analysis.

This module deliberately keeps X retrieval separate from the analyst LLM.  The
configured Grok model acts only as a bounded search worker through xAI's
Responses API; its evidence digest is then added to the existing Yahoo Finance,
StockTwits, and Reddit prompt blocks.

The feature is disabled by default and degrades to a plaintext placeholder on
missing credentials, transport errors, or malformed responses.  No caller has
to special-case failures, and enabling it never replaces another data source.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
from urllib.request import Request, urlopen

from .config import get_config

logger = logging.getLogger(__name__)

_RESPONSES_API = "https://api.x.ai/v1/responses"


def _search_prompt(ticker: str, start_date: str, end_date: str) -> str:
    """Build a bounded, evidence-first request for X market sentiment."""
    return f"""Search X for market sentiment about {ticker} from {start_date} through {end_date}, inclusive.

Return a concise evidence digest for a downstream financial sentiment analyst.

Selection requirements:
- Include only posts with meaningful reach or discussion relative to the normal X activity for this ticker. Prioritize posts with substantial views, replies, reposts, quotes, or likes; omit isolated low-impact posts.
- Prefer primary sources, recognized market participants, subject-matter experts, and posts that triggered visible discussion. Do not treat follower count alone as proof of relevance.
- Give the exact X post URL, timestamp, author, and available engagement metrics for every cited post. If engagement cannot be verified, label it unavailable and do not present the post as high-impact evidence.
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
    """Fetch a conservative X sentiment evidence digest using xAI X Search.

    The global ``x_search_enabled`` switch is checked before credentials or
    network access.  This guarantees that the optional feature is completely
    inert unless a user explicitly enables it.
    """
    config = get_config()
    if not config.get("x_search_enabled", False):
        return "<x_search disabled>"

    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("X Search is enabled but XAI_API_KEY is not set")
        return "<x_search unavailable: XAI_API_KEY is not set>"

    model = str(config.get("x_search_model") or "grok-4.3").strip()
    request_timeout = timeout if timeout is not None else float(config.get("x_search_timeout", 30))
    body = {
        "model": model,
        "input": _search_prompt(ticker, start_date, end_date),
        "tools": [
            {
                "type": "x_search",
                "from_date": start_date,
                "to_date": end_date,
            }
        ],
        "max_output_tokens": 1600,
    }
    req = Request(
        _RESPONSES_API,
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
    except (
        OSError,
        http.client.HTTPException,
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:
        logger.warning("xAI X Search failed for %s: %s", ticker, exc)
        return f"<x_search unavailable: {type(exc).__name__}>"

    text = _response_text(payload)
    if not text:
        logger.warning("xAI X Search returned no usable text for %s", ticker)
        return "<x_search unavailable: empty response>"
    return text
