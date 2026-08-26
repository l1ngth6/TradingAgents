"""Shared market-outlook vocabularies and a deterministic heuristic parser.

Stocks use Buy / Overweight / Hold / Underweight / Sell. Cryptocurrency runs
use directional Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish
labels so the outlook does not imply a benchmark allocation. Both are consumed
by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.
"""

from __future__ import annotations

import re

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

CRYPTO_OUTLOOKS_5_TIER: tuple[str, ...] = (
    "Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish",
)

_ALL_RATINGS = RATINGS_5_TIER + CRYPTO_OUTLOOKS_5_TIER
_CANONICAL_BY_LOWER = {rating.lower(): rating for rating in _ALL_RATINGS}
_RATING_ALTERNATIVES = "|".join(
    re.escape(rating) for rating in sorted(_ALL_RATINGS, key=len, reverse=True)
)

# Matches stock ``Rating`` / ``Recommendation`` and crypto ``Market Outlook``
# headers. Markdown bold wrappers and either a colon or hyphen are tolerated.
_RATING_LABEL_RE = re.compile(
    rf"(?:rating|recommendation|market\s+outlook).*?[:\-][\s*]*"
    rf"({_RATING_ALTERNATIVES})\b",
    re.IGNORECASE,
)
_RATING_WORD_RE = re.compile(
    rf"\b({_RATING_ALTERNATIVES})\b",
    re.IGNORECASE,
)


def parse_rating(text: str, default: str = "Hold") -> str:
    """Extract a stock rating or crypto market outlook from prose text.

    Two-pass strategy:
    1. Look for an explicit Rating, Recommendation, or Market Outlook label.
    2. Fall back to the first supported rating phrase found anywhere in the text.

    Returns the canonical label, or ``default`` when none appears.
    """
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m:
            return _CANONICAL_BY_LOWER[m.group(1).lower()]

    for line in text.splitlines():
        m = _RATING_WORD_RE.search(line)
        if m:
            return _CANONICAL_BY_LOWER[m.group(1).lower()]

    return default
