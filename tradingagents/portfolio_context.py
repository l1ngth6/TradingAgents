"""Canonical portfolio context shared by CLI and programmatic runs.

The context is deliberately small: it supplies enough information to turn a
market view into a position-aware action without collecting account balances,
wallet addresses, or other unnecessary private data.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


VALID_PORTFOLIO_STATUSES = {"unknown", "flat", "holding"}
VALID_POSITION_SIDES = {"long", "short", "unknown"}


def _optional_number(value: Any, field: str, *, minimum: float, maximum: float) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1].strip()
        if not cleaned:
            return None
        value = cleaned
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a number") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{field} must be between {minimum:g} and {maximum:g}")
    return number


def _legacy_status(context: Mapping[str, Any]) -> str:
    current = str(context.get("current_position", "unknown")).strip().lower()
    if current in {"", "unknown", "unspecified", "scenario"}:
        return "unknown"
    if current in {"none", "no", "flat", "cash", "no position"}:
        return "flat"
    return "holding"


def normalize_portfolio_context(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and reduce portfolio input to a stable, privacy-minimal shape.

    Empty input means ``unknown`` rather than assuming the user is flat or
    holding. A small compatibility shim accepts the earlier free-form
    ``current_position`` / ``max_drawdown`` keys used by programmatic callers.
    """
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("portfolio_context must be a mapping")

    status = str(value.get("status") or _legacy_status(value)).strip().lower()
    if status not in VALID_PORTFOLIO_STATUSES:
        raise ValueError(
            "portfolio_context.status must be one of: unknown, flat, holding"
        )

    normalized: dict[str, Any] = {"status": status}
    if status != "holding":
        return normalized

    side = str(value.get("side", "unknown")).strip().lower()
    if side not in VALID_POSITION_SIDES:
        raise ValueError("portfolio_context.side must be long, short, or unknown")
    normalized["side"] = side

    aliases = {
        "exposure_pct": value.get("exposure_pct", value.get("position_size_pct")),
        "average_entry_price": value.get("average_entry_price", value.get("cost_basis")),
        "leverage": value.get("leverage"),
        "max_loss_pct": value.get("max_loss_pct", value.get("max_drawdown")),
    }
    limits = {
        "exposure_pct": (0.01, 100.0),
        "average_entry_price": (0.00000001, 1_000_000_000_000.0),
        "leverage": (1.0, 1000.0),
        "max_loss_pct": (0.01, 100.0),
    }
    for key, raw in aliases.items():
        number = _optional_number(raw, key, minimum=limits[key][0], maximum=limits[key][1])
        if number is not None:
            normalized[key] = number
    return normalized


def portfolio_context_fingerprint(value: Mapping[str, Any] | None) -> str:
    """Return a short stable identity used to isolate checkpoint runs."""
    canonical = normalize_portfolio_context(value)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def render_portfolio_context(value: Mapping[str, Any] | None) -> str:
    """Render downstream-only instructions for Trader, Risk, and PM agents."""
    context = normalize_portfolio_context(value)
    status = context["status"]
    if status == "unknown":
        return (
            "Portfolio context: current position is unknown (scenario mode). Do not "
            "assume the user is either flat or invested. Keep the market stance "
            "independent from the position action, and provide both an action if flat "
            "and an action if already holding. Do not invent account values or a "
            "fraction to buy or sell."
        )
    if status == "flat":
        return (
            "Portfolio context: the user is currently flat (no position). Treat Hold "
            "as wait/avoid entry rather than maintain a nonexistent holding. Do not "
            "recommend reducing or exiting a position that does not exist."
        )

    labels = {
        "side": "side",
        "exposure_pct": "current exposure (% of portfolio capital)",
        "average_entry_price": "average entry price",
        "leverage": "leverage",
        "max_loss_pct": "maximum tolerable loss/drawdown (%)",
    }
    details = [
        f"{labels[key]}={val:g}" if isinstance(val, float) else f"{labels[key]}={val}"
        for key, val in context.items()
        if key in labels
    ]
    return (
        "Portfolio context: the user is currently holding a position; "
        + "; ".join(details)
        + ". Distinguish maintain/add/reduce/exit from the independent market stance, "
          "respect any supplied loss limit, and do not infer missing account values."
    )


def portfolio_context_summary(value: Mapping[str, Any] | None) -> str:
    """Compact non-prescriptive summary for CLI logs and report metadata."""
    context = normalize_portfolio_context(value)
    if context["status"] != "holding":
        return context["status"]
    fields = ["status=holding", f"side={context.get('side', 'unknown')}"]
    for key in ("exposure_pct", "average_entry_price", "leverage", "max_loss_pct"):
        if key in context:
            fields.append(f"{key}={context[key]:g}")
    return "; ".join(fields)
