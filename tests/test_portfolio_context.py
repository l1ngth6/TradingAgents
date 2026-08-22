import pytest

from tradingagents.portfolio_context import (
    normalize_portfolio_context,
    portfolio_context_fingerprint,
    render_portfolio_context,
)


@pytest.mark.unit
def test_empty_context_defaults_to_unknown_scenario():
    assert normalize_portfolio_context(None) == {"status": "unknown"}
    rendered = render_portfolio_context(None)
    assert "action if flat" in rendered
    assert "action if already holding" in rendered


@pytest.mark.unit
def test_holding_context_is_typed_and_legacy_aliases_work():
    context = normalize_portfolio_context({
        "current_position": "long BTC",
        "side": "long",
        "position_size_pct": "25%",
        "cost_basis": "71000",
        "leverage": "2.5",
        "max_drawdown": "6%",
    })
    assert context == {
        "status": "holding",
        "side": "long",
        "exposure_pct": 25.0,
        "average_entry_price": 71000.0,
        "leverage": 2.5,
        "max_loss_pct": 6.0,
    }


@pytest.mark.unit
def test_flat_discards_irrelevant_holding_fields():
    assert normalize_portfolio_context({
        "status": "flat", "average_entry_price": 10, "leverage": 3,
    }) == {"status": "flat"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "context",
    [
        {"status": "invalid"},
        {"status": "holding", "side": "up"},
        {"status": "holding", "exposure_pct": 101},
        {"status": "holding", "leverage": 0},
    ],
)
def test_invalid_context_is_rejected(context):
    with pytest.raises(ValueError):
        normalize_portfolio_context(context)


@pytest.mark.unit
def test_fingerprint_is_stable_and_position_sensitive():
    first = portfolio_context_fingerprint({"status": "holding", "side": "long", "leverage": 2})
    reordered = portfolio_context_fingerprint({"leverage": 2.0, "side": "long", "status": "holding"})
    assert first == reordered
    assert first != portfolio_context_fingerprint({"status": "flat"})
