"""Crypto-native data-source regression tests.

No test performs network I/O.  The fixtures model vendor responses around the
requested analysis date so look-ahead filtering and symbol conversion stay
explicitly covered.
"""

from datetime import datetime, timezone

import pytest

from tradingagents.dataflows import alternative_me, binance_crypto
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp() * 1000)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BTC-USD", "BTCUSDT"),
        ("BTCUSD", "BTCUSDT"),
        ("BTC-USDT", "BTCUSDT"),
        ("eth-usdc", "ETHUSDT"),
        ("BNB-USD", "BNBUSDT"),
        ("SUIUSD", "SUIUSDT"),
    ],
)
def test_binance_symbol_conversion(raw, expected):
    assert binance_crypto._binance_symbol(raw) == expected


@pytest.mark.unit
def test_binance_rejects_non_crypto_symbol():
    with pytest.raises(NoMarketDataError):
        binance_crypto._binance_symbol("AAPL")


@pytest.mark.unit
def test_binance_context_filters_rows_after_analysis_date(monkeypatch):
    before = _ms("2026-01-09T12:00:00")
    after = _ms("2026-01-11T12:00:00")
    seen_params = []

    def fake_request(path, params):
        seen_params.append((path, params))
        if path.endswith("fundingRate"):
            return [
                {"fundingTime": before, "fundingRate": "0.0001"},
                {"fundingTime": after, "fundingRate": "0.9999"},
            ]
        if path.endswith("globalLongShortAccountRatio"):
            return [
                {
                    "timestamp": before,
                    "longAccount": "0.55",
                    "shortAccount": "0.45",
                    "longShortRatio": "1.222",
                },
                {
                    "timestamp": after,
                    "longAccount": "0.99",
                    "shortAccount": "0.01",
                    "longShortRatio": "99",
                },
            ]
        if path.endswith("openInterestHist"):
            return [
                {
                    "timestamp": before,
                    "sumOpenInterest": "100",
                    "sumOpenInterestValue": "1000000",
                }
            ]
        return [
            {
                "timestamp": before,
                "buyVol": "12",
                "sellVol": "10",
                "buySellRatio": "1.2",
            }
        ]

    monkeypatch.setattr(binance_crypto, "_request", fake_request)
    report = binance_crypto.get_crypto_derivatives("BTC-USD", "2026-01-10", 7)

    assert "BTCUSDT" in report
    assert "2026-01-09 12:00" in report
    assert "2026-01-11" not in report
    assert "99.000" not in report
    assert all(params["symbol"] == "BTCUSDT" for _, params in seen_params)
    assert len({params["endTime"] for _, params in seen_params}) == 1


@pytest.mark.unit
def test_binance_partial_history_degrades_by_section(monkeypatch):
    before = _ms("2026-01-09T12:00:00")

    def fake_request(path, _params):
        if path.endswith("fundingRate"):
            return [{"fundingTime": before, "fundingRate": "-0.0002"}]
        return []

    monkeypatch.setattr(binance_crypto, "_request", fake_request)
    report = binance_crypto.get_crypto_derivatives("BTC-USD", "2026-01-10", 7)

    assert "-0.0200%" in report
    assert "DATA_UNAVAILABLE" in report
    assert "retained historical window" in report


@pytest.mark.unit
def test_fear_greed_filters_future_observations(monkeypatch):
    rows = [
        {
            "timestamp": str(_ms("2026-01-11T00:00:00") // 1000),
            "value": "90",
            "value_classification": "Extreme Greed",
        },
        {
            "timestamp": str(_ms("2026-01-10T00:00:00") // 1000),
            "value": "40",
            "value_classification": "Fear",
        },
        {
            "timestamp": str(_ms("2026-01-09T00:00:00") // 1000),
            "value": "35",
            "value_classification": "Fear",
        },
    ]
    monkeypatch.setattr(
        alternative_me,
        "_request",
        lambda _limit: {"data": rows, "metadata": {"error": None}},
    )

    report = alternative_me.get_crypto_fear_greed("ETH-USD", "2026-01-10", 2)
    assert "40/100 (Fear)" in report
    assert "2026-01-09" in report
    assert "2026-01-11" not in report
    assert "Bitcoin-centric" in report
    assert "alternative.me/crypto/fear-and-greed-index" in report


@pytest.mark.unit
def test_fear_greed_rejects_equity(monkeypatch):
    monkeypatch.setattr(alternative_me, "_request", lambda _limit: pytest.fail("network call"))
    with pytest.raises(NoMarketDataError):
        alternative_me.get_crypto_fear_greed("NVDA", "2026-01-10")
