"""Crypto-native data-source regression tests.

No test performs network I/O.  The fixtures model vendor responses around the
requested analysis date so look-ahead filtering and symbol conversion stay
explicitly covered.
"""

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.dataflows import (
    alternative_me,
    binance_crypto,
    binance_spot,
    crypto_intelligence,
)
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
        ("BTC/USDT", "BTCUSDT"),
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
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BTC-USD", ("BTCUSDT", "USDT")),
        ("BTC/USD", ("BTCUSDT", "USDT")),
        ("btc/usdt", ("BTCUSDT", "USDT")),
        ("ETH-USDC", ("ETHUSDC", "USDC")),
    ],
)
def test_binance_spot_symbol_conversion(raw, expected):
    assert binance_spot._binance_spot_symbol(raw) == expected


@pytest.mark.unit
def test_binance_spot_uses_only_completed_utc_candles(monkeypatch):
    completed_open = _ms("2026-08-21T00:00:00")
    completed_close = _ms("2026-08-21T23:59:59")
    future_open = _ms("2099-08-22T00:00:00")
    future_close = _ms("2099-08-22T23:59:59")

    monkeypatch.setattr(
        binance_spot,
        "_request_klines",
        lambda _params: [
            [completed_open, "73000", "79000", "72000", "78000", "100", completed_close, "7600000", 1000],
            [future_open, "78000", "80000", "77000", "79000", "50", future_close, "3900000", 500],
        ],
    )
    binance_spot._fetch_kline_rows.cache_clear()

    frame = binance_spot.get_binance_spot_ohlcv_frame(
        "BTC/USDT", "2026-08-21", "2026-08-21"
    )

    assert frame.iloc[-1]["Close"] == 78000
    assert frame.iloc[-1]["Volume"] == 100
    assert frame.iloc[-1]["Quote Volume"] == "7600000"
    assert "Binance Spot BTCUSDT" in frame.attrs["market_data_source"]


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


@pytest.mark.unit
def test_coin_metrics_reported_spot_activity_is_completed_daily_auxiliary(monkeypatch):
    asset_rows = []
    for day in range(1, 12):
        asset_rows.append(
            {
                "asset": "btc",
                "time": f"2026-01-{day:02d}T00:00:00.000000000Z",
                "SplyCur": "20000000",
                "CapMrktCurUSD": "1000000000000",
                "TxCnt": "500000",
                "AdrActCnt": "750000",
                "volume_reported_spot_usd_1d": str(day * 1_000_000_000),
            }
        )

    stable_rows = [
        {
            "asset": asset,
            "time": "2026-01-10T00:00:00.000000000Z",
            "SplyCur": supply,
        }
        for asset, supply in (("usdt", "100"), ("usdc", "50"))
    ]

    def fake_coin_metrics_rows(assets, metrics, _start_time, _end_time):
        if assets == "btc":
            assert "volume_reported_spot_usd_1d" in metrics
            return asset_rows
        return stable_rows

    monkeypatch.setattr(
        crypto_intelligence, "_coin_metrics_rows", fake_coin_metrics_rows
    )
    monkeypatch.delenv("DUNE_API_KEY", raising=False)
    monkeypatch.delenv("DUNE_CRYPTO_ONCHAIN_QUERY_ID", raising=False)
    monkeypatch.delenv("DUNE_CRYPTO_ETF_QUERY_ID", raising=False)
    crypto_intelligence.get_crypto_onchain.cache_clear()

    report = crypto_intelligence.get_crypto_onchain(
        "BTC-USD", "2026-01-01", "2026-01-10"
    )

    assert "Cross-market reported spot activity (auxiliary)" in report
    assert "2026-01-10 ($10,000,000,000.00)" in report
    assert "Consecutive-day change | +11.11%" in report
    assert "Latest vs prior 7-calendar-day mean (7 observations) | +66.67%" in report
    assert "2026-01-11" not in report
    assert "not an exchange candle volume" in report
    assert "replacement for the OHLCV Volume field" in report


@pytest.mark.unit
def test_coin_metrics_current_utc_day_is_not_a_completed_daily_cutoff():
    today = datetime.now(timezone.utc).date()
    expected = (today - timedelta(days=1)).isoformat()

    assert crypto_intelligence._completed_coin_metrics_cutoff(today.isoformat()) == expected
