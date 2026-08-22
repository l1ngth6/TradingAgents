"""Crypto daily-calendar behavior must not depend on the host timezone."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

import cli.main as cli_main
import tradingagents.dataflows.stockstats_utils as stockstats_utils
from cli.models import AssetType
from tradingagents.dataflows.config import set_config
from tradingagents.market_time import current_market_date, validate_analysis_date


@pytest.mark.unit
def test_crypto_market_date_rolls_at_midnight_utc():
    beijing = timezone(timedelta(hours=8))

    assert current_market_date(
        "crypto", now=datetime(2026, 8, 22, 7, 59, tzinfo=beijing)
    ) == date(2026, 8, 21)
    assert current_market_date(
        "crypto", now=datetime(2026, 8, 22, 8, 0, tzinfo=beijing)
    ) == date(2026, 8, 22)


@pytest.mark.unit
def test_non_crypto_market_date_keeps_host_local_calendar():
    beijing_wall_clock = datetime(
        2026, 8, 22, 7, 59, tzinfo=timezone(timedelta(hours=8))
    )

    assert current_market_date("stock", now=beijing_wall_clock) == date(2026, 8, 22)


@pytest.mark.unit
def test_crypto_rejects_date_after_current_utc_calendar_day():
    with pytest.raises(ValueError, match="UTC market date 2026-08-21"):
        validate_analysis_date(
            "2026-08-22", "crypto", market_today=date(2026, 8, 21)
        )

    assert validate_analysis_date(
        "2026-08-21", "crypto", market_today=date(2026, 8, 21)
    ) == date(2026, 8, 21)


@pytest.mark.unit
def test_cli_crypto_date_default_and_future_check_use_utc_day(monkeypatch):
    answers = iter(["2026-08-22", "2026-08-21"])
    defaults = []

    monkeypatch.setattr(cli_main, "current_market_date", lambda asset_type: date(2026, 8, 21))

    def fake_prompt(_message, *, default):
        defaults.append(default)
        return next(answers)

    monkeypatch.setattr(cli_main.typer, "prompt", fake_prompt)
    monkeypatch.setattr(cli_main.console, "print", lambda *_args, **_kwargs: None)

    assert cli_main.get_analysis_date(AssetType.CRYPTO) == "2026-08-21"
    assert defaults == ["2026-08-21", "2026-08-21"]


@pytest.mark.unit
def test_crypto_selection_prompt_explains_utc_date():
    guidance = cli_main._analysis_date_guidance(AssetType.CRYPTO)

    assert "UTC calendar date" in guidance
    assert "00:00 UTC" in guidance
    assert "08:00 Beijing time" in guidance


@pytest.mark.unit
def test_crypto_ohlcv_cache_window_uses_utc_today(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    captured = {}

    monkeypatch.setattr(
        stockstats_utils,
        "current_market_date",
        lambda asset_type: date(2026, 8, 21),
    )

    def fake_download(symbol, start, end, **kwargs):
        captured.update(symbol=symbol, start=start, end=end)
        return pd.DataFrame(
            {
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.0],
                "Volume": [1],
            },
            index=pd.DatetimeIndex(["2026-08-21"], name="Date"),
        )

    monkeypatch.setattr(stockstats_utils.yf, "download", fake_download)

    stockstats_utils.load_ohlcv("BTC-USD", "2026-08-21")

    assert captured["symbol"] == "BTC-USD"
    assert captured["start"] == "2021-08-21"
    assert captured["end"] == "2026-08-22"
