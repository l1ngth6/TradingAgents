"""Free-first crypto-native options, on-chain, and liquidation enrichment."""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from statistics import stdev

import requests

from .errors import NoMarketDataError, VendorNotConfiguredError, VendorRateLimitError
from .symbol_utils import crypto_base

REQUEST_TIMEOUT = 20
DERIBIT_BASE = "https://www.deribit.com/api/v2"
COIN_METRICS_BASE = "https://community-api.coinmetrics.io/v4"
COINALYZE_BASE = "https://api.coinalyze.net/v1"
DUNE_BASE = "https://api.dune.com/api/v1"
REPORTED_SPOT_VOLUME_METRIC = "volume_reported_spot_usd_1d"


def _get_json(url: str, *, params=None, headers=None) -> dict | list:
    response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    if response.status_code == 429:
        raise VendorRateLimitError(f"{urlparse_host(url)} returned HTTP 429")
    if response.status_code in {401, 403}:
        raise VendorNotConfiguredError(f"{urlparse_host(url)} rejected the configured credential")
    response.raise_for_status()
    return response.json()


def urlparse_host(url: str) -> str:
    return url.split("/", 3)[2]


def _base_currency(symbol: str) -> str:
    base = crypto_base(symbol)
    if not base:
        raise NoMarketDataError(symbol, symbol, "not a supported cryptocurrency pair")
    return base.upper()


def _as_float(value) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=64)
def get_deribit_options(symbol: str, analysis_as_of: str = "") -> str:
    """Summarize the public Deribit option surface without requiring a key."""
    currency = _base_currency(symbol)
    if currency not in {"BTC", "ETH"}:
        raise NoMarketDataError(symbol, currency, "Deribit public option coverage is BTC/ETH")
    as_of_dt = datetime.now(timezone.utc)
    if analysis_as_of:
        as_of_dt = datetime.fromisoformat(analysis_as_of.replace("Z", "+00:00"))
        if as_of_dt.tzinfo is None:
            as_of_dt = as_of_dt.replace(tzinfo=timezone.utc)
        as_of_dt = min(as_of_dt.astimezone(timezone.utc), datetime.now(timezone.utc))
    if as_of_dt.date() < datetime.now(timezone.utc).date():
        raise NoMarketDataError(
            symbol,
            currency,
            "Deribit's keyless book-summary endpoint is current-only; refusing to leak a current option surface into a historical task",
        )

    payload = _get_json(
        f"{DERIBIT_BASE}/public/get_book_summary_by_currency",
        params={"currency": currency, "kind": "option"},
    )
    rows = payload.get("result", []) if isinstance(payload, dict) else []
    if not rows:
        raise NoMarketDataError(symbol, currency, "Deribit returned no option summaries")

    realized = {}
    try:
        chart = _get_json(
            f"{DERIBIT_BASE}/public/get_tradingview_chart_data",
            params={
                "instrument_name": f"{currency}-PERPETUAL",
                "start_timestamp": int((as_of_dt - timedelta(days=45)).timestamp() * 1000),
                "end_timestamp": int(as_of_dt.timestamp() * 1000),
                "resolution": "1D",
            },
        )
        closes = (chart.get("result") or {}).get("close", []) if isinstance(chart, dict) else []
        closes = [float(value) for value in closes if _as_float(value) not in (None, 0)]
        log_returns = [math.log(b / a) for a, b in zip(closes, closes[1:], strict=False)]
        for days in (7, 30):
            sample = log_returns[-days:]
            if len(sample) >= 2:
                realized[days] = stdev(sample) * math.sqrt(365) * 100
    except Exception:
        realized = {}

    by_expiry: dict[str, list[dict]] = defaultdict(list)
    total_call_oi = total_put_oi = total_call_volume = total_put_volume = 0.0
    concentrations = []
    for row in rows:
        name = str(row.get("instrument_name", ""))
        parts = name.split("-")
        if len(parts) < 4:
            continue
        expiry, strike_text, option_type = parts[-3], parts[-2], parts[-1]
        strike = _as_float(strike_text)
        oi = _as_float(row.get("open_interest")) or 0.0
        volume = _as_float(row.get("volume")) or 0.0
        mark_iv = _as_float(row.get("mark_iv"))
        underlying = _as_float(row.get("underlying_price"))
        parsed = {"name": name, "strike": strike, "type": option_type, "oi": oi, "volume": volume,
                  "mark_iv": mark_iv, "underlying": underlying}
        by_expiry[expiry].append(parsed)
        concentrations.append(parsed | {"expiry": expiry})
        if option_type == "C":
            total_call_oi += oi
            total_call_volume += volume
        elif option_type == "P":
            total_put_oi += oi
            total_put_volume += volume

    term_rows = []
    for expiry, contracts in by_expiry.items():
        candidates = [r for r in contracts if r["strike"] and r["underlying"] and r["mark_iv"] is not None]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda r: abs(r["strike"] - r["underlying"]))
        atm_pair = [r["mark_iv"] for r in candidates if r["strike"] == nearest["strike"] and r["mark_iv"] is not None]
        atm_iv = sum(atm_pair) / len(atm_pair)
        term_rows.append((expiry, atm_iv, nearest["strike"], sum(r["oi"] for r in contracts)))
    try:
        term_rows.sort(key=lambda item: datetime.strptime(item[0], "%d%b%y"))
    except ValueError:
        term_rows.sort(key=lambda item: item[0])
    top_oi = sorted(
        (row for row in concentrations if row["strike"] is not None),
        key=lambda row: row["oi"],
        reverse=True,
    )[:10]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"## Deribit public options snapshot for {currency}", "",
        f"- Observed at: {now}", f"- Requested analysis-as-of: {analysis_as_of or 'latest'}",
        "- Source: https://www.deribit.com/ (public API, no authentication)",
        "- Nature: observed public book summaries plus derived aggregates",
        "- 25-delta risk reversal and dealer gamma exposure: DATA_UNAVAILABLE from this keyless aggregate endpoint; do not estimate them.",
        "- Max pain is intentionally omitted as a low-confidence auxiliary statistic.", "",
        "### Aggregate positioning", "", "| Metric | Value |", "|---|---:|",
        f"| Call open interest | {total_call_oi:,.2f} |",
        f"| Put open interest | {total_put_oi:,.2f} |",
        f"| Put/call OI ratio | {(total_put_oi / total_call_oi) if total_call_oi else float('nan'):.3f} |",
        f"| Call volume | {total_call_volume:,.2f} |",
        f"| Put volume | {total_put_volume:,.2f} |",
        f"| 7-day annualized realized volatility | {f'{realized[7]:.2f}%' if 7 in realized else 'DATA_UNAVAILABLE'} |",
        f"| 30-day annualized realized volatility | {f'{realized[30]:.2f}%' if 30 in realized else 'DATA_UNAVAILABLE'} |",
        "", "### ATM IV term structure", "", "| Expiry | ATM mark IV | ATM strike | Total OI |", "|---|---:|---:|---:|",
    ]
    for expiry, iv, strike, oi in term_rows[:12]:
        lines.append(f"| {expiry} | {iv:.2f}% | {strike:,.0f} | {oi:,.2f} |")
    lines += ["", "### Largest strike/expiry OI concentrations", "", "| Expiry | Strike | Type | OI | Mark IV |", "|---|---:|---|---:|---:|"]
    for row in top_oi:
        iv = "N/A" if row["mark_iv"] is None else f"{row['mark_iv']:.2f}%"
        lines.append(f"| {row['expiry']} | {row['strike']:,.0f} | {row['type']} | {row['oi']:,.2f} | {iv} |")
    return "\n".join(lines)


def _coin_metrics_rows(
    assets: str, metrics: str, start_time: str, end_time: str
) -> list[dict]:
    payload = _get_json(
        f"{COIN_METRICS_BASE}/timeseries/asset-metrics",
        params={
            "assets": assets,
            "metrics": metrics,
            "frequency": "1d",
            "start_time": start_time,
            "end_time": end_time,
            "page_size": 10000,
        },
    )
    return payload.get("data", []) if isinstance(payload, dict) else []


def _completed_coin_metrics_cutoff(end_date: str) -> str:
    """Return the last UTC day safe for completed daily aggregates.

    Coin Metrics may expose a row for the current UTC day before all venues have
    completed that day.  That row is useful as a live observation but is not
    comparable with complete 24-hour observations, so this integration excludes
    it from deterministic daily activity statistics.
    """
    requested = datetime.strptime(end_date, "%Y-%m-%d").date()
    yesterday_utc = datetime.now(timezone.utc).date() - timedelta(days=1)
    return min(requested, yesterday_utc).isoformat()


def _metric_observations(
    rows: list[dict], metric: str, end_date: str
) -> list[tuple[datetime, float]]:
    """Parse, deduplicate, and sort non-null daily Coin Metrics observations."""
    by_day: dict[str, tuple[datetime, float]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = _as_float(row.get(metric))
        raw_time = str(row.get("time", ""))
        if value is None or value < 0 or not raw_time:
            continue
        try:
            # Asset-metrics daily timestamps may contain nanosecond fractions,
            # which are not parsed consistently across all supported Python
            # versions. Only the UTC calendar day is relevant here.
            observed_at = datetime.strptime(raw_time[:10], "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        day = observed_at.date().isoformat()
        if day <= end_date:
            by_day[day] = (observed_at, value)
    return [by_day[day] for day in sorted(by_day)]


def _pct_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current / previous - 1) * 100


def _fmt_usd(value: float | None) -> str:
    return "N/A" if value is None else f"${value:,.2f}"


def _fmt_percent(value: float | None) -> str:
    return "N/A" if value is None else f"{value:+.2f}%"


def _reported_spot_activity_section(rows: list[dict], cutoff: str) -> str:
    """Render cross-market reported spot volume and conservative diagnostics."""
    observations = _metric_observations(rows, REPORTED_SPOT_VOLUME_METRIC, cutoff)
    if not observations:
        return (
            "### Cross-market reported spot activity\n\n"
            "DATA_UNAVAILABLE: Coin Metrics returned no completed, non-null "
            f"{REPORTED_SPOT_VOLUME_METRIC} observations through {cutoff}."
        )

    latest_time, latest = observations[-1]
    previous_time = previous = None
    daily_change = None
    if len(observations) >= 2:
        previous_time, previous = observations[-2]
        if (latest_time.date() - previous_time.date()).days == 1:
            daily_change = _pct_change(latest, previous)

    seven_start = latest_time.date() - timedelta(days=7)
    thirty_start = latest_time.date() - timedelta(days=30)
    seven_values = [
        value
        for observed_at, value in observations
        if seven_start <= observed_at.date() < latest_time.date()
    ]
    thirty_values = [
        value
        for observed_at, value in observations
        if thirty_start <= observed_at.date() < latest_time.date()
    ]
    seven_average = sum(seven_values) / len(seven_values) if seven_values else None
    thirty_average = sum(thirty_values) / len(thirty_values) if thirty_values else None
    versus_seven = (
        _pct_change(latest, seven_average) if seven_average not in (None, 0) else None
    )
    versus_thirty = (
        _pct_change(latest, thirty_average) if thirty_average not in (None, 0) else None
    )

    # Seven observations is a deliberately conservative minimum. A z-score or
    # percentile over two or three available days looks precise but is not useful.
    percentile = z_score = None
    if len(thirty_values) >= 7:
        percentile = (
            sum(value <= latest for value in thirty_values) / len(thirty_values) * 100
        )
        sample_stdev = stdev(thirty_values)
        if sample_stdev:
            z_score = (latest - thirty_average) / sample_stdev

    previous_label = (
        f"{previous_time.date().isoformat()} ({_fmt_usd(previous)})"
        if previous_time is not None
        else "N/A"
    )
    gap_note = ""
    if previous_time is not None and daily_change is None:
        gap_note = (
            "- Day-over-day change is unavailable because the two latest "
            "observations are not consecutive UTC days."
        )

    lines = [
        "### Cross-market reported spot activity (auxiliary)",
        "",
        "- Metric: `volume_reported_spot_usd_1d` — reported USD spot volume across "
        "Coin Metrics' covered centralized and decentralized markets.",
        f"- Completed-daily cutoff: {cutoff}; the current UTC day is excluded from "
        "daily comparisons even if the API exposes a partial row.",
        "- Data quality: reported volume is not Coin Metrics' paid trusted-volume "
        "metric and can include low-quality or wash-traded venue activity.",
        "- Interpretation: use only as a broad participation/activity cross-check. "
        "It is not an exchange candle volume, order flow, capital inflow, or a "
        "replacement for the OHLCV Volume field used by VWMA/MFI/OBV.",
        gap_note,
        "",
        "| Derived activity measure | Value |",
        "|---|---:|",
        f"| Latest completed observation | {latest_time.date().isoformat()} ({_fmt_usd(latest)}) |",
        f"| Previous available observation | {previous_label} |",
        f"| Consecutive-day change | {_fmt_percent(daily_change)} |",
        f"| Latest vs prior 7-calendar-day mean ({len(seven_values)} observations) | {_fmt_percent(versus_seven)} |",
        f"| Latest vs prior 30-calendar-day mean ({len(thirty_values)} observations) | {_fmt_percent(versus_thirty)} |",
        f"| Percentile versus prior 30 days | {f'{percentile:.1f}%' if percentile is not None else 'N/A (requires at least 7 prior observations)'} |",
        f"| Z-score versus prior 30 days | {f'{z_score:+.2f}' if z_score is not None else 'N/A (requires at least 7 prior observations and nonzero variance)'} |",
        "",
        "#### Recent completed observations",
        "",
        "| Day (UTC) | Reported spot volume USD |",
        "|---|---:|",
    ]
    for observed_at, value in observations[-7:]:
        lines.append(f"| {observed_at.date().isoformat()} | {_fmt_usd(value)} |")
    return "\n".join(lines)


def _dune_latest_result(query_id: str) -> list[dict]:
    key = os.getenv("DUNE_API_KEY", "").strip()
    if not key:
        raise VendorNotConfiguredError("DUNE_API_KEY is not configured")
    payload = _get_json(
        f"{DUNE_BASE}/query/{query_id}/results",
        params={"limit": 1000},
        headers={"X-Dune-API-Key": key},
    )
    result = payload.get("result", {}) if isinstance(payload, dict) else {}
    return result.get("rows", []) if isinstance(result, dict) else []


def _filter_dune_rows(rows: list[dict], end_date: str) -> tuple[list[dict], bool]:
    """Best-effort cutoff filter for user-defined Dune result schemas."""
    cutoff = datetime.strptime(end_date, "%Y-%m-%d").date()
    filtered = []
    found_timestamp = False
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp_value = next(
            (
                value
                for key, value in row.items()
                if str(key).lower() in {"date", "day", "time", "timestamp", "block_date", "block_time"}
            ),
            None,
        )
        if timestamp_value is None:
            continue
        found_timestamp = True
        try:
            if isinstance(timestamp_value, (int, float)):
                unix_value = timestamp_value / 1000 if timestamp_value > 10_000_000_000 else timestamp_value
                row_date = datetime.fromtimestamp(unix_value, tz=timezone.utc).date()
            else:
                row_date = datetime.fromisoformat(
                    str(timestamp_value).replace("Z", "+00:00")
                ).date()
        except (TypeError, ValueError, OSError):
            continue
        if row_date <= cutoff:
            filtered.append(row)
    return filtered, found_timestamp


@lru_cache(maxsize=64)
def get_crypto_onchain(symbol: str, start_date: str, end_date: str) -> str:
    """Return free Coin Metrics activity/network context plus configured Dune data."""
    base = _base_currency(symbol).lower()
    coin_metrics_cutoff = _completed_coin_metrics_cutoff(end_date)
    sections = []
    successes = []
    failures = []
    try:
        rows = _coin_metrics_rows(
            base,
            f"SplyCur,CapMrktCurUSD,TxCnt,AdrActCnt,{REPORTED_SPOT_VOLUME_METRIC}",
            start_date,
            end_date,
        )
        rows = [
            row
            for row in rows
            if isinstance(row, dict)
            and str(row.get("time", ""))[:10] <= coin_metrics_cutoff
        ]
        rows.sort(key=lambda row: str(row.get("time", "")))
        latest = rows[-1] if rows else None
        if latest:
            successes.append("Coin Metrics Community")
            sections.append(
                "### Coin Metrics Community network metrics (latest completed day)\n\n"
                "| Time | Current supply | Market cap USD | Tx count | Active addresses |\n|---|---:|---:|---:|---:|\n"
                f"| {latest.get('time')} | {latest.get('SplyCur', 'N/A')} | {latest.get('CapMrktCurUSD', 'N/A')} | "
                f"{latest.get('TxCnt', 'N/A')} | {latest.get('AdrActCnt', 'N/A')} |"
            )
            sections.append(_reported_spot_activity_section(rows, coin_metrics_cutoff))
        else:
            failures.append("Coin Metrics Community: no rows in requested window")
    except Exception as exc:  # optional source must degrade independently
        failures.append(f"Coin Metrics Community: {exc}")

    try:
        stable_rows = _coin_metrics_rows("usdt,usdc", "SplyCur", start_date, end_date)
        stable_rows = [
            row
            for row in stable_rows
            if isinstance(row, dict)
            and str(row.get("time", ""))[:10] <= coin_metrics_cutoff
        ]
        stable_rows.sort(
            key=lambda row: (str(row.get("asset", "")), str(row.get("time", "")))
        )
        if stable_rows:
            successes.append("Coin Metrics stablecoin supply")
            latest_by_asset = {}
            first_by_asset = {}
            for row in stable_rows:
                asset = row.get("asset", "unknown")
                first_by_asset.setdefault(asset, row)
                latest_by_asset[asset] = row
            stable_lines = ["### Stablecoin supply change", "", "| Asset | Start supply | Latest supply | Change |", "|---|---:|---:|---:|"]
            for asset, latest in latest_by_asset.items():
                first = first_by_asset[asset]
                start = _as_float(first.get("SplyCur"))
                finish = _as_float(latest.get("SplyCur"))
                change = (finish - start) if None not in (start, finish) else None
                stable_lines.append(f"| {asset.upper()} | {start if start is not None else 'N/A'} | {finish if finish is not None else 'N/A'} | {change if change is not None else 'N/A'} |")
            sections.append("\n".join(stable_lines))
        else:
            failures.append("Coin Metrics stablecoin supply: no rows")
    except Exception as exc:
        failures.append(f"Coin Metrics stablecoin supply: {exc}")

    for label, env_name in (("Configured on-chain query", "DUNE_CRYPTO_ONCHAIN_QUERY_ID"), ("Configured ETF-flow query", "DUNE_CRYPTO_ETF_QUERY_ID")):
        query_id = os.getenv(env_name, "").strip()
        if not query_id:
            failures.append(f"{label}: DATA_UNAVAILABLE ({env_name} not configured)")
            continue
        if not query_id.isdigit():
            failures.append(f"{label}: DATA_UNAVAILABLE ({env_name} must be numeric)")
            continue
        try:
            rows = _dune_latest_result(query_id)
            filtered, timestamped = _filter_dune_rows(rows, end_date)
            historical = end_date < datetime.now(timezone.utc).date().isoformat()
            if timestamped:
                rows = filtered
            elif historical:
                failures.append(
                    f"{label}: latest result has no recognized timestamp column; refused for historical cutoff"
                )
                continue
            successes.append(f"Dune query {query_id}")
            sections.append(f"### {label} (Dune query {query_id})\n\n```json\n{json.dumps(rows[:50], ensure_ascii=False, indent=2)}\n```")
        except Exception as exc:
            failures.append(f"{label}: {exc}")

    header = [
        f"## Free-first crypto network and market-activity context for {base.upper()}", "",
        f"- Requested window: {start_date} through {end_date}",
        f"- Coin Metrics completed-daily cutoff: {coin_metrics_cutoff} (current UTC day excluded)",
        f"- Retrieved at: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "- Primary source: https://coinmetrics.io/ (Community API, no authentication)",
        f"- Successful sources: {', '.join(successes) if successes else 'none'}",
        f"- Failed/unavailable sources: {'; '.join(failures) if failures else 'none'}",
        "- Exchange netflows and whale-wallet alerts are DATA_UNAVAILABLE unless a configured Dune query supplies a licensed definition. Do not infer them from supply metrics.",
    ]
    return "\n".join(header + [""] + sections)


@lru_cache(maxsize=64)
def get_coinalyze_liquidations(symbol: str, start_time: str, end_time: str, interval: str = "1hour") -> str:
    """Return actual historical liquidations; this is not a latent heatmap."""
    key = os.getenv("COINALYZE_API_KEY", "").strip()
    if not key:
        raise VendorNotConfiguredError("COINALYZE_API_KEY is not configured")
    base = _base_currency(symbol)
    coinalyze_symbol = os.getenv("COINALYZE_SYMBOL_OVERRIDE", f"{base}USDT_PERP.A")
    start_ts = int(datetime.fromisoformat(start_time.replace("Z", "+00:00")).timestamp())
    end_ts = int(datetime.fromisoformat(end_time.replace("Z", "+00:00")).timestamp())
    payload = _get_json(
        f"{COINALYZE_BASE}/liquidation-history",
        params={
            "symbols": coinalyze_symbol,
            "interval": interval,
            "from": start_ts,
            "to": end_ts,
            "convert_to_usd": "true",
        },
        headers={"api_key": key},
    )
    histories = payload if isinstance(payload, list) else []
    rows = histories[0].get("history", []) if histories and isinstance(histories[0], dict) else []
    if not rows:
        raise NoMarketDataError(symbol, coinalyze_symbol, "Coinalyze returned no liquidation rows")
    lines = [
        f"## Coinalyze actual liquidation history for {coinalyze_symbol}", "",
        f"- Window: {start_time} through {end_time}",
        "- Source: https://coinalyze.net/ (free API key; USD-converted values)",
        "- Nature: observed/aggregated historical liquidations, not predicted liquidation levels or a heatmap.", "",
        "| Time (UTC) | Long liquidations | Short liquidations | Imbalance (long-short) |", "|---|---:|---:|---:|",
    ]
    for row in rows[-168:]:
        if isinstance(row, dict):
            timestamp = row.get("t")
            long_value = _as_float(row.get("l")) or 0.0
            short_value = _as_float(row.get("s")) or 0.0
        elif isinstance(row, (list, tuple)) and len(row) >= 3:
            timestamp = row[0]
            long_value = _as_float(row[1]) or 0.0
            short_value = _as_float(row[2]) or 0.0
        else:
            continue
        if timestamp is None:
            continue
        lines.append(f"| {datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat(timespec='minutes')} | {long_value:,.2f} | {short_value:,.2f} | {long_value - short_value:+,.2f} |")
    return "\n".join(lines)
