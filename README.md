<p align="center">
  <img src="assets/TauricResearch.png" style="width: 60%; height: auto;">
</p>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2412.20138" target="_blank"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.20138-B31B1B?logo=arxiv"/></a>
  <a href="https://discord.com/invite/hk9PGKShPK" target="_blank"><img alt="Discord" src="https://img.shields.io/badge/Discord-TradingResearch-7289da?logo=discord&logoColor=white&color=7289da"/></a>
  <a href="https://x.com/TauricResearch" target="_blank"><img alt="X Follow" src="https://img.shields.io/badge/X-TauricResearch-white?logo=x&logoColor=white"/></a>
  <a href="https://github.com/TauricResearch/" target="_blank"><img alt="Community" src="https://img.shields.io/badge/GitHub_Community-TauricResearch-14C290?logo=discourse"/></a>
</div>
<br>
<div align="center">
  <a href="https://github.com/TauricResearch" target="_blank"><img alt="TradingAgents #1 Repository of the Day" src="https://trendshift.io/api/badge/repositories/16192" width="250" height="55"/></a>
</div>
<br>
<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=de">Deutsch</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=es">Español</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=fr">français</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ja">日本語</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ko">한국어</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=pt">Português</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=ru">Русский</a> | 
  <a href="https://www.readme-i18n.com/TauricResearch/TradingAgents?lang=zh">中文</a>
</div>

---

# TradingAgents: Multi-Agents LLM Financial Trading Framework

## News
- [2026-07] **TradingAgents v0.3.1** released with correctness and stability fixes: Alpha Vantage look-ahead filtering, graph-router crash-safety, graph-shape-aware checkpoint resume, working crypto sentiment sources, a configurable LLM retry budget, Bedrock API-key auth, and Claude Sonnet 5 / Fable 5 support. See [CHANGELOG.md](CHANGELOG.md) for the full list.
- [2026-06] **TradingAgents v0.3.0** released with a verified data-access contract, an expanded provider registry (NVIDIA, Kimi, Groq, Mistral, Bedrock, and any OpenAI-compatible endpoint), FRED and Polymarket data vendors, a current-generation model catalog, and a CI gate.
- [2026-05] **TradingAgents v0.2.5** released with the grounded Sentiment Analyst, GPT-5.5 etc. model coverage, Qwen/GLM/MiniMax dual-region support, `TRADINGAGENTS_*` env-var configurability with API-key auto-detection, remote Ollama support, non-US alpha benchmarks, and ticker path-traversal hardening.
- [2026-04] **TradingAgents v0.2.4** released with structured-output agents (Research Manager, Trader, Portfolio Manager), LangGraph checkpoint resume, persistent decision log, DeepSeek/Qwen/GLM/Azure provider support, Docker, and a Windows UTF-8 encoding fix.
- [2026-03] **TradingAgents v0.2.3** released with multi-language support, GPT-5.4 family models, unified model catalog, backtesting date fidelity, and proxy support.
- [2026-03] **TradingAgents v0.2.2** released with GPT-5.4/Gemini 3.1/Claude 4.6 model coverage, five-tier rating scale, OpenAI Responses API, Anthropic effort control, and cross-platform stability.
- [2026-02] **TradingAgents v0.2.0** released with multi-provider LLM support (GPT-5.x, Gemini 3.x, Claude 4.x, Grok 4.x) and improved system architecture.
- [2026-01] **Trading-R1** [Technical Report](https://arxiv.org/abs/2509.11420) released, with [Terminal](https://github.com/TauricResearch/Trading-R1) expected to land soon.

<div align="center">

🚀 [TradingAgents](#tradingagents-framework) | ⚡ [Installation & CLI](#installation-and-cli) | 🎬 [Demo](https://www.youtube.com/watch?v=90gr5lwjIho) | 📦 [Package Usage](#tradingagents-package) | 🤝 [Contributing](#contributing) | 📄 [Citation](#citation)

</div>

> 🎉 **TradingAgents** officially released! We have received numerous inquiries about the work, and we would like to express our thanks for the enthusiasm in our community.
>
> So we decided to fully open-source the framework. Looking forward to building impactful projects with you!

## TradingAgents Framework

TradingAgents is a multi-agent trading framework that mirrors the dynamics of real-world trading firms. By deploying specialized LLM-powered agents: from fundamental analysts, sentiment experts, and technical analysts, to trader, risk management team, the platform collaboratively evaluates market conditions and informs trading decisions. Moreover, these agents engage in dynamic discussions to pinpoint the optimal strategy.

<p align="center">
  <img src="assets/schema.png" style="width: 100%; height: auto;">
</p>

> TradingAgents framework is designed for research purposes. Trading performance may vary based on many factors, including the chosen backbone language models, model temperature, trading periods, the quality of data, and other non-deterministic factors. [It is not intended as financial, investment, or trading advice.](https://tauric.ai/disclaimer/)

Our framework decomposes complex trading tasks into specialized roles.

### Analyst Team
- Fundamentals Analyst: Evaluates company financials and performance metrics, identifying intrinsic values and potential red flags.
- Sentiment Analyst: Aggregates news headlines, StockTwits, and Reddit chatter into a single sentiment read to gauge short-term market mood, with optional lower-weight X Search evidence through xAI or a compatible Responses gateway when explicitly enabled.
- News Analyst: Monitors global news and macroeconomic indicators, interpreting the impact of events on market conditions.
- Technical Analyst: Utilizes technical indicators (like MACD and RSI) to detect trading patterns and forecast price movements.

<p align="center">
  <img src="assets/analyst.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

### Researcher Team
- Comprises both bullish and bearish researchers who critically assess the insights provided by the Analyst Team. Through structured debates, they balance potential gains against inherent risks.

<p align="center">
  <img src="assets/researcher.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Trader Agent
- Composes reports from the analysts and researchers to make informed trading decisions, determining the timing and magnitude of trades.

<p align="center">
  <img src="assets/trader.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

### Risk Management and Portfolio Manager
- Continuously evaluates portfolio risk by assessing market volatility, liquidity, and other risk factors. The risk management team evaluates and adjusts trading strategies, providing assessment reports to the Portfolio Manager for final decision.
- The Portfolio Manager approves/rejects the transaction proposal. If approved, the order will be sent to the simulated exchange and executed.

<p align="center">
  <img src="assets/risk.png" width="70%" style="display: inline-block; margin: 0 2%;">
</p>

## Installation and CLI

### Installation

Clone TradingAgents:
```bash
git clone https://github.com/TauricResearch/TradingAgents.git
cd TradingAgents
```

Create a virtual environment in any of your favorite environment managers:
```bash
conda create -n tradingagents python=3.12
conda activate tradingagents
```

Install the package and its dependencies:
```bash
pip install .
```

### Docker

Alternatively, run with Docker:
```bash
cp .env.example .env  # add your API keys
APP_UID="$(id -u)" APP_GID="$(id -g)" docker compose run --rm tradingagents
```

On Linux, the command above builds the container user with the current host
user's UID and GID. Reports are persisted in the host's `./reports` directory.
If that directory does not exist, Docker may create it as root; the container
repairs its ownership before dropping privileges and starting TradingAgents.

For local models with Ollama:
```bash
docker compose --profile ollama run --rm tradingagents-ollama
```

### Required APIs

TradingAgents supports multiple LLM providers. Set the API key for your chosen provider:

```bash
export OPENAI_API_KEY=...          # OpenAI (GPT)
export GOOGLE_API_KEY=...          # Google (Gemini)
export ANTHROPIC_API_KEY=...       # Anthropic (Claude)
export XAI_API_KEY=...             # xAI (Grok)
export DEEPSEEK_API_KEY=...        # DeepSeek
export DASHSCOPE_API_KEY=...       # Qwen — International (dashscope-intl.aliyuncs.com)
export DASHSCOPE_CN_API_KEY=...    # Qwen — China (dashscope.aliyuncs.com)
export ZHIPU_API_KEY=...           # GLM via Z.AI (international)
export ZHIPU_CN_API_KEY=...        # GLM via BigModel (China, open.bigmodel.cn)
export MINIMAX_API_KEY=...         # MiniMax — Global (api.minimax.io)
export MINIMAX_CN_API_KEY=...      # MiniMax — China (api.minimaxi.com)
export OPENROUTER_API_KEY=...      # OpenRouter
export ALPHA_VANTAGE_API_KEY=...   # Alpha Vantage
```

For Azure OpenAI, copy `.env.enterprise.example` to `.env.enterprise` and fill in your credentials.

For AWS Bedrock, install the extra with `pip install ".[bedrock]"`, set `llm_provider: "bedrock"`, configure AWS credentials (environment variables, `~/.aws/credentials`, or an IAM role) and `AWS_DEFAULT_REGION`, and use a Bedrock model ID, e.g. `us.anthropic.claude-opus-4-8-v1:0`.

For local models, configure Ollama with `llm_provider: "ollama"`. The default endpoint is `http://localhost:11434/v1`; set `OLLAMA_BASE_URL` to point at a remote `ollama-serve`. Pull models with `ollama pull <name>`, and pick "Custom model ID" in the CLI for any model not listed by default.

For an OpenAI-compatible server that implements the Responses API, use `llm_provider: "openai_compatible"` and set the endpoint via `backend_url` (or `TRADINGAGENTS_LLM_BACKEND_URL`), e.g. `http://192.168.2.221:8080/v1`. Both the deep-thinking and quick-thinking models use that same base URL and call `/v1/responses`; the client does not fall back to `/v1/chat/completions`. The model IDs are whatever your server exposes. No key is needed for keyless local servers; set `OPENAI_COMPATIBLE_API_KEY` when the endpoint requires one. `TRADINGAGENTS_OPENAI_REASONING_EFFORT` is optional and is forwarded for any model ID, including `gpt-5.6-sol` and `deepseek-v4-*`.

Per-agent model overrides are optional and disabled by default. Set `TRADINGAGENTS_AGENT_LLM_OVERRIDES_ENABLED=true`, then provide a JSON object in `TRADINGAGENTS_AGENT_LLM_OVERRIDES`; see `.env.example` for every supported role and a complete example. Each configured role may override `model`, generic `thinking_level`, or both. Missing fields inherit the role's existing Quick/Deep defaults, while `thinking_level: null` uses the provider default. All role overrides continue to use the globally selected provider, credentials, backend URL, retry budget, temperature, and callbacks.

Alternatively, copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

### CLI Usage

Launch the interactive CLI:
```bash
tradingagents          # installed command
python -m cli.main     # alternative: run directly from source
```
You will see a screen where you can select your desired tickers, analysis date, LLM provider, research depth, and more.

### Markets and tickers

TradingAgents works with any market Yahoo Finance covers, using the exchange-suffixed ticker. Company identity and the alpha benchmark resolve automatically per market.

- US: `AAPL`, `SPY`
- Hong Kong: `0700.HK` · Tokyo: `7203.T` · London: `AZN.L`
- India: `RELIANCE.NS`, `.BO` · Canada: `.TO` · Australia: `.AX`
- China A-shares: Shanghai `.SS`, Shenzhen `.SZ` (e.g. `600519.SS` for Kweichow Moutai)
- Crypto: `BTC-USD`, `ETH-USD`

<p align="center">
  <img src="assets/cli/cli_init.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

An interface will appear showing results as they load, letting you track the agent's progress as it runs.

<p align="center">
  <img src="assets/cli/cli_news.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

<p align="center">
  <img src="assets/cli/cli_transaction.png" width="100%" style="display: inline-block; margin: 0 2%;">
</p>

## TradingAgents Package

### Implementation Details

We built TradingAgents with LangGraph to ensure flexibility and modularity. The framework supports multiple LLM providers: OpenAI, Google, Anthropic, xAI, DeepSeek, Qwen (Alibaba DashScope, international and China endpoints), GLM (Zhipu), MiniMax (global + China), OpenRouter, Ollama for local models, and Azure OpenAI for enterprise.

### Python Usage

To use TradingAgents inside your code, you can import the `tradingagents` module and initialize a `TradingAgentsGraph()` object. The `.propagate()` function will return a decision. You can run `main.py`, here's also a quick example:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

ta = TradingAgentsGraph(debug=True, config=DEFAULT_CONFIG.copy())

# forward propagate
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

You can also adjust the default configuration to set your own choice of LLMs, debate rounds, etc.

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"        # e.g. openai, google, anthropic, deepseek, groq, ollama; openai_compatible targets a custom Responses API endpoint
config["deep_think_llm"] = "gpt-5.5"     # Model for complex reasoning
config["quick_think_llm"] = "gpt-5.4-mini" # Model for quick tasks
config["max_debate_rounds"] = 2

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
print(decision)
```

See `tradingagents/default_config.py` for all configuration options.

### Cryptocurrency-native data

Crypto runs keep Yahoo Finance as the verified completed-daily-OHLCV and
technical-indicator source. A single optional **Crypto Intelligence Analyst**
owns the crypto-native cross-check so the core analysts do not duplicate work.
At task initialization the CLI asks for `Disabled`, `Shadow`, or `Advisory`:

- Shadow (recommended initially) saves the independent report but keeps it out
  of the bull/bear and portfolio decision path.
- Advisory passes only a short cross-validation summary downstream; it is never
  a standalone direction signal.
- Disabled skips the crypto-native agent and the optional heatmap prompt.

The free-first sources are:

- Binance USD-M futures: funding-rate history, open interest, global long/short
  account ratio, taker-flow/CVD proxy, and current futures depth imbalance for
  supported USDT perpetuals.
- Deribit public API: BTC/ETH ATM IV term structure, put/call positioning,
  strike/expiry concentration, and implied-versus-realized volatility context.
- Coin Metrics Community: basic network and stablecoin-supply metrics plus
  `volume_reported_spot_usd_1d`, used as a completed-daily cross-market spot
  activity reference. Optional Dune query IDs can supply user-defined licensed
  on-chain or ETF-flow datasets.
- Coinalyze (optional free API key): actual long/short liquidation history. It
  is explicitly not treated as a predicted liquidation heatmap.
- Alternative.me: the historical Crypto Fear & Greed Index, labeled as a
  Bitcoin-centric broad-market proxy rather than coin-specific sentiment.

#### Cross-market activity versus exchange OHLCV volume

Coin Metrics Community requires no API key. For each crypto run, the project
requests `volume_reported_spot_usd_1d`, the reported USD spot volume across Coin
Metrics' covered centralized and decentralized markets. The deterministic data
layer derives the consecutive-day change, the latest value relative to the prior
7- and 30-calendar-day means, and a percentile and sample z-score against the prior
30 days. It also exposes the latest seven completed observations. Even weekly
decision runs request at least 31 days of this lightweight context; percentile and
z-score remain unavailable until at least seven prior observations exist.

This series is deliberately labeled **cross-market reported spot activity**, not
"the" exact global volume. It is broader than a single exchange but is not the
paid Coin Metrics trusted-volume series, so it can include low-quality venues or
wash trading. It does not measure exchange netflow, aggressive buyer/seller flow,
or new capital entering the asset.

Most importantly, the aggregate series remains isolated inside the Crypto
Intelligence report. It never replaces the market-data provider's `Volume`
column and is never fed into VWMA, MFI, OBV, candlestick, or breakout-volume
calculations. Those indicators must retain the same venue/vendor and candle
boundaries as their OHLC prices. The current UTC day's Coin Metrics daily row is
treated as incomplete and excluded from all daily comparisons, even if the API
already exposes a partial value. This makes the series suitable for checking
whether a move has broad participation without silently mixing incompatible
volume definitions.

#### Optional Coinalyze and Dune configuration

These authenticated sources are optional. Missing credentials, exhausted
credits, stale query results, or an unavailable endpoint are reported in the
Crypto Intelligence report and do not abort the main workflow.

**Coinalyze** supplies observed long/short liquidation history, not predicted
liquidation levels. TradingAgents requests hourly, USD-converted liquidation
rows for the selected perpetual symbol and currently includes the latest 168
rows (about seven days) in the analyst context. This is most useful for checking
whether a recent move actually produced a leveraged long- or short-liquidation
cascade; it is not a standalone directional signal.

1. Sign in or create a Coinalyze account, then generate a key on the
   [Coinalyze API key page](https://coinalyze.net/account/api-key/). The
   [official API documentation](https://api.coinalyze.net/v1/doc/) describes
   the free API and its current limits.
2. Set `COINALYZE_API_KEY` in `.env`.
3. The default symbol is derived as `<BASE>USDT_PERP.A`, for example
   `BTCUSDT_PERP.A`. Set `COINALYZE_SYMBOL_OVERRIDE` when the desired instrument
   uses another symbol from Coinalyze's supported-futures list.

**Dune** is deliberately user-defined. TradingAgents does not create or execute
SQL and does not provide a permanent built-in query. It only retrieves the
latest already-executed result for up to two query IDs:

- `DUNE_CRYPTO_ONCHAIN_QUERY_ID`: preferably exchange inflow/outflow/netflow or
  another clearly defined on-chain capital-flow series.
- `DUNE_CRYPTO_ETF_QUERY_ID`: preferably daily BTC/ETH spot-ETF flow data.

Create a Dune account from the
[Dune APIs & Connectors page](https://dune.com/apis-and-connectors), then follow
the [Dune API authentication instructions](https://docs.dune.com/api-reference/overview/authentication)
(`Settings` -> `API` -> `Create new API key`). A read-only key is sufficient for
the result endpoint used here. Retrieving results may consume Dune credits, but
this integration does not trigger query execution.

Public queries can be starting points, but they are third-party artifacts whose
SQL, schema, refresh schedule, parameters, result ordering, or availability can
change. Inspect and preferably fork them before enabling them.

> **Current recommendation (reviewed 2026-08-22): leave Dune disabled by
> default.** The public queries reviewed for this integration were not reliable
> enough to recommend as ready-to-use sources: query `6450054` returns dataset
> coverage metadata rather than daily netflow; query `3729167` is stale and has
> no current result; query `6946196` has the desired daily CEX-netflow shape but
> currently fails because of an address-type mismatch; and query `6648506` has a
> compatible ETF-flow schema but its cached result was stale, parameter-dependent,
> and not reliably date-sorted. Leaving `DUNE_API_KEY` and both query IDs unset is
> fully supported and does not weaken the main analysis pipeline. Coin Metrics
> remains the default lightweight on-chain context.

If Dune is enabled later, fork and maintain the query under your own account,
verify that its latest execution succeeds and is recent, fix its parameters, and
make the final result compact and explicitly sorted newest-first. Do not treat a
public query ID as a maintained project default merely because its result schema
is currently compatible.

For reliable historical filtering, each query should return a timestamp column
named `date`, `day`, `time`, `timestamp`, `block_date`, or `block_time`. Prefer
`ORDER BY date DESC` and compact result columns such as:

```text
# On-chain flow
date, asset, exchange, inflow_usd, outflow_usd, netflow_usd

# ETF flow
date, fund, inflow_usd, outflow_usd, netflow_usd
```

The project retrieves at most 1,000 rows and exposes at most 50 filtered rows to
the model. A historical run rejects a Dune result that has no recognized
timestamp column. Supplying only `DUNE_API_KEY` without a query ID does not add
any Dune evidence. The integration reads only the most recently cached execution;
it does not run or refresh the query, choose its parameters, verify freshness, or
repair its SQL. Dune evidence therefore remains optional cross-validation rather
than a primary signal.

The CLI can also accept a local liquidation-heatmap image or public HTTPS image
URL. The selected Crypto Intelligence model reads it once in a dedicated,
tool-free multimodal request with `detail=original`. Only the resulting text,
tagged `estimated_visual_extraction`, enters the subsequent numeric/tool-based
analysis; the raw image is not carried through that loop. The image is hashed
and copied into the saved report and can only cross-check numeric sources. It
cannot independently alter the rating.

Historical-capable sources are capped at the requested analysis date. Binance retains some
positioning series for only a limited recent window, so an older analysis may
show funding data while marking open-interest or ratio history unavailable. A
current-only option surface is refused during historical tasks. A network
error, regional restriction, rate limit, missing optional key, or unsupported future
degrades these optional sections without aborting the main analysis. Configure
the vendors through the `data_vendors.crypto_*` entries. Binance, Deribit, Coin
Metrics Community, and Alternative.me require no key; see `.env.example` for
optional Coinalyze and Dune credentials.
The CLI detects crypto pairs automatically; programmatic callers should pass
`asset_type="crypto"` to `propagate()`:

```python
state, decision = ta.propagate(
    "BTC-USD",
    "2026-08-22",
    asset_type="crypto",
    decision_horizon="monthly",             # weekly | monthly | strategic
    crypto_intelligence_mode="shadow",      # disabled | shadow | advisory
    heatmap_input="https://example.com/heatmap.webp",  # optional local path or HTTPS
    portfolio_context={"current_position": "unknown", "max_drawdown": "5%"},
)
```

Crypto analysis uses two explicit cutoffs. `analysis_as_of` allows the latest
UTC date and live news/quotes/crypto-native context, while
`completed_daily_candle_date` is the most recent fully closed 00:00-00:00 UTC
candle. Every SMA/EMA/RSI/MACD/Bollinger/ATR/volume/candlestick calculation is
clamped to the latter, so a live price is never labeled as a daily close or a
confirmed breakout. The CLI default and future-date validation follow UTC
(08:00 in Beijing) rather than the host timezone.

Because crypto trades 24/7, its completed-candle cutoff must have an exact daily
row. A cache created shortly after 00:00 UTC may not yet contain the vendor's
newly published prior-day candle; such an incomplete cache becomes refreshable
after the bounded cache TTL even though the requested date is technically
historical. If a refresh still cannot obtain that exact row, the market tools
report the candle as unavailable instead of silently substituting an older day or
mislabeling the date as a weekend/holiday. Equity and ETF analysis retains the
normal previous-session fallback for genuine non-trading days.

## Persistence and Recovery

TradingAgents persists two kinds of state across runs.

### Decision log

The decision log is always on. Each completed run appends its decision to `~/.tradingagents/memory/trading_memory.md`. On the next run for the same ticker, TradingAgents fetches the realised return (raw and alpha vs SPY), generates a one-paragraph reflection, and injects the most recent same-ticker decisions plus recent cross-ticker lessons into the Portfolio Manager prompt, so each analysis carries forward what worked and what didn't.

Override the path with `TRADINGAGENTS_MEMORY_LOG_PATH`.

### Checkpoint resume

Checkpoint resume is opt-in via `--checkpoint`. When enabled, LangGraph saves state after each node so a crashed or interrupted run resumes from the last successful step instead of starting over. On a resume run you will see `Resuming from step N for <TICKER> on <date>` in the logs; on a new run you will see `Starting fresh`. Checkpoints are cleared automatically on successful completion.

Per-ticker SQLite databases live at `~/.tradingagents/cache/checkpoints/<TICKER>.db` (override the base with `TRADINGAGENTS_CACHE_DIR`). Use `--clear-checkpoints` to reset all of them before a run.

```bash
tradingagents analyze --checkpoint           # enable for this run
tradingagents analyze --clear-checkpoints    # reset before running
```

```python
config = DEFAULT_CONFIG.copy()
config["checkpoint_enabled"] = True
ta = TradingAgentsGraph(config=config)
_, decision = ta.propagate("NVDA", "2026-01-15")
```

## Reproducibility

TradingAgents is LLM-driven, so two runs of the same ticker and date can differ. This is expected for a research tool built on language models, not a defect. The variation comes from a few distinct sources, and it helps to separate them.

Language model sampling is non-deterministic. Even at a fixed temperature, providers do not guarantee byte-identical output across calls, and reasoning models (the default GPT-5.x family, and any thinking-mode model) vary the most because their internal reasoning is itself sampled.

Live data moves. News, StockTwits, Reddit, and optional X Search return different content as time passes, so a run today sees different inputs than a run last week even for the same historical trade date. Pin the analysis date to hold the price, indicator, Binance derivatives, and Crypto Fear & Greed windows fixed, but the social and news sources still reflect "now".

To reduce variation you can lower the sampling temperature. Set `temperature` in your config (or `TRADINGAGENTS_TEMPERATURE` in `.env`); lower values make models that honor it more repeatable. The current curated models are reasoning-first and largely ignore temperature, so for tighter reproducibility use a non-reasoning model, which you can set explicitly via the Custom model ID option.

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "openai"
config["temperature"] = 0.0
# Reasoning models ignore temperature. For tighter reproducibility, set a
# non-reasoning deep/quick model explicitly (e.g. via the Custom model ID option).
```

What does not vary anymore: the analyzed company identity is resolved deterministically from the ticker before any agent runs, and the market analyst grounds exact price and indicator claims in a verified data snapshot. Earlier reports of "different companies" or fabricated price levels across runs are addressed by these two mechanisms.

Backtest results are not guaranteed to match any published figure. Returns depend on the model, the temperature, the date range, data quality, and the sampling above. Treat the framework as a research scaffold for studying multi-agent analysis, not as a strategy with a fixed, replicable return.

## Contributing

Contributions are welcome: bug fixes, documentation, and feature ideas; past contributions are credited per release in [`CHANGELOG.md`](CHANGELOG.md).

## Citation

Please reference our work if you find *TradingAgents* provides you with some help :)

```
@misc{xiao2025tradingagentsmultiagentsllmfinancial,
      title={TradingAgents: Multi-Agents LLM Financial Trading Framework}, 
      author={Yijia Xiao and Edward Sun and Di Luo and Wei Wang},
      year={2025},
      eprint={2412.20138},
      archivePrefix={arXiv},
      primaryClass={q-fin.TR},
      url={https://arxiv.org/abs/2412.20138}, 
}
```
