# TradingAgents/graph/trading_graph.py

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yfinance as yf
from langgraph.prebuilt import ToolNode

# Import the abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_crypto_derivatives,
    get_crypto_liquidations,
    get_crypto_intraday_snapshot,
    get_crypto_onchain,
    get_crypto_options,
    get_fundamentals,
    get_global_news,
    get_income_statement,
    get_indicators,
    get_insider_transactions,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
    get_stock_data,
    get_verified_market_snapshot,
    resolve_instrument_identity,
)
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.heatmap_input import (
    HeatmapInputError,
    normalize_heatmap_inputs,
    stage_heatmap_input,
)
from tradingagents.llm_clients import create_llm_client
from tradingagents.llm_clients.agent_overrides import build_agent_llm_overrides
from tradingagents.market_time import analysis_cutoffs, uses_utc_market_day, validate_analysis_date
from tradingagents.portfolio_context import (
    normalize_portfolio_context,
    portfolio_context_fingerprint,
)
from tradingagents.reporting import write_report_tree

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .propagation import Propagator
from .reflection import Reflector
from .setup import GraphSetup
from .signal_processing import SignalProcessor

logger = logging.getLogger(__name__)

VALID_DECISION_HORIZONS = frozenset({"weekly", "monthly", "strategic"})
VALID_CRYPTO_INTELLIGENCE_MODES = frozenset({"disabled", "shadow", "advisory"})


def _outcome_days_from_decision(decision: str) -> int:
    """Map the persisted fixed horizon label to an outcome-review window."""
    lowered = str(decision).lower()
    if "weekly (3-7 calendar days)" in lowered:
        return 5
    if "strategic (1-3 months)" in lowered:
        return 63
    if "monthly (2-4 weeks)" in lowered:
        return 21
    return 5  # legacy entries created before horizons were persisted


def _coerce_max_retries(value):
    """Validate an ``llm_max_retries`` value to a non-negative int.

    Accepts an int or a numeric string (env vars arrive as strings). Rejects
    booleans and negatives loudly so a misconfiguration fails at startup rather
    than silently disabling retries.
    """
    if isinstance(value, bool):
        raise ValueError(f"llm_max_retries must be an integer, not a boolean: {value!r}")
    try:
        n = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"llm_max_retries must be an integer, got {value!r}") from exc
    if n < 0:
        raise ValueError(f"llm_max_retries must be >= 0, got {n}")
    return n


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=("market", "social", "news", "fundamentals"),
        debug=False,
        config: dict[str, Any] = None,
        callbacks: list | None = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()
        self.agent_llms = build_agent_llm_overrides(self.config, llm_kwargs)

        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.conditional_logic,
            agent_llms=self.agent_llms,
        )

        self.propagator = Propagator(
            max_recur_limit=self.config.get("max_recur_limit", 100),
        )
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Graph-shape-affecting run choices, kept for the checkpoint signature.
        self.selected_analysts = tuple(selected_analysts)

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider in {"openai", "openai_compatible"}:
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        # Sampling temperature is cross-provider: forward it whenever set.
        # float() here so a value coming from a TRADINGAGENTS_TEMPERATURE env
        # string ("0.2") works the same as a programmatic float.
        temperature = self.config.get("temperature")
        if temperature is not None and temperature != "":
            kwargs["temperature"] = float(temperature)

        # SDK retry budget is cross-provider. Forward it only when explicitly set
        # so each provider keeps its own default (usually 2) otherwise (#1091).
        max_retries = self.config.get("llm_max_retries")
        if max_retries is not None and max_retries != "":
            kwargs["max_retries"] = _coerce_max_retries(max_retries)

        return kwargs

    def _create_tool_nodes(self) -> dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                    # Deterministic verification snapshot (bound to the analyst
                    # LLM and required by its prompt; must be executable here or
                    # the call fails and the model reports it "unavailable").
                    get_verified_market_snapshot,
                    get_crypto_intraday_snapshot,
                ]
            ),
            "crypto_intelligence": ToolNode(
                [
                    get_crypto_derivatives,
                    get_crypto_options,
                    get_crypto_onchain,
                    get_crypto_liquidations,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                    get_macro_indicators,
                    get_prediction_markets,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def _resolve_benchmark(self, ticker: str) -> str:
        """Pick the benchmark ticker for alpha calculation against ``ticker``.

        ``config["benchmark_ticker"]`` overrides everything when set; otherwise
        the suffix map matches the ticker's exchange suffix (e.g. ``.T`` for
        Tokyo). US-listed tickers without a dotted suffix fall through to the
        empty-suffix entry (SPY by default). Unrecognised suffixes (including
        US tickers with dots like ``BRK.B``) also fall back to the empty-suffix
        entry, which is the right default because the alpha calculation works
        in USD.
        """
        explicit = self.config.get("benchmark_ticker")
        if explicit:
            return explicit
        benchmark_map = self.config.get("benchmark_map", {})
        ticker_upper = ticker.upper()
        for suffix, benchmark in benchmark_map.items():
            if suffix and ticker_upper.endswith(suffix.upper()):
                return benchmark
        return benchmark_map.get("", "SPY")

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5,
        benchmark: str = "SPY",
    ) -> tuple[float | None, float | None, int | None]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        ``benchmark`` is the index used as the alpha baseline (resolved by the
        caller via ``_resolve_benchmark``). Returns ``(raw_return, alpha_return,
        actual_holding_days)`` or ``(None, None, None)`` if price data is
        unavailable (too recent, delisted, or network error).
        """
        from tradingagents.dataflows.symbol_utils import normalize_symbol

        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            # ``holding_days`` is measured in observed market rows. Allow enough
            # calendar time for weekends and holidays at monthly/strategic horizons.
            end = start + timedelta(days=int(holding_days * 1.6) + 7)
            end_str = end.strftime("%Y-%m-%d")

            # Normalize so the realized-return lookup hits the same instrument
            # the analysis priced (e.g. XAUUSD -> GC=F) (#984). The benchmark is
            # already a canonical Yahoo symbol from ``_resolve_benchmark``.
            stock = yf.Ticker(normalize_symbol(ticker)).history(start=trade_date, end=end_str)
            bench = yf.Ticker(benchmark).history(start=trade_date, end=end_str)

            if len(stock) <= holding_days or len(bench) <= holding_days:
                return None, None, None

            actual_days = holding_days
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            bench_ret = float(
                (bench["Close"].iloc[actual_days] - bench["Close"].iloc[0])
                / bench["Close"].iloc[0]
            )
            alpha = raw - bench_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s vs %s (will retry next run): %s",
                ticker, trade_date, benchmark, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        benchmark = self._resolve_benchmark(ticker)
        updates = []
        for entry in pending:
            holding_days = _outcome_days_from_decision(entry.get("decision", ""))
            raw, alpha, days = self._fetch_returns(
                ticker,
                entry["date"],
                holding_days=holding_days,
                benchmark=benchmark,
            )
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
                benchmark_name=benchmark,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def resolve_instrument_context(self, ticker: str, asset_type: str = "stock") -> str:
        """Resolve ticker identity once and return the full instrument context.

        Deterministic yfinance lookup (cached, fail-open) injected into a
        context string so every agent anchors to the real company instead of
        hallucinating one from the price chart (#814). Both the propagate()
        path and the CLI call this so the resolved identity reaches the whole
        graph regardless of entry point.
        """
        identity = resolve_instrument_identity(ticker)
        return build_instrument_context(ticker, asset_type, identity)

    def _run_signature(
        self,
        asset_type: str,
        decision_horizon: str | None = None,
        crypto_intelligence_mode: str | None = None,
        heatmap_input: str = "",
        portfolio_context: dict[str, Any] | None = None,
        heatmap_inputs: dict[str, str] | None = None,
    ) -> str:
        """Run-identity inputs that must invalidate a checkpoint if changed.

        Keyed into the checkpoint thread ID so a resume under a different analyst
        selection, debate/risk depth, asset mode, data input, horizon, or portfolio
        context starts fresh instead of silently continuing incompatible state.
        """
        normalized_heatmaps = normalize_heatmap_inputs(heatmap_input, heatmap_inputs)
        serialized_heatmaps = json.dumps(normalized_heatmaps, sort_keys=True, separators=(",", ":"))
        heatmap_signature = (
            hashlib.sha256(serialized_heatmaps.encode("utf-8")).hexdigest()[:12]
            if normalized_heatmaps
            else "none"
        )
        return "|".join([
            "analysts=" + ",".join(self.selected_analysts),
            f"debate={self.config['max_debate_rounds']}",
            f"risk={self.config['max_risk_discuss_rounds']}",
            f"asset={asset_type}",
            f"horizon={decision_horizon or self.config.get('decision_horizon', 'monthly')}",
            f"crypto_mode={crypto_intelligence_mode or self.config.get('crypto_intelligence_mode', 'disabled')}",
            f"heatmap={heatmap_signature}",
            f"portfolio={portfolio_context_fingerprint(portfolio_context)}",
        ])

    def propagate(
        self,
        company_name,
        trade_date,
        asset_type: str = "stock",
        *,
        decision_horizon: str | None = None,
        crypto_intelligence_mode: str | None = None,
        heatmap_input: str = "",
        heatmap_inputs: dict[str, str] | None = None,
        portfolio_context: dict[str, Any] | None = None,
    ):
        """Run the trading agents graph for a company on a specific date.

        ``asset_type`` selects between the stock pipeline (default) and the
        crypto pipeline (``"crypto"``) shipped in #567 — the CLI auto-detects
        from the ticker; programmatic callers pass it explicitly. When
        ``checkpoint_enabled`` is set in config, the graph is recompiled with
        a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        # A host-local "today" can already be tomorrow in UTC (e.g. before
        # 08:00 in Beijing). Refuse that future candle on programmatic crypto
        # calls just as the CLI does; historical dates are unchanged.
        if uses_utc_market_day(asset_type):
            validate_analysis_date(trade_date, asset_type)

        decision_horizon = decision_horizon or self.config.get("decision_horizon", "monthly")
        crypto_intelligence_mode = crypto_intelligence_mode or self.config.get(
            "crypto_intelligence_mode", "disabled"
        )
        if decision_horizon not in VALID_DECISION_HORIZONS:
            raise ValueError(f"decision_horizon must be one of {sorted(VALID_DECISION_HORIZONS)}")
        if crypto_intelligence_mode not in VALID_CRYPTO_INTELLIGENCE_MODES:
            raise ValueError(
                "crypto_intelligence_mode must be disabled, shadow, or advisory"
            )
        if asset_type != "crypto":
            crypto_intelligence_mode = "disabled"
        portfolio_context = normalize_portfolio_context(portfolio_context)

        self.ticker = company_name

        # Resolve any pending memory-log entries for this ticker before the pipeline runs.
        self._resolve_pending_entries(company_name)

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date),
                self._run_signature(
                    asset_type,
                    decision_horizon,
                    crypto_intelligence_mode,
                    heatmap_input,
                    portfolio_context,
                    heatmap_inputs,
                ),
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(
                company_name,
                trade_date,
                asset_type=asset_type,
                decision_horizon=decision_horizon,
                crypto_intelligence_mode=crypto_intelligence_mode,
                heatmap_input=heatmap_input,
                heatmap_inputs=heatmap_inputs,
                portfolio_context=portfolio_context,
            )
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def save_reports(self, final_state, ticker, save_path=None) -> Path:
        """Write the markdown report tree for a completed run, like the CLI does.

        Programmatic callers get the same on-disk reports the CLI produces. Pass
        an explicit ``save_path`` or let it default under ``results_dir``.
        """
        if save_path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = (
                Path(self.config["results_dir"])
                / "reports"
                / f"{safe_ticker_component(ticker)}_{stamp}"
            )
        return write_report_tree(final_state, ticker, save_path)

    def create_initial_state(
        self,
        company_name: str,
        trade_date: str,
        *,
        asset_type: str = "stock",
        decision_horizon: str = "monthly",
        crypto_intelligence_mode: str = "disabled",
        heatmap_input: str = "",
        heatmap_inputs: dict[str, str] | None = None,
        portfolio_context: dict[str, Any] | None = None,
        past_context: str = "",
    ) -> dict[str, Any]:
        """Build a task state with deterministic time cutoffs and image staging."""
        if decision_horizon not in VALID_DECISION_HORIZONS:
            raise ValueError(f"decision_horizon must be one of {sorted(VALID_DECISION_HORIZONS)}")
        if crypto_intelligence_mode not in VALID_CRYPTO_INTELLIGENCE_MODES:
            raise ValueError(
                "crypto_intelligence_mode must be disabled, shadow, or advisory"
            )
        if asset_type != "crypto":
            crypto_intelligence_mode = "disabled"
        portfolio_context = normalize_portfolio_context(portfolio_context)
        cutoffs = analysis_cutoffs(trade_date, asset_type)
        normalized_heatmaps = normalize_heatmap_inputs(heatmap_input, heatmap_inputs)
        heatmap_artifacts = {}
        if asset_type == "crypto" and crypto_intelligence_mode != "disabled":
            for role, value in normalized_heatmaps.items():
                try:
                    artifact = stage_heatmap_input(
                        value,
                        self.config["data_cache_dir"],
                        str(trade_date),
                    )
                    artifact["view"] = role
                    heatmap_artifacts[role] = artifact
                except HeatmapInputError as exc:
                    logger.warning("Optional %s liquidation heatmap skipped: %s", role, exc)
                    heatmap_artifacts[role] = {
                        "error": str(exc),
                        "original_input": value,
                        "view": role,
                    }
        heatmap_artifact = heatmap_artifacts.get("overview", {})
        instrument_context = self.resolve_instrument_context(company_name, asset_type)
        return self.propagator.create_initial_state(
            company_name,
            trade_date,
            asset_type=asset_type,
            past_context=past_context,
            instrument_context=instrument_context,
            analysis_as_of=cutoffs.analysis_as_of,
            completed_daily_candle_date=cutoffs.completed_daily_candle_date,
            completed_4h_candle_end=cutoffs.completed_4h_candle_end,
            completed_1h_candle_end=cutoffs.completed_1h_candle_end,
            decision_horizon=decision_horizon,
            crypto_intelligence_mode=crypto_intelligence_mode,
            heatmap_input=heatmap_input,
            heatmap_artifact=heatmap_artifact,
            # Keep the legacy singular representation singular so old callers
            # do not present the same overview input through both state fields.
            heatmap_inputs={} if heatmap_input else normalized_heatmaps,
            heatmap_artifacts=heatmap_artifacts,
            portfolio_context=portfolio_context,
        )

    def _run_graph(
        self,
        company_name,
        trade_date,
        asset_type: str = "stock",
        decision_horizon: str = "monthly",
        crypto_intelligence_mode: str = "disabled",
        heatmap_input: str = "",
        heatmap_inputs: dict[str, str] | None = None,
        portfolio_context: dict[str, Any] | None = None,
    ):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Initialize state — inject memory log context for PM and the
        # deterministically resolved instrument identity for all agents.
        past_context = self.memory_log.get_past_context(company_name)
        init_agent_state = self.create_initial_state(
            company_name,
            trade_date,
            asset_type=asset_type,
            past_context=past_context,
            decision_horizon=decision_horizon,
            crypto_intelligence_mode=crypto_intelligence_mode,
            heatmap_input=heatmap_input,
            heatmap_inputs=heatmap_inputs,
            portfolio_context=portfolio_context,
        )
        args = self.propagator.get_graph_args()

        # Inject thread_id so same ticker+date+graph-shape resumes; a different
        # date or graph shape starts fresh (#1089).
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(
                company_name,
                str(trade_date),
                self._run_signature(
                    asset_type,
                    decision_horizon,
                    crypto_intelligence_mode,
                    heatmap_input,
                    portfolio_context,
                    heatmap_inputs,
                ),
            )
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            trace = []
            last_printed = None
            for chunk in self.graph.stream(init_agent_state, **args):
                if chunk["messages"]:
                    msg = chunk["messages"][-1]
                    # Nodes after the trader don't append to messages, so the
                    # same trailing message repeats across chunks. Print it only
                    # when it changes (#1027); the trace/state merge is unchanged.
                    signature = (type(msg).__name__, getattr(msg, "content", None))
                    if signature != last_printed:
                        msg.pretty_print()
                        last_printed = signature
                    trace.append(chunk)
            # Streamed chunks are per-node deltas. Merge them so the returned
            # state matches what graph.invoke() yields in the non-debug path.
            final_state = dict(init_agent_state)
            for chunk in trace:
                final_state.update(chunk)
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection.
        self.curr_state = final_state

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run.
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date),
                self._run_signature(
                    asset_type,
                    decision_horizon,
                    crypto_intelligence_mode,
                    heatmap_input,
                    portfolio_context,
                    heatmap_inputs,
                ),
            )

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def stream_with_checkpoint(self, init_agent_state: dict[str, Any], callbacks=None):
        """Stream a prebuilt state while honoring CLI checkpoint configuration.

        The interactive CLI builds state itself so it can display live cutoffs and
        per-agent progress. This wrapper gives that path the same checkpoint
        isolation and cleanup as :meth:`propagate`, including the portfolio
        fingerprint in the run signature.
        """
        args = self.propagator.get_graph_args(callbacks=callbacks)
        if not self.config.get("checkpoint_enabled"):
            yield from self.graph.stream(init_agent_state, **args)
            return

        ticker = str(init_agent_state["company_of_interest"])
        trade_date = str(init_agent_state["trade_date"])
        signature = self._run_signature(
            str(init_agent_state.get("asset_type", "stock")),
            str(init_agent_state.get("decision_horizon", "monthly")),
            str(init_agent_state.get("crypto_intelligence_mode", "disabled")),
            str(init_agent_state.get("heatmap_input", "")),
            init_agent_state.get("portfolio_context"),
            init_agent_state.get("heatmap_inputs"),
        )
        completed = False
        with get_checkpointer(self.config["data_cache_dir"], ticker) as saver:
            checkpointed_graph = self.workflow.compile(checkpointer=saver)
            args.setdefault("config", {}).setdefault("configurable", {})[
                "thread_id"
            ] = thread_id(ticker, trade_date, signature)
            step = checkpoint_step(
                self.config["data_cache_dir"], ticker, trade_date, signature
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, ticker, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", ticker, trade_date)
            try:
                yield from checkpointed_graph.stream(init_agent_state, **args)
                completed = True
            finally:
                if completed:
                    clear_checkpoint(
                        self.config["data_cache_dir"], ticker, trade_date, signature
                    )

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "analysis_as_of": final_state.get("analysis_as_of"),
            "completed_daily_candle_date": final_state.get("completed_daily_candle_date"),
            "completed_4h_candle_end": final_state.get("completed_4h_candle_end"),
            "completed_1h_candle_end": final_state.get("completed_1h_candle_end"),
            "decision_horizon": final_state.get("decision_horizon"),
            "crypto_intelligence_mode": final_state.get("crypto_intelligence_mode"),
            "portfolio_context": final_state.get("portfolio_context", {"status": "unknown"}),
            "heatmap_artifact": final_state.get("heatmap_artifact", {}),
            "heatmap_inputs": final_state.get("heatmap_inputs", {}),
            "heatmap_artifacts": final_state.get("heatmap_artifacts", {}),
            "heatmap_visual_report": final_state.get("heatmap_visual_report", ""),
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "crypto_intelligence_report": final_state.get("crypto_intelligence_report", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
