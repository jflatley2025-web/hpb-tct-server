"""CLI interface for Local Chart Overlay."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import click
import yaml

from local_chart_overlay.models.schematic import FrozenSchematic
from local_chart_overlay.models.render_payload import RenderPayload
from local_chart_overlay.storage.sqlite_store import SqliteStore
from local_chart_overlay.ingest.json_adapter import JsonAdapter
from local_chart_overlay.ingest.csv_adapter import CsvAdapter


def _resolve_config(config_path: Optional[str]) -> dict:
    """Load config from yaml file, falling back to defaults."""
    defaults = {
        "db_path": os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "overlay.db"
        ),
        "pine": {
            "version": 6,
            "output_dir": os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "exports"
            ),
            "max_trades_per_file": 50,
        },
    }
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        module_dir = os.path.dirname(os.path.abspath(__file__))
        # Resolve relative paths in user config against module directory
        if "db_path" in user_cfg and not os.path.isabs(user_cfg["db_path"]):
            user_cfg["db_path"] = os.path.join(module_dir, user_cfg["db_path"])
        if "pine" in user_cfg:
            pine = user_cfg["pine"]
            if "output_dir" in pine and not os.path.isabs(pine["output_dir"]):
                pine["output_dir"] = os.path.join(module_dir, pine["output_dir"])
        # Merge, user overrides defaults
        defaults.update(user_cfg)
        if "pine" in user_cfg:
            defaults["pine"] = {**defaults.get("pine", {}), **user_cfg["pine"]}
    return defaults


def _get_store(ctx: click.Context) -> SqliteStore:
    if "store" not in ctx.obj:
        ctx.obj["store"] = SqliteStore(ctx.obj["config"]["db_path"])
    return ctx.obj["store"]


@click.group()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(),
    help="Path to config.yaml",
)
@click.pass_context
def cli(ctx, config_path):
    """Local Chart Overlay — frozen schematic rendering for TradingView."""
    ctx.ensure_object(dict)
    if config_path is None:
        # Default: config.yaml next to this file
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.yaml"
        )
    ctx.obj["config"] = _resolve_config(config_path)


# ── Import Commands ───────────────────────────────────────────────────


@cli.command("import-json")
@click.argument("file_path", type=click.Path(exists=True))
@click.pass_context
def import_json(ctx, file_path):
    """Import trades from a JSON trade log (schematics_5b format)."""
    store = _get_store(ctx)
    adapter = JsonAdapter(file_path)
    pairs = adapter.extract()

    imported = 0
    for trade, schematic in pairs:
        tid = store.upsert_trade(trade)
        if schematic:
            store.upsert_schematic(tid, schematic)
        imported += 1

    click.echo(f"Imported {imported} trades from {file_path}")
    click.echo(f"Total trades in DB: {store.trade_count()}")


@cli.command("import-csv")
@click.argument("file_path", type=click.Path(exists=True))
@click.option(
    "--mapping",
    type=click.Path(exists=True),
    help="YAML file with column mapping overrides",
)
@click.pass_context
def import_csv(ctx, file_path, mapping):
    """Import trades from a CSV trade report."""
    store = _get_store(ctx)

    col_mapping = None
    if mapping:
        with open(mapping) as f:
            col_mapping = yaml.safe_load(f)

    adapter = CsvAdapter(file_path, column_mapping=col_mapping)
    pairs = adapter.extract()

    imported = 0
    for trade, schematic in pairs:
        tid = store.upsert_trade(trade)
        if schematic:
            store.upsert_schematic(tid, schematic)
        imported += 1

    click.echo(f"Imported {imported} trades from {file_path}")
    click.echo(f"Total trades in DB: {store.trade_count()}")


# ── Query Commands ────────────────────────────────────────────────────


@cli.command("list-trades")
@click.option("--symbol", help="Filter by symbol (e.g., BTCUSDT)")
@click.option(
    "--direction",
    type=click.Choice(["bullish", "bearish"]),
    help="Filter by direction",
)
@click.option(
    "--source",
    type=click.Choice(["json", "csv", "postgres"]),
    help="Filter by source type",
)
@click.pass_context
def list_trades(ctx, symbol, direction, source):
    """List all imported trades."""
    store = _get_store(ctx)
    trades = store.list_trades(symbol=symbol, direction=direction, source_type=source)

    if not trades:
        click.echo("No trades found.")
        return

    click.echo(f"{'ID':>4}  {'Symbol':<12} {'Dir':<8} {'TF':<5} {'Model':<12} "
               f"{'Entry':>12} {'PnL%':>8} {'Win':>4} {'Schematic':>9}  Opened")
    click.echo("-" * 110)

    for tid, trade in trades:
        sch = store.get_schematic(tid)
        sch_status = "yes" if sch else "no"
        pnl_str = f"{trade.pnl_pct:+.2f}%" if trade.pnl_pct is not None else "n/a"
        win_str = "W" if trade.is_win else ("L" if trade.is_win is False else "?")
        opened_str = trade.opened_at.strftime("%Y-%m-%d %H:%M") if trade.opened_at else "?"

        click.echo(
            f"{tid:>4}  {trade.symbol:<12} {trade.direction:<8} "
            f"{(trade.timeframe or 'n/a'):<5} {(trade.model or 'n/a'):<12} "
            f"{trade.entry_price:>12.2f} {pnl_str:>8} {win_str:>4} "
            f"{sch_status:>9}  {opened_str}"
        )

    click.echo(f"\nTotal: {len(trades)} trades")


@cli.command("show-trade")
@click.argument("trade_id", type=int)
@click.pass_context
def show_trade(ctx, trade_id):
    """Show full details for a single trade."""
    store = _get_store(ctx)
    trade = store.get_trade(trade_id)
    if not trade:
        click.echo(f"Trade {trade_id} not found.")
        return

    sch = store.get_schematic(trade_id)

    click.echo(f"\n=== Trade #{trade_id} ===")
    click.echo(f"Source:      {trade.source_id} ({trade.source_type})")
    click.echo(f"Symbol:      {trade.symbol}")
    click.echo(f"Direction:   {trade.direction} ({trade.side})")
    click.echo(f"Timeframe:   {trade.timeframe or 'n/a'}")
    click.echo(f"Model:       {trade.model or 'n/a'}")
    click.echo(f"Entry:       {trade.entry_price}")
    click.echo(f"Stop:        {trade.stop_price}")
    click.echo(f"Target:      {trade.target_price}")
    if trade.tp1_price:
        click.echo(f"TP1:         {trade.tp1_price}")
    click.echo(f"R:R:         {trade.rr or 'n/a'}")
    click.echo(f"Score:       {trade.entry_score or 'n/a'}")
    click.echo(f"Opened:      {trade.opened_at}")
    click.echo(f"Closed:      {trade.closed_at or 'still open'}")
    if trade.exit_price:
        click.echo(f"Exit price:  {trade.exit_price}")
    click.echo(f"Exit reason: {trade.exit_reason or 'n/a'}")
    click.echo(f"PnL:         {trade.pnl_pct}% / ${trade.pnl_dollars}")
    click.echo(f"Win:         {trade.is_win}")
    if trade.htf_bias:
        click.echo(f"HTF Bias:    {trade.htf_bias}")

    click.echo(f"\n--- Schematic ---")
    if not sch:
        click.echo("No schematic attached. Use 'attach-schematic' to add one.")
    else:
        click.echo(f"Completeness: {sch.completeness:.0%}")
        click.echo(f"Tap 1:  price={sch.tap1_price}  time={sch.tap1_time}")
        click.echo(f"Tap 2:  price={sch.tap2_price}  time={sch.tap2_time}")
        click.echo(f"Tap 3:  price={sch.tap3_price}  time={sch.tap3_time}")
        click.echo(f"Range:  high={sch.range_high_price}  low={sch.range_low_price}  "
                    f"EQ={sch.range_eq_price}")
        if sch.range_high_time:
            click.echo(f"        high_time={sch.range_high_time}  low_time={sch.range_low_time}")
        click.echo(f"BOS:    price={sch.bos_price}  time={sch.bos_time}")
        click.echo(f"Sweep:  {sch.sweep_type or 'n/a'}")
        click.echo(f"Edited: {sch.manually_edited}  last={sch.last_edited_at}")
        if sch.notes:
            click.echo(f"Notes:  {sch.notes}")

    # Annotations
    ann = store.annotations.get(trade_id)
    click.echo(f"\n--- Annotations ---")
    click.echo(f"Tags:   {ann.tags_csv if ann.has_tags else '(none)'}")
    click.echo(f"Notes:  {ann.notes if ann.has_notes else '(none)'}")


# ── Tagging + Notes ──────────────────────────────────────────────────


@cli.command("tag-trade")
@click.argument("trade_id", type=int)
@click.option("--add", "add_tags", multiple=True, help="Tag(s) to add")
@click.option("--remove", "remove_tags", multiple=True, help="Tag(s) to remove")
@click.option("--list", "list_tags", is_flag=True, help="List current tags")
@click.pass_context
def tag_trade(ctx, trade_id, add_tags, remove_tags, list_tags):
    """Add, remove, or list tags on a trade."""
    store = _get_store(ctx)
    trade = store.get_trade(trade_id)
    if not trade:
        click.echo(f"Trade {trade_id} not found.")
        return

    svc = store.annotations

    if add_tags:
        ann = svc.add_tags(trade_id, list(add_tags))
        click.echo(f"Tags: {ann.tags_csv}")
    if remove_tags:
        ann = svc.remove_tags(trade_id, list(remove_tags))
        click.echo(f"Tags: {ann.tags_csv}")
    if list_tags or (not add_tags and not remove_tags):
        ann = svc.get(trade_id)
        if ann.has_tags:
            click.echo(f"Tags: {ann.tags_csv}")
        else:
            click.echo("No tags.")


@cli.command("note-trade")
@click.argument("trade_id", type=int)
@click.option("--set", "set_note", default=None, help="Set note text")
@click.option("--clear", "clear_note", is_flag=True, help="Clear note")
@click.pass_context
def note_trade(ctx, trade_id, set_note, clear_note):
    """Set or clear a note on a trade."""
    store = _get_store(ctx)
    trade = store.get_trade(trade_id)
    if not trade:
        click.echo(f"Trade {trade_id} not found.")
        return

    svc = store.annotations

    if clear_note:
        svc.clear_note(trade_id)
        click.echo("Note cleared.")
    elif set_note is not None:
        ann = svc.set_note(trade_id, set_note)
        click.echo(f"Note: {ann.notes}")
    else:
        ann = svc.get(trade_id)
        click.echo(f"Note: {ann.notes if ann.has_notes else '(none)'}")


# ── Schematic Attachment ──────────────────────────────────────────────


def _prompt_float(label: str, current: Optional[float] = None) -> Optional[float]:
    """Prompt for a float value, showing current if exists."""
    suffix = f" [{current}]" if current is not None else ""
    val = click.prompt(f"  {label}{suffix}", default="", show_default=False)
    if val == "" and current is not None:
        return current
    if val == "" or val.lower() in ("none", "skip", ""):
        return None
    return float(val)


def _prompt_datetime(label: str, current: Optional[datetime] = None) -> Optional[datetime]:
    """Prompt for a datetime value (ISO 8601)."""
    suffix = f" [{current}]" if current is not None else ""
    val = click.prompt(
        f"  {label}{suffix} (ISO 8601 or 'skip')", default="", show_default=False
    )
    if val == "" and current is not None:
        return current
    if val == "" or val.lower() in ("none", "skip"):
        return None
    dt = datetime.fromisoformat(val)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@cli.command("attach-schematic")
@click.argument("trade_id", type=int)
@click.option("--interactive/--no-interactive", default=True, help="Interactive prompts")
@click.pass_context
def attach_schematic(ctx, trade_id, interactive):
    """Attach or update schematic data for a trade."""
    store = _get_store(ctx)
    trade = store.get_trade(trade_id)
    if not trade:
        click.echo(f"Trade {trade_id} not found.")
        return

    existing = store.get_schematic(trade_id)

    click.echo(f"\nAttaching schematic to Trade #{trade_id}: "
               f"{trade.symbol} {trade.direction} @ {trade.entry_price}")

    if not interactive:
        click.echo("Non-interactive mode: use CLI options (not yet implemented).")
        return

    click.echo("\nEnter values (press Enter to skip, or keep existing):\n")

    tap1_price = _prompt_float("Tap 1 price", existing.tap1_price if existing else None)
    tap1_time = _prompt_datetime("Tap 1 time", existing.tap1_time if existing else None)
    tap2_price = _prompt_float("Tap 2 price", existing.tap2_price if existing else None)
    tap2_time = _prompt_datetime("Tap 2 time", existing.tap2_time if existing else None)
    tap3_price = _prompt_float("Tap 3 price", existing.tap3_price if existing else None)
    tap3_time = _prompt_datetime("Tap 3 time", existing.tap3_time if existing else None)

    range_high = _prompt_float("Range high price", existing.range_high_price if existing else None)
    range_high_time = _prompt_datetime(
        "Range high time", existing.range_high_time if existing else None
    )
    range_low = _prompt_float("Range low price", existing.range_low_price if existing else None)
    range_low_time = _prompt_datetime(
        "Range low time", existing.range_low_time if existing else None
    )

    bos_price = _prompt_float("BOS price", existing.bos_price if existing else None)
    bos_time = _prompt_datetime("BOS time", existing.bos_time if existing else None)

    sweep = click.prompt(
        f"  Sweep type [{existing.sweep_type if existing else ''}]",
        default=existing.sweep_type if existing else "",
        show_default=False,
    )
    notes = click.prompt(
        f"  Notes [{existing.notes if existing else ''}]",
        default=existing.notes if existing else "",
        show_default=False,
    )

    now = datetime.now(timezone.utc)
    sch = FrozenSchematic(
        tap1_price=tap1_price,
        tap1_time=tap1_time,
        tap2_price=tap2_price,
        tap2_time=tap2_time,
        tap3_price=tap3_price,
        tap3_time=tap3_time,
        range_high_price=range_high,
        range_high_time=range_high_time,
        range_low_price=range_low,
        range_low_time=range_low_time,
        bos_price=bos_price,
        bos_time=bos_time,
        sweep_type=sweep or None,
        model_label=trade.model,
        timeframe=trade.timeframe,
        version=existing.version if existing else 1,
        source="manual",
        created_at=existing.created_at if existing else now,
        updated_at=now,
        manually_edited=True,
        last_edited_at=now,
        notes=notes or None,
    )

    store.upsert_schematic(trade_id, sch)
    click.echo(f"\nSchematic saved (completeness: {sch.completeness:.0%})")


# ── Semi-automated Suggestion ─────────────────────────────────────────


@cli.command("suggest-schematic")
@click.argument("trade_id", type=int)
@click.option("--lookback", default=200, help="Bars to look back for range/taps")
@click.option("--accept", is_flag=True, help="Auto-accept top suggestion without prompts")
@click.option(
    "--ohlcv-csv", default=None, type=click.Path(exists=True),
    help="Load OHLCV from local CSV instead of fetching from MEXC",
)
@click.pass_context
def suggest_schematic(ctx, trade_id, lookback, accept, ohlcv_csv):
    """Suggest schematic anchors from OHLCV pivot analysis.

    Fetches candle data around the trade, detects pivots, and proposes
    tap/range/BOS candidates with confidence scores. Requires confirmation.
    """
    from local_chart_overlay.analysis.ohlcv_window import OhlcvWindow
    from local_chart_overlay.analysis.pivot_detector import PivotDetector
    from local_chart_overlay.analysis.range_suggester import RangeSuggester
    from local_chart_overlay.analysis.tap_suggester import TapSuggester

    store = _get_store(ctx)
    trade = store.get_trade(trade_id)
    if not trade:
        click.echo(f"Trade {trade_id} not found.")
        return

    if not trade.timeframe:
        click.echo("Trade has no timeframe — cannot suggest schematic.")
        return

    click.echo(f"\n=== Suggesting schematic for Trade #{trade_id} ===")
    click.echo(f"    {trade.symbol} {trade.direction} {trade.timeframe} "
               f"@ {trade.entry_price}")
    click.echo(f"    Opened: {trade.opened_at}")
    click.echo(f"    Stop: {trade.stop_price}")

    # Step 1: Load OHLCV
    click.echo(f"\n[1/4] Loading OHLCV data ({lookback} bars lookback)...")
    if ohlcv_csv:
        df = OhlcvWindow.load_from_csv(ohlcv_csv)
        ohlcv_source = f"local_csv:{Path(ohlcv_csv).name}"
        click.echo(f"       Loaded {len(df)} candles from {ohlcv_csv}")
    else:
        module_dir = Path(__file__).parent
        cache_dir = module_dir / "data" / "ohlcv_cache"
        loader = OhlcvWindow(cache_dir=cache_dir)
        # Check if cache exists before fetching
        from local_chart_overlay.analysis.ohlcv_window import TF_SECONDS
        tf_sec = TF_SECONDS.get(trade.timeframe, 3600)
        start_t = trade.opened_at - timedelta(seconds=tf_sec * lookback)
        end_t = trade.opened_at + timedelta(seconds=tf_sec * 20)
        cached = loader._load_cache(trade.symbol, trade.timeframe, start_t, end_t)
        df = loader.load(
            symbol=trade.symbol,
            timeframe=trade.timeframe,
            center_time=trade.opened_at,
            lookback_bars=lookback,
            lookahead_bars=20,
        )
        ohlcv_source = "cached_parquet" if cached is not None else "mexc_api"
        click.echo(f"       Fetched {len(df)} candles ({ohlcv_source})")

    if df.empty:
        click.echo("       No candle data available. Cannot suggest.")
        return

    # Step 2: Detect pivots
    click.echo("[2/4] Detecting pivots...")
    detector = PivotDetector(min_strength=2, max_strength=8)
    pivots = detector.detect(df)
    pivot_highs = [p for p in pivots if p.is_high]
    pivot_lows = [p for p in pivots if p.is_low]
    click.echo(f"       Found {len(pivot_highs)} swing highs, {len(pivot_lows)} swing lows")

    # Step 3: Suggest range
    click.echo("[3/4] Searching for consolidation ranges...")
    range_suggester = RangeSuggester()
    range_candidates = range_suggester.suggest(
        pivots, trade.opened_at, trade.entry_price, trade.direction
    )

    best_range = range_candidates[0] if range_candidates else None
    if best_range:
        click.echo(f"       Best range: {best_range.high_price:.2f} – "
                    f"{best_range.low_price:.2f} "
                    f"(EQ {best_range.eq_price:.2f})")
        click.echo(f"       Confidence: {best_range.confidence:.0%}  "
                    f"Tags: {', '.join(best_range.reason_tags)}")
        click.echo(f"       Touches: {best_range.num_touches_high}H / "
                    f"{best_range.num_touches_low}L")
        if len(range_candidates) > 1:
            click.echo(f"       ({len(range_candidates) - 1} more candidates available)")
    else:
        click.echo("       No range candidates found.")

    # Step 4: Suggest taps
    click.echo("[4/4] Proposing tap anchors...")
    tap_suggester = TapSuggester()
    suggestion = tap_suggester.suggest(
        pivots, best_range, trade.opened_at, trade.entry_price,
        trade.stop_price, trade.direction,
    )
    suggestion.trade_id = trade_id

    # Display results
    click.echo(f"\n{'='*60}")
    click.echo(f"SUGGESTION SUMMARY  (overall: {suggestion.overall_confidence:.0%})")
    click.echo(f"{'='*60}")

    _display_tap("Tap 1", suggestion.tap1_candidates)
    _display_tap("Tap 2", suggestion.tap2_candidates)
    _display_tap("Tap 3", suggestion.tap3_candidates)

    if suggestion.bos_price:
        click.echo(f"\nBOS:    price={suggestion.bos_price:.2f}  "
                    f"time={suggestion.bos_time}  "
                    f"conf={suggestion.bos_confidence:.0%}")

    if best_range:
        click.echo(f"\nRange:  high={best_range.high_price:.2f}  "
                    f"low={best_range.low_price:.2f}  "
                    f"EQ={best_range.eq_price:.2f}")

    # Confirm + save
    if not any([suggestion.best_tap1, suggestion.best_tap2, suggestion.best_tap3,
                best_range, suggestion.bos_price]):
        click.echo("\nNo suggestions could be generated. Try adjusting --lookback.")
        return

    if accept:
        do_save = True
    else:
        click.echo("")
        do_save = click.confirm("Accept top suggestions and save schematic?")

    if do_save:
        now = datetime.now(timezone.utc)
        sch = FrozenSchematic(
            tap1_price=suggestion.best_tap1.price if suggestion.best_tap1 else None,
            tap1_time=suggestion.best_tap1.time if suggestion.best_tap1 else None,
            tap2_price=suggestion.best_tap2.price if suggestion.best_tap2 else None,
            tap2_time=suggestion.best_tap2.time if suggestion.best_tap2 else None,
            tap3_price=suggestion.best_tap3.price if suggestion.best_tap3 else None,
            tap3_time=suggestion.best_tap3.time if suggestion.best_tap3 else None,
            range_high_price=best_range.high_price if best_range else None,
            range_high_time=best_range.high_time if best_range else None,
            range_low_price=best_range.low_price if best_range else None,
            range_low_time=best_range.low_time if best_range else None,
            bos_price=suggestion.bos_price,
            bos_time=suggestion.bos_time,
            sweep_type=None,
            model_label=trade.model,
            timeframe=trade.timeframe,
            version=1,
            source="derived",
            created_at=now,
            updated_at=now,
            manually_edited=False,
            confidence=suggestion.overall_confidence,
            data_source=ohlcv_source,
        )
        store.upsert_schematic(trade_id, sch)
        click.echo(f"\nSchematic saved (completeness: {sch.completeness:.0%}, "
                    f"source: derived)")
    else:
        click.echo("\nNot saved. Use 'attach-schematic' to manually refine.")


def _display_tap(label: str, candidates: list):
    """Display tap candidates with score breakdown."""
    if not candidates:
        click.echo(f"\n{label}:   (no candidates)")
        return
    best = candidates[0]
    click.echo(f"\n{label}:   price={best.price:.2f}  time={best.time}  "
               f"conf={best.confidence:.0%}")
    click.echo(f"        tags: {', '.join(best.reason_tags)}  "
               f"pivot_str={best.pivot_strength}")
    if best.score_breakdown:
        parts = []
        for comp, weight in sorted(
            best.score_breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            parts.append(f"{comp}={weight:+.0%}")
        click.echo(f"        breakdown: {', '.join(parts)}")
    if len(candidates) > 1:
        click.echo(f"        +{len(candidates) - 1} alternatives: "
                    f"{', '.join(f'{c.price:.2f}({c.confidence:.0%})' for c in candidates[1:3])}")


# ── Accuracy Evaluation ──────────────────────────────────────────────


@cli.command("score-accuracy")
@click.argument("trade_id", type=int)
@click.option("--lookback", default=200, help="Bars to look back for suggestion")
@click.option(
    "--ohlcv-csv", default=None, type=click.Path(exists=True),
    help="Load OHLCV from local CSV",
)
@click.pass_context
def score_accuracy(ctx, trade_id, lookback, ohlcv_csv):
    """Compare auto-suggestion against a confirmed schematic.

    Requires an already-confirmed schematic for the trade.
    Re-runs the suggestion engine and measures error vs confirmed anchors.
    """
    from local_chart_overlay.analysis.ohlcv_window import OhlcvWindow, TF_SECONDS
    from local_chart_overlay.analysis.pivot_detector import PivotDetector
    from local_chart_overlay.analysis.range_suggester import RangeSuggester
    from local_chart_overlay.analysis.tap_suggester import TapSuggester
    from local_chart_overlay.analysis.accuracy_scorer import AccuracyScorer

    store = _get_store(ctx)
    trade = store.get_trade(trade_id)
    if not trade:
        click.echo(f"Trade {trade_id} not found.")
        return

    confirmed = store.get_schematic(trade_id)
    if not confirmed:
        click.echo(f"No confirmed schematic for trade {trade_id}. "
                    "Use attach-schematic or suggest-schematic first.")
        return

    if not trade.timeframe:
        click.echo("Trade has no timeframe.")
        return

    click.echo(f"\nScoring suggestion accuracy for Trade #{trade_id}...")

    # Load OHLCV
    if ohlcv_csv:
        df = OhlcvWindow.load_from_csv(ohlcv_csv)
    else:
        loader = OhlcvWindow(
            cache_dir=Path(__file__).parent / "data" / "ohlcv_cache"
        )
        df = loader.load(trade.symbol, trade.timeframe, trade.opened_at,
                         lookback_bars=lookback, lookahead_bars=20)

    if df.empty:
        click.echo("No candle data available.")
        return

    # Run suggestion engine
    detector = PivotDetector(min_strength=2, max_strength=8)
    pivots = detector.detect(df)
    range_suggester = RangeSuggester()
    range_candidates = range_suggester.suggest(
        pivots, trade.opened_at, trade.entry_price, trade.direction
    )
    best_range = range_candidates[0] if range_candidates else None

    tap_suggester = TapSuggester()
    suggestion = tap_suggester.suggest(
        pivots, best_range, trade.opened_at, trade.entry_price,
        trade.stop_price, trade.direction,
    )
    suggestion.trade_id = trade_id

    # Score
    scorer = AccuracyScorer()
    report = scorer.score(suggestion, confirmed)
    click.echo(f"\n{report.summary()}")


# ── Pine Export ───────────────────────────────────────────────────────


@cli.command("export-pine")
@click.argument("trade_ids", nargs=-1, type=int)
@click.option("--all", "export_all", is_flag=True, help="Export all trades")
@click.option(
    "--output", "-o", default=None, type=click.Path(), help="Output directory"
)
@click.option("--batch/--individual", default=True, help="Batch or individual files")
@click.option(
    "--group-by",
    type=click.Choice(["none", "symbol", "symbol-tf"]),
    default="symbol-tf",
    help="Auto-group batch files (default: symbol+timeframe)",
)
@click.pass_context
def export_pine(ctx, trade_ids, export_all, output, batch, group_by):
    """Export Pine Script overlay for selected trades."""
    from collections import defaultdict
    from local_chart_overlay.rendering.pine_generator import (
        PineGenerator, validate_timestamp_alignment,
    )

    store = _get_store(ctx)
    config = ctx.obj["config"]

    output_dir = Path(output) if output else Path(config["pine"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if export_all:
        data = store.get_trades_with_schematics()
    elif trade_ids:
        data = store.get_trades_with_schematics(trade_ids=list(trade_ids))
    else:
        click.echo("Specify trade IDs or use --all")
        return

    if not data:
        click.echo("No trades found to export.")
        return

    # Build render payloads
    payloads = []
    for tid, trade, sch in data:
        payloads.append(RenderPayload(trade_id=tid, trade=trade, schematic=sch))

    # Validate timestamp alignment before export
    alignment_warnings = validate_timestamp_alignment(payloads)
    if alignment_warnings:
        click.echo("\n--- Timestamp alignment warnings ---")
        for w in alignment_warnings:
            click.echo(f"  ! {w}")
        click.echo("(Drawings will snap to nearest candle open in TradingView)\n")

    generator = PineGenerator()
    max_per_file = config["pine"].get("max_trades_per_file", 50)

    if not batch:
        for payload in payloads:
            out_path = generator.generate_single(payload, output_dir)
            store.record_export([payload.trade_id], str(out_path))
            click.echo(f"Trade #{payload.trade_id} -> {out_path}")
    elif group_by == "none":
        # Flat chunking (original behavior)
        for i in range(0, len(payloads), max_per_file):
            chunk = payloads[i : i + max_per_file]
            chunk_num = (i // max_per_file) + 1
            out_path = generator.generate_batch(chunk, output_dir, chunk_num)
            ids_in_chunk = [p.trade_id for p in chunk]
            store.record_export(ids_in_chunk, str(out_path))
            click.echo(f"Exported {len(chunk)} trades -> {out_path}")
    else:
        # Group by symbol or symbol+timeframe
        groups: dict[str, list[RenderPayload]] = defaultdict(list)
        for p in payloads:
            if group_by == "symbol":
                key = p.trade.symbol
            else:  # symbol-tf
                key = f"{p.trade.symbol}_{p.trade.timeframe or 'mixed'}"
            groups[key].append(p)

        file_num = 0
        for group_key, group_payloads in sorted(groups.items()):
            # Chunk within each group if needed
            for i in range(0, len(group_payloads), max_per_file):
                chunk = group_payloads[i : i + max_per_file]
                file_num += 1
                out_path = generator.generate_batch(
                    chunk, output_dir, file_num, label=group_key
                )
                ids_in_chunk = [p.trade_id for p in chunk]
                store.record_export(ids_in_chunk, str(out_path))
                click.echo(f"[{group_key}] {len(chunk)} trades -> {out_path}")

    click.echo(f"\nDone. {len(payloads)} trade(s) exported to {output_dir}")


# ── Share Package ─────────────────────────────────────────────────────


@cli.command("export-share-package")
@click.option("--trade", "trade_ids", multiple=True, type=int, help="Trade ID(s)")
@click.option("--all", "export_all", is_flag=True, help="Export all trades")
@click.option(
    "--output-dir", "-o", default=None, type=click.Path(),
    help="Output directory (default: exports/share/)",
)
@click.option(
    "--group-by",
    type=click.Choice(["none", "symbol", "symbol-tf"]),
    default="symbol-tf",
    help="How to group trades into packages",
)
@click.pass_context
def export_share_package(ctx, trade_ids, export_all, output_dir, group_by):
    """Generate a one-click share package (Pine + HTML launcher)."""
    from collections import defaultdict
    from local_chart_overlay.share.package_builder import PackageBuilder

    store = _get_store(ctx)
    config = ctx.obj["config"]

    if export_all:
        data = store.get_trades_with_schematics()
    elif trade_ids:
        data = store.get_trades_with_schematics(trade_ids=list(trade_ids))
    else:
        click.echo("Specify --trade <id> or --all")
        return

    if not data:
        click.echo("No trades found.")
        return

    payloads = [
        RenderPayload(trade_id=tid, trade=trade, schematic=sch)
        for tid, trade, sch in data
    ]

    out_dir = Path(output_dir) if output_dir else (
        Path(config["pine"]["output_dir"]) / "share"
    )
    builder = PackageBuilder(output_dir=out_dir)

    if len(payloads) == 1 or group_by == "none":
        # Single package for all trades
        if len(payloads) == 1:
            pkg = builder.build_single(payloads[0])
        else:
            pkg = builder.build_batch(payloads, label="all_trades")
        click.echo(f"Share package -> {pkg}")
    else:
        groups: dict[str, list[RenderPayload]] = defaultdict(list)
        for p in payloads:
            if group_by == "symbol":
                key = p.trade.symbol
            else:
                key = f"{p.trade.symbol}_{p.trade.timeframe or 'mixed'}"
            groups[key].append(p)

        for group_key, group_payloads in sorted(groups.items()):
            if len(group_payloads) == 1:
                pkg = builder.build_single(group_payloads[0], label=group_key)
            else:
                pkg = builder.build_batch(group_payloads, label=group_key)
            click.echo(f"[{group_key}] {len(group_payloads)} trade(s) -> {pkg}")

    click.echo(f"\nDone. Open the HTML file in any browser to copy the Pine script.")


# ── Trade Replay Inspector ────────────────────────────────────────────


@cli.command("replay-trade")
@click.argument("trade_id", type=int)
@click.option("--output-dir", "-o", default=None, type=click.Path(), help="Output directory")
@click.option("--ohlcv-csv", default=None, type=click.Path(exists=True),
              help="Load OHLCV from local CSV instead of MEXC")
@click.option("--lookback", default=200, type=int, help="Bars to look back")
@click.option("--include-suggestion/--no-suggestion", default=True,
              help="Include auto-suggested schematic comparison")
@click.option("--include-accuracy/--no-accuracy", default=True,
              help="Include accuracy report")
@click.option("--open", "open_browser", is_flag=True, help="Open in browser after generation")
@click.pass_context
def replay_trade(ctx, trade_id, output_dir, ohlcv_csv, lookback,
                 include_suggestion, include_accuracy, open_browser):
    """Generate an interactive HTML replay for a trade."""
    from local_chart_overlay.replay.replay_builder import (
        ReplayBuilder, build_suggested_anchors, build_comparisons_from_schematics,
    )
    from local_chart_overlay.replay.replay_models import AccuracySummary, AnchorPoint
    from local_chart_overlay.replay.replay_template import render_replay_html
    from local_chart_overlay.analysis.ohlcv_window import OhlcvWindow, TF_SECONDS

    store = _get_store(ctx)
    config = ctx.obj["config"]

    trade = store.get_trade(trade_id)
    if not trade:
        click.echo(f"Trade {trade_id} not found.")
        return

    confirmed = store.get_schematic(trade_id)

    click.echo(f"\n=== Replay: Trade #{trade_id} ===")
    click.echo(f"    {trade.symbol} {trade.direction} {trade.timeframe or 'n/a'} "
               f"@ {trade.entry_price}")

    # Load OHLCV
    candles_df = None
    if trade.timeframe:
        click.echo(f"    Loading OHLCV ({lookback} bar lookback)...")
        if ohlcv_csv:
            candles_df = OhlcvWindow.load_from_csv(ohlcv_csv)
        else:
            module_dir = Path(__file__).parent
            loader = OhlcvWindow(cache_dir=module_dir / "data" / "ohlcv_cache")
            candles_df = loader.load(
                trade.symbol, trade.timeframe, trade.opened_at,
                lookback_bars=lookback, lookahead_bars=30,
            )
        click.echo(f"    {len(candles_df)} candles loaded")

    # Build suggested comparison if requested
    suggested_anchors = None
    comparisons = None
    accuracy_summary = None

    if include_suggestion and confirmed and trade.timeframe and candles_df is not None:
        click.echo("    Running suggestion engine for comparison...")
        try:
            from local_chart_overlay.analysis.pivot_detector import PivotDetector
            from local_chart_overlay.analysis.range_suggester import RangeSuggester
            from local_chart_overlay.analysis.tap_suggester import TapSuggester

            detector = PivotDetector(min_strength=2, max_strength=8)
            pivots = detector.detect(candles_df)
            range_sugg = RangeSuggester()
            range_cands = range_sugg.suggest(
                pivots, trade.opened_at, trade.entry_price, trade.direction
            )
            best_range = range_cands[0] if range_cands else None

            tap_sugg = TapSuggester()
            suggestion = tap_sugg.suggest(
                pivots, best_range, trade.opened_at, trade.entry_price,
                trade.stop_price, trade.direction,
            )

            # Build a synthetic FrozenSchematic from suggestion for comparison
            sugg_sch = FrozenSchematic(
                tap1_price=suggestion.best_tap1.price if suggestion.best_tap1 else None,
                tap1_time=suggestion.best_tap1.time if suggestion.best_tap1 else None,
                tap2_price=suggestion.best_tap2.price if suggestion.best_tap2 else None,
                tap2_time=suggestion.best_tap2.time if suggestion.best_tap2 else None,
                tap3_price=suggestion.best_tap3.price if suggestion.best_tap3 else None,
                tap3_time=suggestion.best_tap3.time if suggestion.best_tap3 else None,
                range_high_price=best_range.high_price if best_range else None,
                range_high_time=best_range.high_time if best_range else None,
                range_low_price=best_range.low_price if best_range else None,
                range_low_time=best_range.low_time if best_range else None,
                bos_price=suggestion.bos_price,
                bos_time=suggestion.bos_time,
            )
            suggested_anchors = build_suggested_anchors(sugg_sch)
            comparisons = build_comparisons_from_schematics(confirmed, sugg_sch)

            # Accuracy
            if include_accuracy:
                from local_chart_overlay.analysis.accuracy_scorer import AccuracyScorer
                scorer = AccuracyScorer()
                suggestion.trade_id = trade_id
                report = scorer.score(suggestion, confirmed)
                accuracy_summary = AccuracySummary(
                    hit_rate=report.hit_rate,
                    avg_price_error_pct=report.avg_price_error_pct,
                    avg_time_error_seconds=report.avg_time_error_seconds,
                    anchor_results=[
                        {
                            "name": e.anchor_name,
                            "hit": e.hit,
                            "price_err_pct": e.price_error_pct,
                            "time_err_sec": e.time_error_seconds,
                            "note": e.note,
                        }
                        for e in report.anchor_errors
                        if e.price_error is not None
                    ],
                )
        except Exception as e:
            click.echo(f"    Suggestion engine error: {e}")

    # Build payload (include annotations)
    ann = store.annotations.get(trade_id)
    builder = ReplayBuilder()
    payload = builder.build(
        trade_id=trade_id,
        trade=trade,
        confirmed=confirmed,
        candles_df=candles_df,
        suggested_anchors=suggested_anchors,
        comparisons=comparisons,
        accuracy=accuracy_summary,
        tags=ann.tags,
        notes=ann.notes,
    )

    # Write files
    out_dir = Path(output_dir) if output_dir else (
        Path(config["pine"]["output_dir"]) / "replay"
    )
    trade_dir = out_dir / f"trade_{trade_id}"
    trade_dir.mkdir(parents=True, exist_ok=True)

    # HTML
    title = f"Trade #{trade_id} — {trade.symbol} {trade.timeframe or ''} {trade.direction}"
    html_content = render_replay_html(
        payload.to_json(), title=title, pine_script=payload.pine_script,
    )
    html_path = trade_dir / "replay.html"
    html_path.write_text(html_content, encoding="utf-8")

    # JSON (for debugging)
    json_path = trade_dir / "replay_data.json"
    json_path.write_text(payload.to_json(), encoding="utf-8")

    # README
    readme = _build_replay_readme(trade_id, trade)
    (trade_dir / "README.txt").write_text(readme, encoding="utf-8")

    click.echo(f"\n    Replay generated -> {html_path}")

    if open_browser:
        import webbrowser
        webbrowser.open(str(html_path.resolve()))
        click.echo("    Opened in browser.")


def _build_replay_readme(trade_id: int, trade: TradeRecord) -> str:
    return f"""Trade Replay — #{trade_id}
========================

How to use:
  1. Open replay.html in your browser
  2. Use arrow keys or buttons to step through stages
  3. Toggle confirmed/suggested anchors to compare
  4. Review accuracy in the side panel

Controls:
  Arrow Right / Space  = next stage
  Arrow Left           = previous stage
  Checkboxes           = toggle layers

Trade:
  Symbol:    {trade.symbol}
  Direction: {trade.direction}
  Timeframe: {trade.timeframe or 'n/a'}
  Entry:     {trade.entry_price}
"""


# ── Replay Share Package ──────────────────────────────────────────────


@cli.command("export-replay-package")
@click.option("--trade", "trade_ids", multiple=True, type=int, help="Trade ID(s)")
@click.option("--all", "export_all", is_flag=True, help="Export all trades")
@click.option("--output-dir", "-o", default=None, type=click.Path(), help="Output directory")
@click.option("--ohlcv-csv", default=None, type=click.Path(exists=True),
              help="Load OHLCV from local CSV")
@click.option("--lookback", default=200, type=int, help="Bars to look back")
@click.option("--group-by", type=click.Choice(["none", "symbol", "symbol-tf"]),
              default="symbol-tf", help="How to group trades")
@click.option("--open", "open_browser", is_flag=True, help="Open replay in browser")
@click.pass_context
def export_replay_package(ctx, trade_ids, export_all, output_dir, ohlcv_csv,
                          lookback, group_by, open_browser):
    """Generate a portable replay + Pine share package."""
    from collections import defaultdict
    from local_chart_overlay.replay_share.package_builder import ReplayShareBuilder
    from local_chart_overlay.replay.replay_builder import (
        build_suggested_anchors, build_comparisons_from_schematics,
    )
    from local_chart_overlay.replay.replay_models import AccuracySummary
    from local_chart_overlay.analysis.ohlcv_window import OhlcvWindow

    store = _get_store(ctx)
    config = ctx.obj["config"]

    if export_all:
        data = store.get_trades_with_schematics()
    elif trade_ids:
        data = store.get_trades_with_schematics(trade_ids=list(trade_ids))
    else:
        click.echo("Specify --trade <id> or --all")
        return

    if not data:
        click.echo("No trades found.")
        return

    out_dir = Path(output_dir) if output_dir else (
        Path(config["pine"]["output_dir"]) / "replay_share"
    )
    builder = ReplayShareBuilder(output_dir=out_dir)

    # Group trades
    groups: dict[str, list[tuple[int, TradeRecord, FrozenSchematic | None]]] = defaultdict(list)
    for tid, trade, sch in data:
        if group_by == "none":
            key = f"trade_{tid}"
        elif group_by == "symbol":
            key = trade.symbol
        else:
            key = f"{trade.symbol}_{trade.timeframe or 'mixed'}"
        groups[key].append((tid, trade, sch))

    last_html = None
    for group_key, group_trades in sorted(groups.items()):
        for tid, trade, confirmed in group_trades:
            label = f"{group_key}/trade_{tid}" if len(group_trades) > 1 else group_key

            # Load OHLCV
            candles_df = None
            if trade.timeframe:
                if ohlcv_csv:
                    candles_df = OhlcvWindow.load_from_csv(ohlcv_csv)
                else:
                    module_dir = Path(__file__).parent
                    loader = OhlcvWindow(cache_dir=module_dir / "data" / "ohlcv_cache")
                    try:
                        candles_df = loader.load(
                            trade.symbol, trade.timeframe, trade.opened_at,
                            lookback_bars=lookback, lookahead_bars=30,
                        )
                    except Exception as e:
                        click.echo(f"    OHLCV fetch failed for trade {tid}: {e}")

            # Run suggestion engine if confirmed schematic exists
            suggested_anchors = None
            comparisons = None
            accuracy_summary = None
            if confirmed and trade.timeframe and candles_df is not None:
                try:
                    from local_chart_overlay.analysis.pivot_detector import PivotDetector
                    from local_chart_overlay.analysis.range_suggester import RangeSuggester
                    from local_chart_overlay.analysis.tap_suggester import TapSuggester
                    from local_chart_overlay.analysis.accuracy_scorer import AccuracyScorer

                    detector = PivotDetector(min_strength=2, max_strength=8)
                    pivots = detector.detect(candles_df)
                    range_sugg = RangeSuggester()
                    range_cands = range_sugg.suggest(
                        pivots, trade.opened_at, trade.entry_price, trade.direction
                    )
                    best_range = range_cands[0] if range_cands else None

                    tap_sugg = TapSuggester()
                    suggestion = tap_sugg.suggest(
                        pivots, best_range, trade.opened_at, trade.entry_price,
                        trade.stop_price, trade.direction,
                    )

                    sugg_sch = FrozenSchematic(
                        tap1_price=suggestion.best_tap1.price if suggestion.best_tap1 else None,
                        tap1_time=suggestion.best_tap1.time if suggestion.best_tap1 else None,
                        tap2_price=suggestion.best_tap2.price if suggestion.best_tap2 else None,
                        tap2_time=suggestion.best_tap2.time if suggestion.best_tap2 else None,
                        tap3_price=suggestion.best_tap3.price if suggestion.best_tap3 else None,
                        tap3_time=suggestion.best_tap3.time if suggestion.best_tap3 else None,
                        range_high_price=best_range.high_price if best_range else None,
                        range_high_time=best_range.high_time if best_range else None,
                        range_low_price=best_range.low_price if best_range else None,
                        range_low_time=best_range.low_time if best_range else None,
                        bos_price=suggestion.bos_price,
                        bos_time=suggestion.bos_time,
                    )
                    suggested_anchors = build_suggested_anchors(sugg_sch)
                    comparisons = build_comparisons_from_schematics(confirmed, sugg_sch)

                    scorer = AccuracyScorer()
                    suggestion.trade_id = tid
                    report = scorer.score(suggestion, confirmed)
                    accuracy_summary = AccuracySummary(
                        hit_rate=report.hit_rate,
                        avg_price_error_pct=report.avg_price_error_pct,
                        avg_time_error_seconds=report.avg_time_error_seconds,
                        anchor_results=[
                            {
                                "name": e.anchor_name, "hit": e.hit,
                                "price_err_pct": e.price_error_pct,
                                "time_err_sec": e.time_error_seconds,
                                "note": e.note,
                            }
                            for e in report.anchor_errors if e.price_error is not None
                        ],
                    )
                except Exception as e:
                    click.echo(f"    Suggestion engine error for trade {tid}: {e}")

            ann = store.annotations.get(tid)
            pkg = builder.build(
                trade_id=tid,
                trade=trade,
                confirmed=confirmed,
                candles_df=candles_df,
                suggested_anchors=suggested_anchors,
                comparisons=comparisons,
                accuracy=accuracy_summary,
                label=label,
                tags=ann.tags,
                notes=ann.notes,
            )
            click.echo(f"[{label}] -> {pkg}")
            last_html = pkg / "replay.html"

    click.echo(f"\nDone. {len(data)} trade(s) packaged to {out_dir}")

    if open_browser and last_html and last_html.exists():
        import webbrowser
        webbrowser.open(str(last_html.resolve()))
        click.echo("Opened in browser.")


# ── Replay Index ──────────────────────────────────────────────────────


@cli.command("build-replay-index")
@click.option("--input-dir", "-i", default=None, type=click.Path(),
              help="Directory containing replay share packages")
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Output path for index.html")
@click.option("--recursive", is_flag=True, help="Scan subdirectories recursively")
@click.option("--open", "open_browser", is_flag=True, help="Open in browser after generation")
@click.pass_context
def build_replay_index(ctx, input_dir, output, recursive, open_browser):
    """Generate an index.html for all replay share packages in a folder."""
    from local_chart_overlay.replay_index.index_builder import IndexBuilder

    config = ctx.obj["config"]

    if not input_dir:
        input_dir = Path(config["pine"]["output_dir"]) / "replay_share"
    input_dir = Path(input_dir)

    if not input_dir.exists():
        click.echo(f"Input directory does not exist: {input_dir}")
        return

    builder = IndexBuilder()
    out = builder.build(
        input_dir=input_dir,
        output_path=output,
        recursive=recursive,
    )

    click.echo(f"Index generated -> {out}")

    if open_browser:
        import webbrowser
        webbrowser.open(str(out.resolve()))
        click.echo("Opened in browser.")
