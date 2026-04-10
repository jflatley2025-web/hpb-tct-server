"""Pine Script generator — renders frozen schematic data into .pine files."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from local_chart_overlay.models.render_payload import RenderPayload


TEMPLATE_DIR = Path(__file__).parent / "templates"

# Timeframe durations in seconds for alignment validation
_TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "8h": 28800, "12h": 43200, "1d": 86400, "1w": 604800,
}


def validate_timestamp_alignment(
    payloads: list[RenderPayload],
) -> list[str]:
    """Check if timestamps align with candle boundaries.

    Returns a list of warning strings. Empty list = all clear.
    TradingView uses xloc.bar_time which snaps to candle open times.
    If a timestamp falls mid-candle, the drawing will snap to the
    nearest candle open — which may be visually offset.
    """
    warnings = []
    for p in payloads:
        tf = p.trade.timeframe
        if not tf or tf not in _TF_SECONDS:
            continue
        interval = _TF_SECONDS[tf]

        # Check entry timestamp alignment
        if p.entry_time_ms:
            entry_sec = p.entry_time_ms // 1000
            remainder = entry_sec % interval
            if remainder != 0:
                offset_pct = (remainder / interval) * 100
                if offset_pct > 10:  # only warn if >10% into the candle
                    warnings.append(
                        f"Trade #{p.trade_id}: entry time is {offset_pct:.0f}% "
                        f"into a {tf} candle (may snap to nearest candle open)"
                    )

        # Check schematic tap timestamps
        if p.schematic:
            for label, ms in [
                ("tap1", p.tap1_time_ms), ("tap2", p.tap2_time_ms),
                ("tap3", p.tap3_time_ms),
            ]:
                if ms and ms > 0:
                    remainder = (ms // 1000) % interval
                    if remainder != 0:
                        offset_pct = (remainder / interval) * 100
                        if offset_pct > 25:  # higher threshold for taps
                            warnings.append(
                                f"Trade #{p.trade_id}: {label} time is "
                                f"{offset_pct:.0f}% into a {tf} candle"
                            )
    return warnings


class PineGenerator:
    """Generates Pine Script v6 overlay files from RenderPayload objects.

    All data is baked into the script as static arrays.
    Pine does ZERO calculation — only draws precomputed anchors.
    """

    def __init__(self):
        self._env = Environment(
            loader=FileSystemLoader(str(TEMPLATE_DIR)),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self._template = self._env.get_template("overlay.pine.j2")

    def generate_batch(
        self,
        payloads: list[RenderPayload],
        output_dir: Path,
        chunk_num: int = 1,
        label: str | None = None,
    ) -> Path:
        """Generate a single .pine file containing multiple trades.

        Args:
            label: Optional group label (e.g. "BTCUSDT_1h") used in filename
                   and indicator title for symbol+timeframe grouped exports.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        context = self._build_template_context(payloads, chunk_num, label=label)
        rendered = self._template.render(**context)

        if label:
            safe_label = label.replace("/", "_").replace(" ", "_")
            filename = f"tct_overlay_{safe_label}_{chunk_num}.pine"
        else:
            filename = f"tct_overlay_batch_{chunk_num}.pine"
        out_path = output_dir / filename
        out_path.write_text(rendered, encoding="utf-8")
        return out_path

    def generate_single(self, payload: RenderPayload, output_dir: Path) -> Path:
        """Generate a .pine file for a single trade."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        context = self._build_template_context([payload], 1)
        rendered = self._template.render(**context)

        filename = f"tct_overlay_trade_{payload.trade_id}.pine"
        out_path = output_dir / filename
        out_path.write_text(rendered, encoding="utf-8")
        return out_path

    def _build_template_context(
        self, payloads: list[RenderPayload], chunk_num: int,
        label: str | None = None,
    ) -> dict:
        """Build the full context dict consumed by the Jinja2 template."""
        n = len(payloads)

        # Pre-compute all array data
        symbols = []
        directions = []
        entry_prices = []
        stop_prices = []
        target_prices = []
        tp1_prices = []
        entry_times = []
        exit_times = []
        tap1_prices = []
        tap1_times = []
        tap2_prices = []
        tap2_times = []
        tap3_prices = []
        tap3_times = []
        range_high_prices = []
        range_low_prices = []
        range_high_times = []
        range_low_times = []
        range_eq_prices = []
        bos_prices = []
        bos_times = []
        sweep_types = []
        labels = []
        pnl_pcts = []
        rrs = []
        models = []
        win_flags = []

        for p in payloads:
            t = p.trade
            s = p.schematic

            symbols.append(t.symbol)
            directions.append(p.direction_int)
            entry_prices.append(t.entry_price)
            stop_prices.append(t.stop_price)
            target_prices.append(t.target_price)
            tp1_prices.append(t.tp1_price or 0.0)
            entry_times.append(p.entry_time_ms)
            exit_times.append(p.exit_time_ms)
            labels.append(p.pine_label)
            pnl_pcts.append(t.pnl_pct if t.pnl_pct is not None else 0.0)
            rrs.append(t.rr if t.rr is not None else 0.0)
            models.append(t.model or "n/a")
            win_flags.append(1 if t.is_win else 0)

            if s:
                tap1_prices.append(s.tap1_price or 0.0)
                tap1_times.append(p.tap1_time_ms)
                tap2_prices.append(s.tap2_price or 0.0)
                tap2_times.append(p.tap2_time_ms)
                tap3_prices.append(s.tap3_price or 0.0)
                tap3_times.append(p.tap3_time_ms)
                range_high_prices.append(s.range_high_price or 0.0)
                range_low_prices.append(s.range_low_price or 0.0)
                range_high_times.append(p.range_high_time_ms)
                range_low_times.append(p.range_low_time_ms)
                range_eq_prices.append(s.range_eq_price or 0.0)
                bos_prices.append(s.bos_price or 0.0)
                bos_times.append(p.bos_time_ms)
                sweep_types.append(s.sweep_type or "")
            else:
                tap1_prices.append(0.0)
                tap1_times.append(0)
                tap2_prices.append(0.0)
                tap2_times.append(0)
                tap3_prices.append(0.0)
                tap3_times.append(0)
                range_high_prices.append(0.0)
                range_low_prices.append(0.0)
                range_high_times.append(0)
                range_low_times.append(0)
                range_eq_prices.append(0.0)
                bos_prices.append(0.0)
                bos_times.append(0)
                sweep_types.append("")

        def _fmt_floats(arr):
            return ", ".join(f"{v}" for v in arr)

        def _fmt_ints(arr):
            return ", ".join(str(v) for v in arr)

        def _fmt_strings(arr):
            return ", ".join(f'"{v}"' for v in arr)

        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        batch_name = label or f"Batch {chunk_num}"

        return {
            "trade_count": n,
            "chunk_num": chunk_num,
            "batch_name": batch_name,
            "generated_at": generated_at,
            # Float arrays
            "entry_prices_csv": _fmt_floats(entry_prices),
            "stop_prices_csv": _fmt_floats(stop_prices),
            "target_prices_csv": _fmt_floats(target_prices),
            "tp1_prices_csv": _fmt_floats(tp1_prices),
            "tap1_prices_csv": _fmt_floats(tap1_prices),
            "tap2_prices_csv": _fmt_floats(tap2_prices),
            "tap3_prices_csv": _fmt_floats(tap3_prices),
            "range_high_prices_csv": _fmt_floats(range_high_prices),
            "range_low_prices_csv": _fmt_floats(range_low_prices),
            "range_eq_prices_csv": _fmt_floats(range_eq_prices),
            "bos_prices_csv": _fmt_floats(bos_prices),
            "pnl_pcts_csv": _fmt_floats(pnl_pcts),
            "rrs_csv": _fmt_floats(rrs),
            # Int arrays (timestamps in ms + flags)
            "directions_csv": _fmt_ints(directions),
            "entry_times_csv": _fmt_ints(entry_times),
            "exit_times_csv": _fmt_ints(exit_times),
            "tap1_times_csv": _fmt_ints(tap1_times),
            "tap2_times_csv": _fmt_ints(tap2_times),
            "tap3_times_csv": _fmt_ints(tap3_times),
            "range_high_times_csv": _fmt_ints(range_high_times),
            "range_low_times_csv": _fmt_ints(range_low_times),
            "bos_times_csv": _fmt_ints(bos_times),
            "win_flags_csv": _fmt_ints(win_flags),
            # String arrays
            "symbols_csv": _fmt_strings(symbols),
            "labels_csv": _fmt_strings(labels),
            "models_csv": _fmt_strings(models),
            "sweep_types_csv": _fmt_strings(sweep_types),
        }
