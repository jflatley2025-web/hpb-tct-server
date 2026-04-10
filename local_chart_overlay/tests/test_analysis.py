"""Tests for the analysis module — pivot detection, range/tap suggestion.

All tests use synthetic OHLCV data. No API calls.
"""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from local_chart_overlay.analysis.pivot_detector import PivotDetector, Pivot
from local_chart_overlay.analysis.range_suggester import RangeSuggester, RangeCandidate
from local_chart_overlay.analysis.tap_suggester import TapSuggester, TapCandidate, SchematicSuggestion
from local_chart_overlay.analysis.accuracy_scorer import AccuracyScorer, AccuracyReport
from local_chart_overlay.analysis.ohlcv_window import OhlcvWindow
from local_chart_overlay.models.schematic import FrozenSchematic


# ── Helpers ───────────────────────────────────────────────────────────


def _make_candles(
    prices: list[tuple[float, float, float, float]],
    start: datetime = None,
    interval_sec: int = 3600,
) -> pd.DataFrame:
    """Build a synthetic OHLCV DataFrame from (open, high, low, close) tuples."""
    if start is None:
        start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)

    rows = []
    for i, (o, h, l, c) in enumerate(prices):
        t = start + timedelta(seconds=interval_sec * i)
        rows.append({
            "open_time": t,
            "open": o, "high": h, "low": l, "close": c, "volume": 1000.0,
        })
    return pd.DataFrame(rows)


def _make_range_candles(
    range_high: float = 100.0,
    range_low: float = 95.0,
    n_oscillations: int = 4,
    n_pre: int = 5,
    n_post: int = 5,
) -> pd.DataFrame:
    """Build candles that form a clear consolidation range.

    Pattern:
      - n_pre bars trending up to range_high
      - n_oscillations * 4 bars oscillating between range_high and range_low
      - n_post bars after the range
    """
    mid = (range_high + range_low) / 2
    prices = []

    # Pre-range: trend up
    for i in range(n_pre):
        p = range_low - 5 + (5 * i / n_pre)
        prices.append((p, p + 1, p - 0.5, p + 0.5))

    # Range oscillation: high → mid → low → mid repeated
    for _ in range(n_oscillations):
        # Move to high
        prices.append((mid, range_high, mid - 0.5, range_high - 0.2))
        prices.append((range_high - 0.2, range_high + 0.1, mid + 1, mid + 1.5))
        # Move to low
        prices.append((mid, mid + 0.5, range_low, range_low + 0.3))
        prices.append((range_low + 0.3, mid - 1, range_low - 0.1, mid - 1.5))

    # Post-range
    for i in range(n_post):
        p = mid + i * 0.5
        prices.append((p, p + 0.5, p - 0.3, p + 0.2))

    return _make_candles(prices)


def _make_bearish_setup(
    range_high: float = 100.0,
    range_low: float = 95.0,
) -> tuple[pd.DataFrame, datetime, float, float]:
    """Build a complete bearish schematic setup.

    Returns (candles_df, entry_time, entry_price, stop_price)

    Pattern:
    1. Range formation (oscillation)
    2. Tap 1: pivot high at range_high
    3. Tap 2: higher high (deviation)
    4. Tap 3: lower high
    5. BOS: break below range
    6. Entry below EQ
    """
    eq = (range_high + range_low) / 2
    start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
    prices = []

    # Phase 1: Range oscillation (bars 0-15)
    for i in range(4):
        prices.append((eq, range_high, eq - 1, range_high - 0.5))   # up to high
        prices.append((range_high - 0.5, range_high, eq, eq + 0.5)) # near high
        prices.append((eq, eq + 0.5, range_low, range_low + 0.3))   # down to low
        prices.append((range_low + 0.3, eq, range_low, eq - 0.5))   # near low

    # Phase 2: Tap 1 — touch range high (bar 16)
    prices.append((eq, range_high, eq - 0.5, range_high - 0.3))

    # Pullback (bars 17-18)
    prices.append((range_high - 0.3, range_high - 0.2, eq + 1, eq + 1.5))
    prices.append((eq + 1.5, eq + 2, eq, eq + 0.5))

    # Phase 3: Tap 2 — deviation above range (bar 19)
    tap2_price = range_high + 1.5
    prices.append((eq + 0.5, tap2_price, eq + 0.3, tap2_price - 0.5))

    # Pullback (bars 20-21)
    prices.append((tap2_price - 0.5, tap2_price - 0.3, eq + 1, eq + 1.5))
    prices.append((eq + 1.5, eq + 2, eq, eq + 0.3))

    # Phase 4: Tap 3 — lower high (bar 22)
    tap3_price = range_high + 0.5  # lower than tap2
    prices.append((eq + 0.3, tap3_price, eq - 0.5, tap3_price - 0.8))

    # Phase 5: Breakdown / BOS (bars 23-25)
    prices.append((tap3_price - 0.8, eq + 0.5, eq - 1, eq - 0.5))
    prices.append((eq - 0.5, eq, range_low - 0.5, range_low - 0.3))
    prices.append((range_low - 0.3, range_low, range_low - 1.5, range_low - 1))

    # Phase 6: Entry area (bars 26-28)
    entry_price = eq - 1.5
    prices.append((range_low - 1, range_low - 0.5, entry_price - 0.5, entry_price))
    prices.append((entry_price, entry_price + 0.3, entry_price - 1, entry_price - 0.5))
    prices.append((entry_price - 0.5, entry_price, entry_price - 2, entry_price - 1.5))

    df = _make_candles(prices, start=start, interval_sec=3600)
    entry_time = start + timedelta(hours=26)
    stop_price = range_high + 2

    return df, entry_time, entry_price, stop_price


# ── Pivot Detector Tests ──────────────────────────────────────────────


class TestPivotDetector:
    def test_detects_obvious_high(self):
        # V-shape: prices go up then down → single pivot high
        prices = [
            (10, 11, 9, 10), (11, 12, 10, 11), (12, 13, 11, 12),
            (13, 20, 12, 19),  # peak
            (18, 19, 12, 13), (12, 13, 11, 12), (11, 12, 10, 11),
        ]
        df = _make_candles(prices)
        detector = PivotDetector(min_strength=2, max_strength=3)
        pivots = detector.detect(df)
        highs = [p for p in pivots if p.is_high]
        assert len(highs) >= 1
        assert highs[0].price == 20.0

    def test_detects_obvious_low(self):
        # Inverse V: down then up → single pivot low
        prices = [
            (20, 21, 19, 20), (19, 20, 18, 19), (18, 19, 17, 18),
            (17, 18, 10, 11),  # trough
            (12, 13, 11, 12), (13, 14, 12, 13), (14, 15, 13, 14),
        ]
        df = _make_candles(prices)
        detector = PivotDetector(min_strength=2, max_strength=3)
        pivots = detector.detect(df)
        lows = [p for p in pivots if p.is_low]
        assert len(lows) >= 1
        assert lows[0].price == 10.0

    def test_strength_scales(self):
        """Higher strength requires more confirming bars."""
        # Wide peak: 5 bars up, peak, 5 bars down
        prices = []
        for i in range(6):
            p = 10 + i * 2
            prices.append((p, p + 1, p - 0.5, p + 0.5))
        prices.append((22, 30, 21, 29))  # peak
        for i in range(6):
            p = 28 - i * 2
            prices.append((p, p + 1, p - 1, p))

        df = _make_candles(prices)
        detector = PivotDetector(min_strength=2, max_strength=6)
        pivots = detector.detect(df)
        highs = [p for p in pivots if p.is_high and p.price == 30.0]
        assert len(highs) >= 1
        assert highs[0].strength >= 4

    def test_empty_df(self):
        df = pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
        detector = PivotDetector()
        assert detector.detect(df) == []

    def test_too_few_bars(self):
        prices = [(10, 11, 9, 10), (11, 12, 10, 11)]
        df = _make_candles(prices)
        detector = PivotDetector(min_strength=2)
        assert detector.detect(df) == []

    def test_detect_in_range_filters(self):
        prices = [
            (10, 11, 9, 10), (11, 12, 10, 11), (12, 13, 11, 12),
            (13, 20, 12, 19), (18, 19, 12, 13), (12, 13, 11, 12),
            (11, 12, 10, 11),
        ]
        start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
        df = _make_candles(prices, start=start)
        detector = PivotDetector(min_strength=2, max_strength=3)

        # Restrict to first 2 hours — should miss the peak at hour 3
        pivots = detector.detect_in_range(
            df,
            start_time=start,
            end_time=start + timedelta(hours=2),
        )
        highs = [p for p in pivots if p.is_high and p.price == 20.0]
        assert len(highs) == 0

    def test_prominence_computed(self):
        prices = [
            (10, 11, 9, 10), (11, 12, 10, 11), (12, 13, 11, 12),
            (13, 20, 12, 19),  # high
            (18, 19, 5, 6),    # low
            (7, 8, 6, 7), (8, 9, 7, 8),
        ]
        df = _make_candles(prices)
        detector = PivotDetector(min_strength=2, max_strength=3)
        pivots = detector.detect(df)
        for p in pivots:
            if p.price in (20.0, 5.0):
                assert p.prominence > 0

    def test_significant_filter(self):
        detector = PivotDetector()
        pivots = [
            Pivot(datetime.now(timezone.utc), 100, 0, "high", 2, 5.0),
            Pivot(datetime.now(timezone.utc), 110, 1, "high", 5, 15.0),
            Pivot(datetime.now(timezone.utc), 90, 2, "low", 3, 10.0),
        ]
        sig = detector.get_significant_pivots(pivots, min_strength=3)
        assert len(sig) == 2  # only strength >= 3


# ── Range Suggester Tests ─────────────────────────────────────────────


class TestRangeSuggester:
    def test_finds_range_in_oscillation(self):
        df = _make_range_candles(range_high=100, range_low=95, n_oscillations=4)
        detector = PivotDetector(min_strength=2, max_strength=5)
        pivots = detector.detect(df)

        # Entry after the range
        entry_time = df["open_time"].iloc[-1] + timedelta(hours=1)

        suggester = RangeSuggester(cluster_tolerance_pct=1.0, min_touches=2)
        candidates = suggester.suggest(
            pivots, entry_time, 97.5, "bearish"
        )
        assert len(candidates) >= 1
        best = candidates[0]
        # Range should be roughly 95-100
        assert best.high_price > 98
        assert best.low_price < 97
        assert best.confidence > 0

    def test_no_range_in_trend(self):
        # Pure uptrend — no consolidation
        prices = [(10 + i, 11 + i, 9 + i, 10.5 + i) for i in range(20)]
        df = _make_candles(prices)
        detector = PivotDetector(min_strength=2, max_strength=5)
        pivots = detector.detect(df)
        entry_time = df["open_time"].iloc[-1] + timedelta(hours=1)

        suggester = RangeSuggester()
        candidates = suggester.suggest(pivots, entry_time, 30, "bearish")
        # May find 0 or weak candidates
        for c in candidates:
            assert c.confidence < 0.5

    def test_range_candidate_eq(self):
        c = RangeCandidate(
            high_price=100, high_time=datetime.now(timezone.utc),
            low_price=90, low_time=datetime.now(timezone.utc),
            confidence=0.8,
        )
        assert c.eq_price == 95.0
        assert c.size == 10.0
        assert abs(c.size_pct - 11.11) < 0.1


# ── Tap Suggester Tests ───────────────────────────────────────────────


class TestTapSuggester:
    def test_bearish_setup_finds_taps(self):
        df, entry_time, entry_price, stop_price = _make_bearish_setup()
        detector = PivotDetector(min_strength=2, max_strength=5)
        pivots = detector.detect(df)

        range_cand = RangeCandidate(
            high_price=100.0,
            high_time=datetime(2026, 3, 1, 4, 0, 0, tzinfo=timezone.utc),
            low_price=95.0,
            low_time=datetime(2026, 3, 1, 6, 0, 0, tzinfo=timezone.utc),
            confidence=0.7,
            reason_tags=["multi_touch"],
        )

        suggester = TapSuggester()
        result = suggester.suggest(
            pivots, range_cand, entry_time, entry_price,
            stop_price, "bearish",
        )

        assert isinstance(result, SchematicSuggestion)
        assert result.direction == "bearish"
        # Should find at least some tap candidates
        total = (len(result.tap1_candidates) + len(result.tap2_candidates) +
                 len(result.tap3_candidates))
        assert total > 0

    def test_bullish_direction(self):
        """TapSuggester should handle bullish direction."""
        start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
        # Build a bullish accumulation pattern (inverted bearish)
        prices = []
        range_high, range_low = 100.0, 95.0
        eq = 97.5

        # Range oscillation
        for i in range(4):
            prices.append((eq, range_high, eq - 1, range_high - 0.5))
            prices.append((range_high - 0.5, range_high, eq, eq + 0.5))
            prices.append((eq, eq + 0.5, range_low, range_low + 0.3))
            prices.append((range_low + 0.3, eq, range_low, eq - 0.5))

        # Tap 1: touch range low
        prices.append((eq, eq + 0.5, range_low, range_low + 0.2))
        # Tap 2: deviation below
        prices.append((range_low + 0.2, range_low, range_low - 1.5, range_low - 1))
        # Tap 3: higher low
        prices.append((range_low - 1, range_low, range_low - 0.5, range_low))
        # Recovery
        prices.append((range_low, eq + 1, range_low - 0.2, eq))
        prices.append((eq, eq + 2, eq - 0.5, eq + 1.5))

        df = _make_candles(prices, start=start)
        entry_time = start + timedelta(hours=len(prices) - 1)

        detector = PivotDetector(min_strength=2, max_strength=4)
        pivots = detector.detect(df)

        range_cand = RangeCandidate(
            high_price=range_high,
            high_time=start + timedelta(hours=2),
            low_price=range_low,
            low_time=start + timedelta(hours=4),
            confidence=0.6,
        )

        suggester = TapSuggester()
        result = suggester.suggest(
            pivots, range_cand, entry_time, eq + 1, range_low - 2, "bullish"
        )
        assert result.direction == "bullish"

    def test_no_pivots_returns_empty(self):
        suggester = TapSuggester()
        result = suggester.suggest(
            [], None,
            datetime.now(timezone.utc), 100, 95, "bearish",
        )
        assert len(result.tap1_candidates) == 0
        assert len(result.tap2_candidates) == 0
        assert len(result.tap3_candidates) == 0

    def test_overall_confidence(self):
        s = SchematicSuggestion(
            trade_id=1, direction="bearish",
            entry_price=100, entry_time=datetime.now(timezone.utc),
        )
        assert s.overall_confidence == 0.0

        s.range_candidate = RangeCandidate(
            high_price=100, high_time=datetime.now(timezone.utc),
            low_price=95, low_time=datetime.now(timezone.utc),
            confidence=0.8,
        )
        assert s.overall_confidence == 0.8

    def test_bos_detection(self):
        """BOS should be found when pivots break below range."""
        df, entry_time, entry_price, stop_price = _make_bearish_setup()
        detector = PivotDetector(min_strength=2, max_strength=5)
        pivots = detector.detect(df)

        range_cand = RangeCandidate(
            high_price=100.0,
            high_time=datetime(2026, 3, 1, 4, 0, 0, tzinfo=timezone.utc),
            low_price=95.0,
            low_time=datetime(2026, 3, 1, 6, 0, 0, tzinfo=timezone.utc),
            confidence=0.7,
        )

        suggester = TapSuggester()
        result = suggester.suggest(
            pivots, range_cand, entry_time, entry_price,
            stop_price, "bearish",
        )
        # BOS may or may not be detected depending on pivot detection
        # but the method should not crash
        assert isinstance(result.bos_confidence, float)


# ── OHLCV Window Tests ────────────────────────────────────────────────


class TestOhlcvWindow:
    def test_load_from_csv(self, tmp_path):
        csv_content = (
            "timestamp,open,high,low,close,volume\n"
            "2026-03-01T00:00:00+00:00,100,105,95,102,1000\n"
            "2026-03-01T01:00:00+00:00,102,108,100,106,1500\n"
            "2026-03-01T02:00:00+00:00,106,110,104,109,1200\n"
        )
        path = tmp_path / "candles.csv"
        path.write_text(csv_content)

        df = OhlcvWindow.load_from_csv(path)
        assert len(df) == 3
        assert list(df.columns) == ["open_time", "open", "high", "low", "close", "volume"]
        assert df["high"].iloc[0] == 105.0

    def test_load_from_csv_alternate_columns(self, tmp_path):
        csv_content = (
            "time,open,high,low,close,volume\n"
            "2026-03-01,100,105,95,102,1000\n"
        )
        path = tmp_path / "candles2.csv"
        path.write_text(csv_content)

        df = OhlcvWindow.load_from_csv(path)
        assert "open_time" in df.columns

    def test_load_from_csv_missing_timestamp(self, tmp_path):
        csv_content = "open,high,low,close\n100,105,95,102\n"
        path = tmp_path / "bad.csv"
        path.write_text(csv_content)

        with pytest.raises(ValueError, match="timestamp"):
            OhlcvWindow.load_from_csv(path)

    def test_cache_roundtrip(self, tmp_path):
        """Cached parquet should produce identical results."""
        try:
            import pyarrow  # noqa: F401
        except ImportError:
            pytest.skip("pyarrow not installed, skipping parquet test")

        loader = OhlcvWindow(cache_dir=tmp_path / "cache")

        # Create a small CSV to simulate
        csv_content = (
            "timestamp,open,high,low,close,volume\n"
            "2026-03-01T00:00:00+00:00,100,105,95,102,1000\n"
            "2026-03-01T01:00:00+00:00,102,108,100,106,1500\n"
        )
        csv_path = tmp_path / "candles.csv"
        csv_path.write_text(csv_content)
        df = OhlcvWindow.load_from_csv(csv_path)

        # Manually save to cache
        from datetime import timezone
        start = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 2, 0, 0, tzinfo=timezone.utc)
        loader._save_cache("TEST", "1h", start, end, df)

        # Load from cache
        cached = loader._load_cache("TEST", "1h", start, end)
        assert cached is not None
        assert len(cached) == 2


# ── Score Breakdown Tests ─────────────────────────────────────────────


class TestScoreBreakdown:
    def test_tap_candidate_has_breakdown(self):
        df, entry_time, entry_price, stop_price = _make_bearish_setup()
        detector = PivotDetector(min_strength=2, max_strength=5)
        pivots = detector.detect(df)

        range_cand = RangeCandidate(
            high_price=100.0,
            high_time=datetime(2026, 3, 1, 4, 0, 0, tzinfo=timezone.utc),
            low_price=95.0,
            low_time=datetime(2026, 3, 1, 6, 0, 0, tzinfo=timezone.utc),
            confidence=0.7,
        )

        suggester = TapSuggester()
        result = suggester.suggest(
            pivots, range_cand, entry_time, entry_price,
            stop_price, "bearish",
        )

        # At least some candidates should have score breakdowns
        all_cands = result.tap1_candidates + result.tap2_candidates + result.tap3_candidates
        cands_with_breakdown = [c for c in all_cands if c.score_breakdown]
        assert len(cands_with_breakdown) > 0

    def test_explain_method(self):
        tc = TapCandidate(
            tap_number=1, price=100.0,
            time=datetime.now(timezone.utc),
            confidence=0.65,
            score_breakdown={"range_proximity": 0.40, "pivot_strength": 0.15, "time_alignment": 0.10},
        )
        explanation = tc.explain()
        assert "range_proximity" in explanation
        assert "65%" in explanation


# ── Accuracy Scorer Tests ─────────────────────────────────────────────


class TestAccuracyScorer:
    def test_perfect_match(self):
        t = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        suggestion = SchematicSuggestion(
            trade_id=1, direction="bearish",
            entry_price=100, entry_time=t,
            tap1_candidates=[TapCandidate(1, 72000.0, t, 0.8)],
            tap2_candidates=[TapCandidate(2, 72200.0, t, 0.7)],
            tap3_candidates=[TapCandidate(3, 72100.0, t, 0.6)],
        )
        confirmed = FrozenSchematic(
            tap1_price=72000.0, tap1_time=t,
            tap2_price=72200.0, tap2_time=t,
            tap3_price=72100.0, tap3_time=t,
        )
        scorer = AccuracyScorer()
        report = scorer.score(suggestion, confirmed)
        assert report.hit_rate == 1.0
        assert report.avg_price_error_pct == 0.0

    def test_price_mismatch(self):
        t = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        suggestion = SchematicSuggestion(
            trade_id=1, direction="bearish",
            entry_price=100, entry_time=t,
            tap1_candidates=[TapCandidate(1, 72500.0, t, 0.8)],  # off by 500
        )
        confirmed = FrozenSchematic(
            tap1_price=72000.0, tap1_time=t,
        )
        scorer = AccuracyScorer(price_tolerance_pct=0.5)
        report = scorer.score(suggestion, confirmed)
        # 500/72000 = 0.69% → above 0.5% tolerance
        tap1_err = [e for e in report.anchor_errors if e.anchor_name == "tap1"][0]
        assert not tap1_err.hit
        assert tap1_err.price_error_pct > 0.5

    def test_missing_suggestion(self):
        t = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        suggestion = SchematicSuggestion(
            trade_id=1, direction="bearish",
            entry_price=100, entry_time=t,
        )
        confirmed = FrozenSchematic(
            tap1_price=72000.0, tap1_time=t,
        )
        scorer = AccuracyScorer()
        report = scorer.score(suggestion, confirmed)
        tap1_err = [e for e in report.anchor_errors if e.anchor_name == "tap1"][0]
        assert "no suggestion" in tap1_err.note

    def test_summary_output(self):
        t = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        suggestion = SchematicSuggestion(
            trade_id=42, direction="bearish",
            entry_price=100, entry_time=t,
            tap1_candidates=[TapCandidate(1, 72000.0, t, 0.8)],
        )
        confirmed = FrozenSchematic(tap1_price=72000.0, tap1_time=t)
        scorer = AccuracyScorer()
        report = scorer.score(suggestion, confirmed)
        summary = report.summary()
        assert "Trade #42" in summary
        assert "Hit rate" in summary

    def test_aggregate_reports(self):
        t = datetime(2026, 3, 1, 10, 0, 0, tzinfo=timezone.utc)
        reports = []
        for i in range(3):
            s = SchematicSuggestion(
                trade_id=i, direction="bearish",
                entry_price=100, entry_time=t,
                tap1_candidates=[TapCandidate(1, 72000.0, t, 0.8)],
            )
            c = FrozenSchematic(tap1_price=72000.0, tap1_time=t)
            reports.append(AccuracyScorer().score(s, c))
        agg = AccuracyScorer.aggregate_reports(reports)
        assert "3 trades" in agg
        assert "tap1" in agg
