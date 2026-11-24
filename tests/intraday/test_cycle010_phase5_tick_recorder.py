"""CYCLE-010 Phase 5 Tests: Read-Only NSE Tick/Seconds Recorder & Causal Multi-TF Parquet Archiver."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from scripts.record_nse_ticks import main as recorder_cli_main
from trader.config import IST_TZ
from trader.data.tick_recorder import (
    LIVE_TRADING_ENABLED,
    READ_ONLY_MODE,
    SUPPORTED_TIMEFRAMES,
    AngelOneSmartApiReadOnlyAdapter,
    CausalTickAggregator,
    OfflineJsonlTickAdapter,
    ReadOnlyNseTickRecorder,
    TickEvent,
    ZerodhaKiteReadOnlyAdapter,
    align_tick_bucket_ist,
    assert_read_only_recorder_config,
    is_within_nse_session_ist,
    load_recorded_bars_parquet,
    parse_angel_one_tick,
    parse_zerodha_kite_tick,
    sha256_file,
)


def _ist(y: int, m: int, d: int, hh: int, mm: int, ss: int) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=IST_TZ)


class TestReadOnlySafetyInvariants:
    """Verify strictly read-only market-data posture with zero order-routing capabilities."""

    def test_global_safety_constants_and_guard(self) -> None:
        assert LIVE_TRADING_ENABLED is False
        assert READ_ONLY_MODE is True
        assert_read_only_recorder_config(live_trading_enabled=False, read_only=True)

        with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
            assert_read_only_recorder_config(live_trading_enabled=True, read_only=True)
        with pytest.raises(RuntimeError, match="SAFETY VIOLATION"):
            assert_read_only_recorder_config(live_trading_enabled=False, read_only=False)

    def test_zero_order_routing_methods_in_source_and_adapters(self) -> None:
        forbidden = ("place_order", "modify_order", "cancel_order", "placeOrder", "modifyOrder", "cancelOrder")
        for rel in ("trader/data/tick_recorder.py", "scripts/record_nse_ticks.py"):
            src = Path(rel).read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in src, f"Forbidden order-routing symbol {token!r} found in {rel}"

        for adapter_cls in (ZerodhaKiteReadOnlyAdapter, AngelOneSmartApiReadOnlyAdapter, OfflineJsonlTickAdapter):
            for token in forbidden:
                assert not hasattr(adapter_cls, token)


class TestNseSessionFilterAndAlignment:
    """Verify NSE regular-hours filtering and 09:15-15:30 IST causal bucket alignment."""

    def test_nse_session_hours_boundaries(self) -> None:
        # 2026-03-16 is a Monday
        assert is_within_nse_session_ist(_ist(2026, 3, 16, 9, 14, 59)) is False
        assert is_within_nse_session_ist(_ist(2026, 3, 16, 9, 15, 0)) is True
        assert is_within_nse_session_ist(_ist(2026, 3, 16, 15, 29, 59)) is True
        assert is_within_nse_session_ist(_ist(2026, 3, 16, 15, 30, 0)) is False
        # Weekend rejection: 2026-03-14 Saturday, 2026-03-15 Sunday
        assert is_within_nse_session_ist(_ist(2026, 3, 14, 11, 0, 0)) is False
        assert is_within_nse_session_ist(_ist(2026, 3, 15, 11, 0, 0)) is False

    def test_bucket_alignment_and_1530_clamping(self) -> None:
        t0, t1 = align_tick_bucket_ist(_ist(2026, 3, 16, 9, 15, 4), "5s")
        assert t0 == _ist(2026, 3, 16, 9, 15, 0)
        assert t1 == _ist(2026, 3, 16, 9, 15, 5)

        t0_15m, t1_15m = align_tick_bucket_ist(_ist(2026, 3, 16, 15, 29, 58), "15m")
        assert t0_15m == _ist(2026, 3, 16, 15, 15, 0)
        assert t1_15m == _ist(2026, 3, 16, 15, 30, 0)


class TestCausalTickAggregator:
    """Verify multi-timeframe aggregation, out-of-order handling, zero lookahead, and multi-day isolation."""

    def test_out_of_order_ticks_and_ohlcv_exactness(self) -> None:
        agg = CausalTickAggregator(timeframes=("1s", "5s", "1m", "5m", "15m"))
        # Ticks arrive out of chronological order inside the same 5s bucket [09:15:00, 09:15:05)
        ticks = [
            TickEvent(symbol="TCS", timestamp_ist=_ist(2026, 3, 16, 9, 15, 3), ltp=3920.0, volume_delta=50),
            TickEvent(symbol="TCS", timestamp_ist=_ist(2026, 3, 16, 9, 15, 1), ltp=3900.0, volume_delta=100),
            TickEvent(symbol="TCS", timestamp_ist=_ist(2026, 3, 16, 9, 15, 4), ltp=3910.0, volume_delta=25),
            TickEvent(symbol="TCS", timestamp_ist=_ist(2026, 3, 16, 9, 15, 2), ltp=3895.0, volume_delta=40),
            # Pre-open tick must be rejected
            TickEvent(symbol="TCS", timestamp_ist=_ist(2026, 3, 16, 9, 14, 55), ltp=3880.0, volume_delta=10),
        ]
        for t in ticks:
            agg.ingest_tick(t)

        assert agg.accepted_ticks == 4
        assert agg.rejected_outside_hours_ticks == 1
        assert agg.out_of_order_ticks >= 1

        # Watermark at 09:15:04 must NOT flush the 5s bar closing at 09:15:05 (zero lookahead)
        early = agg.flush_closed_bars(_ist(2026, 3, 16, 9, 15, 4))
        assert all(b.timeframe == "1s" for b in early)
        assert len(early) == 3  # 09:15:01, 09:15:02, 09:15:03 closed by 09:15:04

        # Watermark at 09:15:05 flushes 09:15:04 (1s) AND [09:15:00, 09:15:05) (5s)
        at_5s = agg.flush_closed_bars(_ist(2026, 3, 16, 9, 15, 5))
        bar_5s = [b for b in at_5s if b.timeframe == "5s"][0]
        assert bar_5s.open == 3900.0  # earliest chronological tick (09:15:01)
        assert bar_5s.high == 3920.0
        assert bar_5s.low == 3895.0
        assert bar_5s.close == 3910.0  # latest chronological tick (09:15:04)
        assert bar_5s.volume == 215.0
        assert bar_5s.tick_count == 4

        # Finalize flushes remaining 1m, 5m, 15m bars
        rest = agg.finalize_all()
        assert {b.timeframe for b in rest} == {"1m", "5m", "15m"}

    def test_cumulative_volume_monotonic_delta_and_multi_day_isolation(self) -> None:
        agg = CausalTickAggregator(timeframes=("1m",))
        # Day 1: cumulative volume rises 1000 -> 1250 -> 1400
        agg.ingest_tick(
            TickEvent(symbol="RELIANCE", timestamp_ist=_ist(2026, 3, 16, 9, 15, 10), ltp=1280.0, volume_delta=100, cumulative_volume=1000)
        )
        agg.ingest_tick(
            TickEvent(symbol="RELIANCE", timestamp_ist=_ist(2026, 3, 16, 9, 15, 25), ltp=1282.0, cumulative_volume=1250)
        )
        agg.ingest_tick(
            TickEvent(symbol="RELIANCE", timestamp_ist=_ist(2026, 3, 16, 9, 15, 45), ltp=1281.0, cumulative_volume=1400)
        )
        # Day 2: separate date must never merge with Day 1
        agg.ingest_tick(
            TickEvent(symbol="RELIANCE", timestamp_ist=_ist(2026, 3, 17, 9, 15, 10), ltp=1290.0, volume_delta=80, cumulative_volume=500)
        )
        bars = agg.finalize_all()
        assert len(bars) == 2
        assert bars[0].ts_open_ist.date().isoformat() == "2026-03-16"
        assert bars[0].volume == pytest.approx(500.0)  # 100 + 250 + 150
        assert bars[1].ts_open_ist.date().isoformat() == "2026-03-17"
        assert bars[1].volume == pytest.approx(80.0)


class TestBrokerAdaptersAndPayloadParsing:
    """Verify broker credential guardrails and payload normalization without network calls."""

    def test_missing_credentials_fail_loudly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in ("KITE_API_KEY", "KITE_ACCESS_TOKEN", "ANGEL_API_KEY", "ANGEL_CLIENT_CODE", "ANGEL_AUTH_TOKEN", "ANGEL_FEED_TOKEN"):
            monkeypatch.delenv(var, raising=False)

        with pytest.raises(RuntimeError, match="KITE_API_KEY"):
            ZerodhaKiteReadOnlyAdapter({"TCS": 2953217}).validate_credentials()
        with pytest.raises(RuntimeError, match="ANGEL_API_KEY"):
            AngelOneSmartApiReadOnlyAdapter({"TCS": "11536"}).validate_credentials()

    def test_parse_zerodha_and_angel_payloads(self) -> None:
        kite_raw = {
            "instrument_token": 2953217,
            "last_price": 3912.55,
            "volume_traded": 125000,
            "last_traded_quantity": 15,
            "exchange_timestamp": _ist(2026, 3, 16, 10, 5, 12),
            "depth": {"buy": [{"price": 3912.50}], "sell": [{"price": 3912.60}]},
        }
        kt = parse_zerodha_kite_tick(kite_raw, {2953217: "TCS"})
        assert kt is not None
        assert kt.symbol == "TCS"
        assert kt.ltp == 3912.55
        assert kt.bid == 3912.50
        assert kt.ask == 3912.60
        assert kt.provider == "zerodha_kite"

        angel_raw = {
            "token": "11536",
            "last_traded_price": 391255,  # paise
            "volume_trade_for_the_day": 125000,
            "last_traded_quantity": 20,
            "exchange_timestamp": int(_ist(2026, 3, 16, 10, 5, 12).timestamp() * 1000),
        }
        at = parse_angel_one_tick(angel_raw, {"11536": "TCS"})
        assert at is not None
        assert at.symbol == "TCS"
        assert at.ltp == pytest.approx(3912.55)
        assert at.provider == "angel_one_smartapi"

        # Sub-₹100 stock encoded in paise (₹20.50 -> 2050 paise) must convert accurately
        sub100 = parse_angel_one_tick(
            {"token": "9999", "last_traded_price": 2050, "exchange_timestamp": int(_ist(2026, 3, 16, 10, 5, 12).timestamp() * 1000)},
            {"9999": "YESBANK"},
        )
        assert sub100 is not None
        assert sub100.ltp == pytest.approx(20.50)

        # Heartbeat / malformed / zero-price packets return None safely
        assert parse_zerodha_kite_tick({"instrument_token": 2953217}, {2953217: "TCS"}) is None
        assert parse_zerodha_kite_tick({"instrument_token": 2953217, "last_price": 0.0}, {2953217: "TCS"}) is None
        assert parse_angel_one_tick({"token": "11536", "last_traded_price": 0}, {"11536": "TCS"}) is None


class TestParquetArchiveAndCliEndToEnd:
    """Verify end-to-end offline JSONL recording -> Parquet + SHA-256 provenance + CLI."""

    def test_end_to_end_cli_and_parquet_roundtrip(self, tmp_path: Path) -> None:
        jsonl_path = tmp_path / "sample_ticks.jsonl"
        rows = [
            {"symbol": "TCS", "timestamp_ist": "2026-03-16T09:15:01+05:30", "ltp": 3900.0, "volume_delta": 50},
            {"symbol": "TCS", "timestamp_ist": "2026-03-16T09:15:02+05:30", "ltp": 3905.0, "volume_delta": 30},
            {"symbol": "TCS", "timestamp_ist": "2026-03-16T09:15:06+05:30", "ltp": 3902.0, "volume_delta": 20},
            {"symbol": "TCS", "timestamp_ist": "2026-03-16T15:30:05+05:30", "ltp": 3910.0, "volume_delta": 10},  # rejected
        ]
        jsonl_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        out_dir = tmp_path / "ticks_archive"

        # 1. Dry-run check
        rc_dry = recorder_cli_main(
            ["--provider", "offline_jsonl", "--jsonl-input", str(jsonl_path), "--symbols", "TCS", "--dry-run"]
        )
        assert rc_dry == 0

        # 2. Full run
        rc_run = recorder_cli_main(
            [
                "--provider",
                "offline_jsonl",
                "--jsonl-input",
                str(jsonl_path),
                "--symbols",
                "TCS",
                "--output-dir",
                str(out_dir),
            ]
        )
        assert rc_run == 0

        day_dir = out_dir / "TCS" / "2026-03-16"
        prov_path = day_dir / "provenance.json"
        assert prov_path.exists()
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        assert prov["live_trading_enabled"] is False
        assert prov["read_only"] is True
        assert prov["raw_tick_count"] == 3
        assert prov["rejected_outside_hours_ticks"] == 1

        # Verify every Parquet file exists and matches its SHA-256 digest in provenance.json
        for fname, expected_sha in prov["file_sha256"].items():
            fpath = day_dir / fname
            assert fpath.exists()
            assert sha256_file(fpath) == expected_sha

        # Round-trip load 5s Parquet bars
        bars_5s = load_recorded_bars_parquet(day_dir / "bars_5s.parquet")
        assert len(bars_5s) == 2
        assert bars_5s[0].open == 3900.0
        assert bars_5s[0].close == 3905.0
        assert bars_5s[0].volume == 80.0
        assert bars_5s[1].open == 3902.0
        assert bars_5s[1].volume == 20.0
        assert set(prov["bar_counts"].keys()) == set(SUPPORTED_TIMEFRAMES)
