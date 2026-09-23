from __future__ import annotations

from datetime import date, datetime, timedelta
from math import sin
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.custom.chan import analysis as chan_analysis
from app.custom.chan.routes import _minute_window_start, get_index_chan, get_index_chan_minute
from app.extensions.loader import configure_backend_extensions
from app.market_time import CN_TZ


def _sample(rows: int = 1200) -> pl.DataFrame:
    records = []
    for index in range(rows):
        close = 100 + 12 * sin(index / 9) + 2 * sin(index / 3)
        open_ = close + sin(index)
        records.append(
            {
                "date": date(2022, 1, 1) + timedelta(days=index),
                "open": open_,
                "high": max(open_, close) + 1,
                "low": min(open_, close) - 1,
                "close": close,
                "volume": 1000 + index,
            }
        )
    return pl.DataFrame(records)


def _minute_sample(days: int = 2) -> pl.DataFrame:
    records = []
    for day in range(days):
        trade_day = date(2026, 8, 20) + timedelta(days=day)
        times = [datetime.combine(trade_day, datetime.min.time()) + timedelta(hours=9, minutes=30 + minute) for minute in range(121)]
        times += [datetime.combine(trade_day, datetime.min.time()) + timedelta(hours=13, minutes=minute) for minute in range(1, 121)]
        for index, dt in enumerate(times):
            value = 100 + day + index / 100
            records.append({
                "symbol": "000001.SH", "datetime": dt,
                "open": value, "high": value + 1, "low": value - 1, "close": value + 0.5,
                "volume": 10.0, "amount": 1000.0,
            })
    return pl.DataFrame(records)


def test_analyze_levels_fallback_builds_linked_entities(monkeypatch):
    monkeypatch.setattr(chan_analysis, "_czsc", None)

    result = chan_analysis.analyze_levels(_sample(), "000001.SH")

    assert result["engine"] == "builtin"
    assert [level["key"] for level in result["levels"]] == ["daily", "weekly", "monthly"]
    assert len(result["levels"][0]["bars"]) == 1200
    assert len(result["levels"][0]["bars"]) > len(result["levels"][1]["bars"]) > len(result["levels"][2]["bars"])
    for level in result["levels"]:
        dates = {row["date"] for row in level["bars"]}
        assert all(pen["start"] in dates and pen["end"] in dates for pen in level["pens"])
        assert all(center["lower"] < center["upper"] for center in level["centers"])


def test_analyze_levels_rejects_missing_ohlc():
    with pytest.raises(ValueError, match="high"):
        chan_analysis.analyze_levels(pl.DataFrame({"date": [date.today()], "open": [1], "low": [1], "close": [1]}), "X")


def test_analyze_levels_uses_czsc():
    result = chan_analysis.analyze_levels(_sample(500), "000001.SH")

    assert result["engine"].startswith("czsc-")
    assert result["levels"][0]["pens"]


def test_index_chan_api_uses_index_repository(monkeypatch):
    monkeypatch.setattr(chan_analysis, "_czsc", None)
    repo = SimpleNamespace(get_index_daily=lambda symbol, start, end: _sample(120))
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repo=repo)))

    result = get_index_chan(request, "000001.SH", date(2024, 1, 1), date(2024, 12, 31))

    assert result["symbol"] == "000001.SH"
    assert result["levels"][0]["bars"]


def test_index_chan_api_rejects_reversed_range():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(repo=None)))

    with pytest.raises(HTTPException, match="start_date"):
        get_index_chan(request, "000001.SH", date(2025, 1, 2), date(2025, 1, 1))


def test_resample_minute_respects_cn_sessions():
    result = chan_analysis.resample_minute(_minute_sample(1), 120)

    assert result.height == 2
    assert result["datetime"].to_list() == [
        datetime(2026, 8, 20, 11, 30),
        datetime(2026, 8, 20, 15, 0),
    ]
    assert result["volume"].to_list() == [1210.0, 1200.0]


def test_index_chan_minute_resamples_from_finest_direct(monkeypatch):
    monkeypatch.setattr(chan_analysis, "_czsc", None)
    one_minute = _minute_sample()
    called: list[int] = []

    def fake_fetch(symbol, start_time, end_time, period, asset_type, capset=None):
        assert symbol == "000001.SH"
        assert asset_type == "index"
        assert capset == "cap"
        called.append(period)
        return one_minute if period == 1 else pl.DataFrame()

    monkeypatch.setattr("app.services.kline_sync.fetch_minute_period", fake_fetch)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(capabilities="cap")))

    result = get_index_chan_minute(request, "000001.SH", 45)

    assert called == [1]
    assert [level["key"] for level in result["levels"]] == ["1f", "5f", "10f", "15f", "30f", "60f", "120f"]
    assert [(level["source"], level["source_period"]) for level in result["levels"]] == [
        ("direct", "1F"),
        *([("synthetic", "1F")] * 6),
    ]
    assert result["levels"][-1]["bars"][-2]["date"].endswith("11:30")
    assert result["levels"][-1]["bars"][-1]["date"].endswith("15:00")


def test_index_chan_minute_falls_back_to_coarser_direct(monkeypatch):
    monkeypatch.setattr(chan_analysis, "_czsc", None)
    five_minute = chan_analysis.resample_minute(_minute_sample(), 5)
    called: list[int] = []

    def fake_fetch(symbol, start_time, end_time, period, asset_type, capset=None):
        called.append(period)
        return five_minute if period == 5 else pl.DataFrame()

    monkeypatch.setattr("app.services.kline_sync.fetch_minute_period", fake_fetch)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    result = get_index_chan_minute(request, "000001.SH", 45)

    assert called == [1, 5]
    assert [(level["source"], level["source_period"]) for level in result["levels"] if level["key"] != "1f"] == [
        ("direct", "5F"),
        *([("synthetic", "5F")] * 5),
    ]


def test_index_chan_minute_allows_custom_120f(monkeypatch):
    monkeypatch.setattr(chan_analysis, "_czsc", None)
    bars = chan_analysis.resample_minute(_minute_sample(1), 120)
    called: list[int] = []

    def fake_fetch(symbol, start_time, end_time, period, asset_type, capset=None):
        called.append(period)
        return bars if period == 120 else pl.DataFrame()

    monkeypatch.setattr("app.services.kline_sync.fetch_minute_period", fake_fetch)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))

    result = get_index_chan_minute(request, "000001.SH", 45)

    assert called[-1] == 120
    assert result["levels"][-1]["source"] == "direct"


def test_alignment_ignores_flat_levels():
    assert chan_analysis._alignment([
        {"direction": "up"}, {"direction": "flat"}, {"direction": "up"},
    ]) == "up"
    assert chan_analysis._alignment([{"direction": "flat"}]) == "mixed"
    assert chan_analysis._alignment([
        {"direction": "up"}, {"direction": "down"},
    ]) == "mixed"


def test_minute_window_uses_index_sessions():
    sessions = [date(2026, 8, 3) + timedelta(days=i) for i in range(10)]
    repo = SimpleNamespace(get_index_daily=lambda symbol, start, end, columns=None: pl.DataFrame({"date": sessions}))
    end = datetime(2026, 8, 12, 10, 0, tzinfo=CN_TZ)

    start = _minute_window_start(repo, "000001.SH", end, 3)

    assert start.date() == sessions[-3]
    assert start.tzinfo == CN_TZ


def test_extension_registers_routes_via_loader() -> None:
    """chan 扩展经 custom 加载器自动注册到 /api/custom/chan, 空数据 fail-soft。"""
    app = FastAPI()
    registry, errors = configure_backend_extensions(app)
    assert "indices.chan" in registry.extension_ids()
    assert errors == ()

    app.state.repo = SimpleNamespace(get_index_daily=lambda symbol, start, end: pl.DataFrame())
    client = TestClient(app)
    resp = client.get("/api/custom/chan", params={
        "symbol": "000001.SH", "start_date": "2026-01-01", "end_date": "2026-01-02",
    })
    assert resp.status_code == 200
    assert resp.json()["levels"] == []
