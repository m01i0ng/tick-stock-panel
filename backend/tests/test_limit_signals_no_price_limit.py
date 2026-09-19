"""维表涨停价为「无涨跌停限制」哨兵值 (>= 10000) 时, 盘后全量与实时路径同口径不判涨跌停。

注册制新股上市前 5 个交易日无涨跌幅限制, 数据源在维表 limit_up 填上万的占位值。
实时路径 (_compute_limit_signals_today) 识别该哨兵后涨停/跌停/炸板/翘板一律为 False;
盘后全量 (compute_limit_signals) 只把哨兵排除出「权威价」, 随后回退按 10%/20%/30%
理论价判断, 于是新股当日涨超 10% 就被记为涨停、连板数 +1, 收盘后看板涨停家数、
连板梯队与涨停类策略都会把它算进去, 与盘中结论相反。
"""
from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from app.indicators import pipeline

YESTERDAY = date(2026, 9, 16)
TODAY = date(2026, 9, 17)


def _rows(prev_close: float, *, open_: float, high: float, low: float, close: float) -> pl.DataFrame:
    return pl.DataFrame({
        "symbol": ["001234.SZ", "001234.SZ"],
        "date": [YESTERDAY, TODAY],
        "open": [prev_close, open_],
        "high": [prev_close, high],
        "low": [prev_close, low],
        "close": [prev_close, close],
        "raw_close": [prev_close, close],
        "raw_high": [prev_close, high],
        "raw_low": [prev_close, low],
        "volume": [1000.0, 1000.0],
    })


def _instruments(limit_up: float, limit_down: float, as_of: date = TODAY) -> pl.DataFrame:
    return pl.DataFrame({
        "symbol": ["001234.SZ"],
        "name": ["新股"],
        "limit_up": [limit_up],
        "limit_down": [limit_down],
        "as_of": [as_of],
    })


def _realtime(rows: pl.DataFrame, instruments: pl.DataFrame) -> dict:
    today = rows.filter(pl.col("date") == TODAY).with_columns(
        pl.lit(rows["close"][0]).alias("_prev_close_raw"),
    )
    return pipeline._compute_limit_signals_today(today, instruments).row(0, named=True)


# 昨收 20.00, 主板理论涨停 22.00 / 跌停 18.00
CASES = {
    "up_close": dict(open_=21.0, high=23.5, low=20.5, close=23.4),        # 涨超 10% 收盘
    "up_touch_fade": dict(open_=21.0, high=22.5, low=20.5, close=21.5),   # 冲过理论涨停价后回落
    "down_close": dict(open_=19.0, high=19.5, low=17.5, close=17.6),      # 跌超 10% 收盘
    "down_touch_up": dict(open_=18.2, high=19.0, low=17.8, close=18.8),   # 跌破理论跌停价后收阳
}
SIGNALS = (
    "signal_limit_up", "signal_broken_limit_up",
    "signal_limit_down", "signal_limit_down_recovery",
)


@pytest.mark.parametrize("case", list(CASES))
def test_full_path_no_price_limit_sentinel_matches_realtime(case):
    rows = _rows(20.0, **CASES[case])
    instruments = _instruments(limit_up=100000.0, limit_down=0.0)

    full = pipeline.compute_limit_signals(rows, instruments).row(-1, named=True)
    realtime = _realtime(rows, instruments)

    for signal in SIGNALS:
        assert realtime[signal] is False, signal
        assert full[signal] is False, signal
    assert full["consecutive_limit_ups"] == 0
    assert full["consecutive_limit_downs"] == 0


def test_down_signals_alone_also_respect_sentinel():
    rows = _rows(20.0, **CASES["down_close"])
    instruments = _instruments(limit_up=100000.0, limit_down=0.0)

    full = pipeline.compute_limit_signals(
        rows, instruments, needed={"signal_limit_down", "consecutive_limit_downs"},
    ).row(-1, named=True)

    assert full["signal_limit_down"] is False
    assert full["consecutive_limit_downs"] == 0


def test_sentinel_only_applies_to_matching_instrument_date():
    """维表日期不是该行情日 (历史行) 时无法得知是否无限制, 保持理论价口径。"""
    rows = _rows(20.0, **CASES["up_close"])
    stale = _instruments(limit_up=100000.0, limit_down=0.0, as_of=YESTERDAY)

    full = pipeline.compute_limit_signals(rows, stale).row(-1, named=True)

    assert full["signal_limit_up"] is True
    assert full["consecutive_limit_ups"] == 1


def test_regular_stock_limit_up_unchanged():
    rows = _rows(20.0, open_=21.0, high=22.0, low=20.5, close=22.0)
    instruments = _instruments(limit_up=22.0, limit_down=18.0)

    full = pipeline.compute_limit_signals(rows, instruments).row(-1, named=True)

    assert full["signal_limit_up"] is True
    assert full["consecutive_limit_ups"] == 1
