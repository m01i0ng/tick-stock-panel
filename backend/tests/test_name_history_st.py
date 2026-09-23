"""ST 涨跌停档位按名称历史 asof 判定回归。

背景: _is_st 取自 instruments 当前名称, 对全部历史行回溯 — 历史上摘帽的
股票 ST 期间被套 10% 档 (真 5% 涨停漏判), 当前 ST 历史未戴帽的股票全程
5% 档 (普通大涨误判涨停)。数据源无历史名称, 修复引入本地逐日名称历史表
(as_of, symbol, name), 按行日期 asof 取名, 无记录回退当前名(冷启动兼容)。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from app.indicators.pipeline import compute_limit_signals
from app.services.instrument_sync import append_name_history

D0 = date(2024, 1, 2)


def _two_day(symbol: str, prev_close: float, today_close: float, trade_date: date) -> pl.DataFrame:
    return pl.DataFrame({
        "symbol": [symbol, symbol],
        "date": [trade_date - timedelta(days=1), trade_date],
        "raw_close": [prev_close, today_close],
        "close": [prev_close, today_close],
        "raw_high": [prev_close, today_close],
        "raw_low": [prev_close, today_close],
        "open": [prev_close, today_close],
        "high": [prev_close, today_close],
        "low": [prev_close, today_close],
        "volume": [1000.0, 1000.0],
        "change_pct": [0.0, today_close / prev_close - 1],
        "vol_ratio_5d": [1.0, 1.0],
    })


def test_st_flag_uses_name_history_asof():
    inst = pl.DataFrame({
        "symbol": ["600000.SH", "600001.SH", "600002.SH"],
        "name": ["北大荒", "平安银行", "ST 方大"],
    })
    name_history = pl.DataFrame({
        "as_of": [D0, date(2024, 1, 4)],
        "symbol": ["600000.SH", "600000.SH"],
        "name": ["ST 北大荒", "北大荒"],
    })
    df = pl.concat([
        _two_day("600000.SH", 10.0, 10.55, date(2024, 1, 3)),   # asof 名 "ST 北大荒"
        _two_day("600000.SH", 10.0, 10.55, date(2024, 1, 8)),   # asof 名 "北大荒"(已摘帽)
        _two_day("600001.SH", 10.0, 10.55, date(2024, 1, 3)),   # 无历史记录 → 当前名
        _two_day("600002.SH", 10.0, 10.55, date(2024, 1, 3)),   # 无历史记录 → 当前 ST 名
    ]).sort(["symbol", "date"])

    out = compute_limit_signals(df, inst, name_history=name_history).sort(["symbol", "date"])

    def _sig(symbol: str, day: date) -> bool:
        row = out.filter((pl.col("symbol") == symbol) & (pl.col("date") == day))
        return row["signal_limit_up"][0]

    # +5.5%: ST 5% 档 (限价 10.50) → 涨停; 主板 10% 档 (限价 11.00) → 非涨停
    assert _sig("600000.SH", date(2024, 1, 3)) is True    # 历史名含 ST → 5% 档
    assert _sig("600000.SH", date(2024, 1, 8)) is False   # 摘帽后 → 10% 档
    assert _sig("600001.SH", date(2024, 1, 3)) is False   # 无记录回退当前名(非 ST)
    assert _sig("600002.SH", date(2024, 1, 3)) is True    # 无记录回退当前名(ST)


def test_st_flag_without_name_history_keeps_current_name_behavior():
    inst = pl.DataFrame({"symbol": ["600000.SH"], "name": ["北大荒"]})
    df = _two_day("600000.SH", 10.0, 10.55, date(2024, 1, 3))

    out = compute_limit_signals(df, inst, name_history=None)

    assert out["signal_limit_up"].to_list()[-1] is False


def test_append_name_history_is_daily_and_idempotent(tmp_path):
    day1 = pl.DataFrame({"symbol": ["600000.SH", "600001.SH"], "name": ["ST 北大荒", "平安银行"]})
    append_name_history(tmp_path, day1, as_of=D0)
    # 同日重复同步: 覆盖同键, 不产生重复行
    append_name_history(tmp_path, day1, as_of=D0)
    history = pl.read_parquet(tmp_path / "instruments" / "name_history.parquet")
    assert history.height == 2
    assert set(history.columns) == {"as_of", "symbol", "name"}

    # 次日摘帽: 追加新行, 旧行保留
    day2 = pl.DataFrame({"symbol": ["600000.SH"], "name": ["北大荒"]})
    append_name_history(tmp_path, day2, as_of=D0 + timedelta(days=2))
    history = pl.read_parquet(tmp_path / "instruments" / "name_history.parquet")
    assert history.height == 3
    assert (
        history.filter(pl.col("symbol") == "600000.SH").sort("as_of")["name"].to_list()
        == ["ST 北大荒", "北大荒"]
    )
