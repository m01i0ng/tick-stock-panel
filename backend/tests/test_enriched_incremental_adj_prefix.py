"""增量模式历史前缀口径回归。

背景: 增量分支把 enriched 窄表(前复权 OHLC)直接拼进新日K喂给
compute_enriched, 后者先以 close 覆盖 raw_close 再整体复权 — 历史前缀
被二次复权; 当除权事件恰为首个新日期时 _prev_raw_close 取到双重复权价,
涨停参考价过低导致误判涨停并落盘 consecutive_limit_ups。
易感前缀 = 尾行 close != raw_close(事件已折算进历史, 如 adj-only 全量
重算后的落盘态)。修复: 拼接前用 raw_* 列恢复前缀原始价口径。
"""
from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from app.indicators import pipeline

SYMBOL = "600000.SH"
D0 = date(2026, 8, 1)
# 10转10: 除权因子 2.0, 事件日 D5
# 原始收盘: D0..D4 依次 19.0 19.5 20.5 19.8 20.0, D5(除权日) 10.80
# D5 除权基准 = 20.0 / 2 = 10.00, D5 +8% 非涨停; 双重复权路径基准 5.00 → 误判涨停
RAW_CLOSES = [19.0, 19.5, 20.5, 19.8, 20.0, 10.8]
FACTOR = 2.0
EVENT_INDEX = 5


def _kline_rows() -> list[dict]:
    rows = []
    for i, close in enumerate(RAW_CLOSES):
        rows.append({
            "symbol": SYMBOL,
            "date": D0 + timedelta(days=i),
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 1000.0 + i,
            "amount": 100000.0 + i,
            "quote_ts": 0,
        })
    return rows


def _prepare(tmp_path):
    raw = pl.DataFrame(_kline_rows())
    for frame in raw.partition_by("date"):
        out = tmp_path / "kline_daily" / f"date={frame['date'][0]}" / "part.parquet"
        out.parent.mkdir(parents=True)
        frame.write_parquet(out)
    instruments = pl.DataFrame({
        "symbol": [SYMBOL], "name": ["stock"], "float_shares": [1000000.0],
    })
    factors = pl.DataFrame({
        "symbol": [SYMBOL],
        "trade_date": [D0 + timedelta(days=EVENT_INDEX)],
        "ex_factor": [FACTOR],
    })
    for name, frame in (("instruments/all.parquet", instruments), ("adj_factor/all.parquet", factors)):
        out = tmp_path / name
        out.parent.mkdir(parents=True)
        frame.write_parquet(out)

    # 易感 enriched 前缀: adj-only 全量重算后的落盘态
    # (事件已折算: D0..D4 close = raw/2, raw_close = 原始价)
    prefix_rows = []
    for i, close in enumerate(RAW_CLOSES[:EVENT_INDEX]):
        prefix_rows.append({
            "symbol": SYMBOL,
            "date": D0 + timedelta(days=i),
            "open": close / FACTOR,
            "high": (close + 0.05) / FACTOR,
            "low": (close - 0.05) / FACTOR,
            "close": close / FACTOR,
            "volume": 1000.0 + i,
            "amount": 100000.0 + i,
            "raw_close": close,
            "raw_high": close + 0.05,
            "raw_low": close - 0.05,
        })
    prefix = pl.DataFrame(prefix_rows)
    for frame in prefix.partition_by("date"):
        out = tmp_path / "kline_daily_enriched" / f"date={frame['date'][0]}" / "part.parquet"
        out.parent.mkdir(parents=True)
        frame.write_parquet(out)
    return raw, instruments, factors


def test_incremental_prefix_restores_raw_price_basis(tmp_path, monkeypatch):
    raw, instruments, factors = _prepare(tmp_path)
    monkeypatch.setattr(pipeline, "_custom_signal_exprs", {})

    # 全量基准: 同一份数据全量重算的 D5 行
    baseline = pipeline.compute_enriched(
        raw, instruments=instruments, factors=factors,
    ).filter(pl.col("date") == D0 + timedelta(days=EVENT_INDEX))

    written = pipeline.run_pipeline(tmp_path, new_dates_only=True)
    assert written > 0
    incremental = pl.read_parquet(
        str(tmp_path / "kline_daily_enriched" / f"date={D0 + timedelta(days=EVENT_INDEX)}" / "*.parquet"),
    )

    assert incremental.height == 1
    # 双重复权路径的除权基准 = 20/4 = 5.0, 涨停价 5.5 < 10.8 → 误判连板 1
    assert incremental["consecutive_limit_ups"][0] == 0
    compare_cols = [c for c in baseline.columns if c in incremental.columns]
    assert_frame_equal(
        incremental.select(compare_cols).sort("symbol", "date"),
        baseline.select(compare_cols).sort("symbol", "date"),
        check_exact=False,
    )
