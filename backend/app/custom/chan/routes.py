"""缠论分析路由: /api/custom/chan (日/周/月) 与 /api/custom/chan/minute (分钟级)。"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Annotated

import polars as pl
from fastapi import APIRouter, HTTPException, Query, Request

from app.custom.chan import analysis
from app.market_time import cn_now
from app.services import kline_sync

_MINUTE_PERIODS = (1, 5, 10, 15, 30, 60, 120)


def get_index_chan(
    request: Request,
    symbol: Annotated[str, Query(description="指数代码, 如 000001.SH")],
    start_date: Annotated[date, Query(description="起始日期 YYYY-MM-DD")],
    end_date: Annotated[date, Query(description="截止日期 YYYY-MM-DD")],
):
    """基于本地指数日 K 返回日、周、月多级别缠论结构。"""
    if start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date 不能晚于 end_date")
    df = request.app.state.repo.get_index_daily(symbol, start_date, end_date)
    if df.is_empty():
        return {"symbol": symbol, "engine": "none", "alignment": "mixed", "levels": []}
    return analysis.analyze_levels(df, symbol)


def _minute_window_start(repo, symbol: str, end: datetime, trading_days: int) -> datetime:
    """往回取 trading_days 个指数交易日。本地日K不够时按 7/5 换算自然日。"""
    span = int(trading_days * 7 / 5) + 10
    fallback = end - timedelta(days=span)
    if repo is None or not hasattr(repo, "get_index_daily"):
        return fallback
    try:
        daily = repo.get_index_daily(symbol, fallback.date(), end.date(), columns=["date"])
    except Exception:
        return fallback
    if daily.is_empty() or "date" not in daily.columns:
        return fallback
    dates = daily.get_column("date").drop_nulls().unique().sort()
    if dates.len() == 0:
        return fallback
    picked = dates.item(-min(trading_days, dates.len()))
    return datetime.combine(picked, time.min, tzinfo=end.tzinfo)


def get_index_chan_minute(
    request: Request,
    symbol: Annotated[str, Query(description="指数代码, 如 000001.SH")],
    days: Annotated[int, Query(ge=5, le=120)] = 45,
):
    """返回 1F~120F 缠论。优先本地 1 分钟, 否则从最细可直取周期各合成一次。"""
    end = cn_now()
    repo = getattr(request.app.state, "repo", None)
    start = _minute_window_start(repo, symbol, end, days)
    capset = getattr(request.app.state, "capabilities", None)
    local = pl.DataFrame()
    if repo is not None and hasattr(repo, "get_minute_range"):
        try:
            local = repo.get_minute_range([symbol], start.date(), end.date(), asset_type="index")
        except Exception:
            local = pl.DataFrame()

    frames: dict[int, tuple[pl.DataFrame, str, str]] = {}
    base_period: int | None = None
    base_df: pl.DataFrame | None = None
    if not local.is_empty():
        base_period, base_df = 1, local
    for period in _MINUTE_PERIODS:
        if base_df is not None and base_period is not None and period % base_period == 0:
            if period == base_period:
                frames[period] = (base_df, "direct", f"{period}F")
            else:
                frames[period] = (
                    analysis.resample_minute(base_df, period), "synthetic", f"{base_period}F",
                )
            continue
        direct = kline_sync.fetch_minute_period(
            symbol, start, end, period, "index", capset=capset,
        )
        if direct.is_empty():
            frames[period] = (pl.DataFrame(), "none", "--")
            continue
        base_period, base_df = period, direct
        frames[period] = (direct, "direct", f"{period}F")
    return analysis.analyze_minute_levels(frames, symbol)


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/custom/chan", tags=["custom-chan"])
    router.add_api_route("", get_index_chan, methods=["GET"])
    router.add_api_route("/minute", get_index_chan_minute, methods=["GET"])
    return router
