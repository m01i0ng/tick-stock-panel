"""缠论分析路由: /api/custom/chan (日/周/月) 与 /api/custom/chan/minute (分钟级)。"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

import polars as pl
from fastapi import APIRouter, HTTPException, Query, Request

from app.custom.chan import analysis
from app.market_time import cn_now
from app.services import kline_sync


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


def get_index_chan_minute(
    request: Request,
    symbol: Annotated[str, Query(description="指数代码, 如 000001.SH")],
    days: Annotated[int, Query(ge=5, le=120)] = 45,
):
    """返回 1F~120F 缠论;直取失败时从最近可整除的较小周期合成。"""
    end = cn_now()
    start = end - timedelta(days=days)
    frames: dict[int, tuple[pl.DataFrame, str, str]] = {}
    for period in (1, 5, 10, 15, 30, 60, 120):
        direct = kline_sync.fetch_minute_period(symbol, start, end, period, "index")
        if not direct.is_empty():
            frames[period] = (direct, "direct", f"{period}F")
            continue
        candidates = [source for source, (df, _, _) in frames.items() if not df.is_empty() and period % source == 0]
        if not candidates:
            frames[period] = (pl.DataFrame(), "none", "--")
            continue
        source = max(candidates)
        frames[period] = (analysis.resample_minute(frames[source][0], period), "synthetic", f"{source}F")
    return analysis.analyze_minute_levels(frames, symbol)


def build_router() -> APIRouter:
    router = APIRouter(prefix="/api/custom/chan", tags=["custom-chan"])
    router.add_api_route("", get_index_chan, methods=["GET"])
    router.add_api_route("/minute", get_index_chan_minute, methods=["GET"])
    return router
