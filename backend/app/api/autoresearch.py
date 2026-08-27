"""Catalog-only autoresearch session API."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Mapping
from datetime import date
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sse_starlette.sse import EventSourceResponse

from app.backtest.factor import FACTOR_COLUMNS
from app.backtest.mining import MAX_BEAM_WIDTH, MAX_COMBINATION_SIZE, MAX_FINALISTS
from app.services.mining_preflight import (
    MiningPreflightError,
    reserve_final_holdout,
)
from app.services.research_sessions import (
    SESSION_STATUSES,
    TERMINAL_SESSION_STATUSES,
    InvalidResearchSessionTransitionError,
    ResearchSessionConflictError,
    ResearchSessionError,
    ResearchSessionValidationError,
    catalog_strategies,
    generate_catalog_proposal,
)

router = APIRouter(prefix="/api/backtest/autoresearch/sessions", tags=["backtest"])
logger = logging.getLogger(__name__)
_SSE_POLL_SECONDS = 0.5
_SSE_HEARTBEAT_SECONDS = 15.0


class ResearchSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    goal: str = Field(min_length=4, max_length=2000)
    asset_type: Literal["stock", "etf"] = "stock"
    start: date | None = None
    end: date | None = None
    budget_profile: Literal["exploratory", "balanced", "strict"] = "balanced"
    commission_pct: float = Field(0.0002, ge=0.0, le=0.05, allow_inf_nan=False)
    stamp_tax_pct: float = Field(0.0005, ge=0.0, le=0.05, allow_inf_nan=False)
    slippage_bps: float = Field(5.0, ge=0.0, le=1000.0, allow_inf_nan=False)
    correlation_threshold: float = Field(0.75, gt=0.0, le=1.0, allow_inf_nan=False)
    max_combination_factors: int = Field(4, ge=1, le=MAX_COMBINATION_SIZE)
    beam_width: int = Field(12, ge=1, le=MAX_BEAM_WIDTH)
    max_finalists: int = Field(MAX_FINALISTS, ge=1, le=MAX_FINALISTS)
    max_trials: int = Field(3, ge=1, le=10)
    max_wall_minutes: int = Field(60, ge=1, le=360)
    patience: int = Field(2, ge=1, le=10)

    @model_validator(mode="after")
    def _valid_ranges(self) -> ResearchSessionCreateRequest:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("start must not be after end")
        if self.patience > self.max_trials:
            raise ValueError("patience must not exceed max_trials")
        return self


@router.post("")
async def create_session(
    payload: ResearchSessionCreateRequest,
    request: Request,
) -> dict[str, Any]:
    from app.services.ai_provider import ai_configured

    if not ai_configured():
        raise HTTPException(status_code=503, detail="AI 未配置, 请先在设置页配置")
    manager = _manager(request)
    try:
        manager.reserve_creation()
    except ResearchSessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    try:
        try:
            holdout = reserve_final_holdout(
                request.app.state.repo.store.data_dir,
                asset_type=payload.asset_type,
                budget_profile=payload.budget_profile,
                start=payload.start,
                end=payload.end,
            )
            strategies = catalog_strategies(
                request.app.state.strategy_engine,
                payload.asset_type,
            )
        except MiningPreflightError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
                headers={"X-Mining-Preflight-Code": exc.code},
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            proposal = await generate_catalog_proposal(
                goal=payload.goal,
                factors=FACTOR_COLUMNS,
                strategies=strategies,
                asset_type=payload.asset_type,
                budget_profile=payload.budget_profile,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"AI 返回的自动研究计划无效: {exc}",
            ) from exc
        except Exception as exc:
            logger.warning("failed to generate initial autoresearch proposal: %s", exc)
            raise HTTPException(
                status_code=502,
                detail=f"AI 自动研究计划生成失败: {exc}",
            ) from exc
        try:
            session = manager.create_session(
                payload.model_dump(mode="json"),
                proposal,
                holdout=holdout,
            )
            return _project_session(session)
        except ResearchSessionConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ResearchSessionValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        manager.release_creation()


@router.get("")
def list_sessions(
    request: Request,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    status: Annotated[list[str] | None, Query()] = None,
) -> dict[str, Any]:
    if status and not set(status) <= SESSION_STATUSES:
        raise HTTPException(status_code=400, detail="unsupported autoresearch session status")
    try:
        items = _manager(request).store.list(limit=limit, statuses=status)
        return {"items": [_project_session(item) for item in items]}
    except ResearchSessionError as exc:
        raise HTTPException(status_code=500, detail="failed to read autoresearch sessions") from exc


@router.get("/{session_id}")
def get_session(session_id: str, request: Request) -> dict[str, Any]:
    return _project_session(_required_session(request, session_id))


@router.post("/{session_id}/approve")
def approve_session(session_id: str, request: Request) -> dict[str, Any]:
    return _control(request, session_id, "approve")


@router.post("/{session_id}/pause")
def pause_session(session_id: str, request: Request) -> dict[str, Any]:
    return _control(request, session_id, "pause")


@router.post("/{session_id}/resume")
def resume_session(session_id: str, request: Request) -> dict[str, Any]:
    return _control(request, session_id, "resume")


@router.post("/{session_id}/stop")
def stop_session(session_id: str, request: Request) -> dict[str, Any]:
    return _control(request, session_id, "stop")


class ResearchSessionSealRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    trial_index: int | None = Field(None, ge=1)


@router.post("/{session_id}/seal")
def seal_session(
    session_id: str,
    request: Request,
    payload: ResearchSessionSealRequest | None = None,
) -> dict[str, Any]:
    trial_index = None if payload is None else payload.trial_index
    manager = _manager(request)
    try:
        return _project_session(manager.seal(session_id, trial_index=trial_index))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="autoresearch session not found") from exc
    except ResearchSessionConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (InvalidResearchSessionTransitionError, ResearchSessionValidationError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MiningPreflightError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
            headers={"X-Mining-Preflight-Code": exc.code},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/{session_id}/events")
def stream_session_events(
    session_id: str,
    request: Request,
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
) -> EventSourceResponse:
    session = _required_session(request, session_id)
    try:
        cursor = max(0, int(last_event_id or 0))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Last-Event-ID must be an integer") from exc
    store = _manager(request).store

    async def generate() -> AsyncIterator[dict[str, str]]:
        nonlocal cursor
        yield {
            "event": "snapshot",
            "data": json.dumps(_project_session(session), ensure_ascii=False, allow_nan=False),
        }
        last_emit = asyncio.get_running_loop().time()
        while True:
            if await request.is_disconnected():
                return
            events = await asyncio.to_thread(store.read_events, session_id, after_id=cursor)
            for event in events:
                cursor = int(event["id"])
                event_type = "failed" if event["type"] == "error" else str(event["type"])
                yield {
                    "id": str(cursor),
                    "event": event_type,
                    "data": json.dumps(event.get("payload") or {}, ensure_ascii=False, allow_nan=False),
                }
                last_emit = asyncio.get_running_loop().time()
            current = await asyncio.to_thread(store.get, session_id)
            if current is None or current["status"] in TERMINAL_SESSION_STATUSES:
                return
            now = asyncio.get_running_loop().time()
            if now - last_emit >= _SSE_HEARTBEAT_SECONDS:
                yield {"event": "heartbeat", "data": "{}"}
                last_emit = now
            await asyncio.sleep(_SSE_POLL_SECONDS)

    return EventSourceResponse(generate(), ping=_SSE_HEARTBEAT_SECONDS)


def _control(request: Request, session_id: str, action: str) -> dict[str, Any]:
    manager = _manager(request)
    try:
        method = getattr(manager, action)
        return _project_session(method(session_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="autoresearch session not found") from exc
    except (InvalidResearchSessionTransitionError, ResearchSessionValidationError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except MiningPreflightError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
            headers={"X-Mining-Preflight-Code": exc.code},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _manager(request: Request):
    manager = getattr(request.app.state, "autoresearch_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="autoresearch manager is unavailable")
    return manager


def _required_session(request: Request, session_id: str) -> dict[str, Any]:
    try:
        session = _manager(request).store.get(session_id)
    except ResearchSessionValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ResearchSessionError as exc:
        raise HTTPException(status_code=500, detail="failed to read autoresearch session") from exc
    if session is None:
        raise HTTPException(status_code=404, detail="autoresearch session not found")
    return session


def _project_session(session: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: session.get(key)
        for key in (
            "session_id",
            "status",
            "goal",
            "asset_type",
            "start",
            "end",
            "adaptive_end",
            "holdout",
            "budget_profile",
            "commission_pct",
            "stamp_tax_pct",
            "slippage_bps",
            "correlation_threshold",
            "max_combination_factors",
            "beam_width",
            "max_finalists",
            "max_trials",
            "max_wall_minutes",
            "patience",
            "created_at",
            "updated_at",
            "started_at",
            "finished_at",
            "approved_at",
            "current_trial",
            "active_run_id",
            "completed_trials",
            "no_improvement_trials",
            "data_snapshot_digest",
            "stop_reason",
            "error",
            "initial_proposal",
            "trials",
            "leaderboard",
        )
    }
