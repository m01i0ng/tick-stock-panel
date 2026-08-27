"""Persistent catalog-only autoresearch sessions and strict AI proposals."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Collection, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from app.services.mining_jobs import canonicalize_request
from app.services.research_assistant import build_plan_messages, parse_plan

ResearchSessionStatus = Literal[
    "awaiting_approval",
    "running",
    "paused",
    "stopping",
    "completed",
    "failed",
    "stopped",
    "interrupted",
]

SESSION_STATUSES = frozenset(
    {
        "awaiting_approval",
        "running",
        "paused",
        "stopping",
        "completed",
        "failed",
        "stopped",
        "interrupted",
    }
)
ACTIVE_SESSION_STATUSES = frozenset({"running", "paused", "stopping"})
TERMINAL_SESSION_STATUSES = frozenset({"completed", "failed", "stopped", "interrupted"})
MAX_SESSION_EVENTS = 512
MAX_SESSION_EVENT_BYTES = 16 * 1024

_SCHEMA_VERSION = 1
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_STORE_LOCK = threading.RLock()
_TRANSITIONS: dict[str, frozenset[str]] = {
    "awaiting_approval": frozenset({"running", "stopped"}),
    "running": frozenset({"paused", "stopping", "completed", "failed", "interrupted"}),
    "paused": frozenset({"running", "stopping", "completed", "failed", "interrupted"}),
    "stopping": frozenset({"stopped", "failed", "interrupted"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "stopped": frozenset(),
    "interrupted": frozenset(),
}


class ResearchSessionError(RuntimeError):
    pass


class ResearchSessionValidationError(ResearchSessionError, ValueError):
    pass


class ResearchSessionConflictError(ResearchSessionError):
    pass


class InvalidResearchSessionTransitionError(ResearchSessionError):
    pass


def catalog_proposal_digest(proposal: Mapping[str, Any]) -> str:
    """Digest only catalog execution choices, not mutable explanatory prose."""
    payload = {
        "factor_names": sorted(str(value) for value in proposal.get("factor_names") or []),
        "strategy_ids": sorted(str(value) for value in proposal.get("strategy_ids") or []),
    }
    encoded = json.dumps(
        canonicalize_request(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=32).hexdigest()


def parse_catalog_proposal(
    text: str,
    *,
    factor_ids: Collection[str],
    strategy_ids: Collection[str],
) -> dict[str, Any]:
    proposal = parse_plan(
        text,
        factor_ids=frozenset(factor_ids),
        strategy_ids=frozenset(strategy_ids),
    )
    proposal["digest"] = catalog_proposal_digest(proposal)
    return proposal


def validate_followup_step(previous: Mapping[str, Any], proposal: Mapping[str, Any]) -> None:
    """Allow only one catalog axis per follow-up, and at most one factor add/remove."""
    prev_factors = {str(value) for value in previous.get("factor_names") or []}
    next_factors = {str(value) for value in proposal.get("factor_names") or []}
    prev_strategies = {str(value) for value in previous.get("strategy_ids") or []}
    next_strategies = {str(value) for value in proposal.get("strategy_ids") or []}
    factor_changed = prev_factors != next_factors
    strategy_changed = prev_strategies != next_strategies
    if factor_changed and strategy_changed:
        raise ValueError("follow-up proposal cannot change factors and strategies together")
    added = next_factors - prev_factors
    removed = prev_factors - next_factors
    if len(added) > 1 or len(removed) > 1:
        raise ValueError("follow-up proposal may add or remove at most one factor")


async def generate_catalog_proposal(
    *,
    goal: str,
    factors: Sequence[Mapping[str, Any]],
    strategies: Sequence[Mapping[str, Any]],
    asset_type: str,
    budget_profile: str,
    history: Sequence[Mapping[str, Any]] = (),
    timeout: float | None = None,
) -> dict[str, Any]:
    """Ask the configured provider for one proposal constrained to runtime catalogs."""
    from app.services.ai_provider import (
        current_ai_model,
        current_ai_provider,
        generate_ai_text,
    )

    messages = build_plan_messages(
        goal,
        factors,
        strategies,
        asset_type=asset_type,
        budget_profile=budget_profile,
    )
    messages[0]["content"] += (
        "\n本自动研究会话会反复读取 outer fold 结果; 这些指标只能称为自适应验证, "
        "不是独立样本外结论或最终 holdout。"
        "compact evidence 可能含 factor_names、benchmark_sharpe、max_abs_correlation、"
        "max_corr_pair、regime_sharpe。它们仍只描述自适应验证。"
    )
    if history:
        compact_history = list(history)[-10:]
        messages[0]["content"] += (
            "\n这是自动研究的后续轮次。必须选择与历史执行组合不同的因子/策略集合。"
            "rationale 必须回应上一轮 compact evidence 的 benchmark_sharpe、max_corr_pair、"
            "regime_sharpe。缺失则写明缺失。"
            "本轮只改一个轴: 只改因子或只改对照策略。改因子时最多加或删 1 个, 可以换成另一个。"
            "历史证据只用于提出新假设, 不得改写或补造指标。"
        )
        messages[1]["content"] += (
            "\n\n历史试验(JSON):\n"
            + json.dumps(compact_history, ensure_ascii=False, allow_nan=False)
        )
    text = await generate_ai_text(
        messages,
        temperature=0.1,
        max_tokens=None,
        timeout=timeout,
    )
    proposal = parse_catalog_proposal(
        text,
        factor_ids={str(item["id"]) for item in factors},
        strategy_ids={str(item["id"]) for item in strategies},
    )
    proposal["ai_provider"] = current_ai_provider()
    proposal["ai_model"] = current_ai_model()
    return proposal


def catalog_strategies(strategy_engine: Any, asset_type: str) -> list[dict[str, Any]]:
    return [
        item
        for item in strategy_engine.list_strategies()
        if item.get("execution_backend") == "matrix_native"
        and "1d" in item.get("timeframes", ["1d"])
        and asset_type in item.get("asset_types", ["stock"])
    ][:8]


class ResearchSessionStore:
    """Keep compact sessions separate from leaf mining run persistence."""

    def __init__(self, data_dir: Path | str) -> None:
        self.sessions_root = (
            Path(data_dir).resolve() / "research" / "autoresearch" / "sessions"
        ).resolve()
        self.sessions_root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        spec: Mapping[str, Any],
        initial_proposal: Mapping[str, Any],
        *,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        safe_id = self._validate_session_id(session_id or uuid.uuid4().hex)
        clean_spec = canonicalize_request(spec)
        reserved = {
            "schema_version",
            "session_id",
            "status",
            "initial_proposal",
            "trials",
            "leaderboard",
            "proposal_digests",
        } & set(clean_spec)
        if reserved:
            raise ResearchSessionValidationError(
                f"session spec contains reserved fields: {sorted(reserved)}"
            )
        proposal = canonicalize_request(initial_proposal)
        if proposal.get("digest") != catalog_proposal_digest(proposal):
            raise ResearchSessionValidationError("initial proposal digest is invalid")
        now = _now_iso()
        session = {
            "schema_version": _SCHEMA_VERSION,
            "session_id": safe_id,
            "status": "awaiting_approval",
            **clean_spec,
            "initial_proposal": proposal,
            "current_trial": None,
            "active_run_id": None,
            "completed_trials": 0,
            "no_improvement_trials": 0,
            "best_score": None,
            "data_snapshot_digest": None,
            "trials": [],
            "leaderboard": [],
            "proposal_digests": [],
            "stop_reason": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "approved_at": None,
            "finished_at": None,
        }
        directory = self._session_dir(safe_id)
        with _STORE_LOCK:
            if directory.exists():
                raise ResearchSessionValidationError(f"session already exists: {safe_id}")
            directory.mkdir(parents=False)
            _atomic_write_text(directory / "events.jsonl", "")
            _atomic_write_json(directory / "session.json", session)
        return session

    def get(self, session_id: str) -> dict[str, Any] | None:
        safe_id = self._validate_session_id(session_id)
        with _STORE_LOCK:
            path = self._session_dir(safe_id) / "session.json"
            if not path.is_file():
                return None
            value = _read_json(path)
            return self._validate_session(value, safe_id)

    def list(
        self,
        *,
        limit: int = 50,
        statuses: Collection[str] | None = None,
    ) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
            raise ResearchSessionValidationError("limit must be between 1 and 200")
        allowed = None if statuses is None else set(statuses)
        if allowed is not None and not allowed <= SESSION_STATUSES:
            raise ResearchSessionValidationError("statuses contains an unsupported session status")
        sessions = []
        for path in self.sessions_root.glob("*/session.json"):
            if not _SESSION_ID_PATTERN.fullmatch(path.parent.name):
                continue
            try:
                session = self.get(path.parent.name)
            except ResearchSessionError:
                continue
            if session is not None and (allowed is None or session["status"] in allowed):
                sessions.append(session)
        sessions.sort(key=lambda item: (str(item.get("created_at") or ""), item["session_id"]), reverse=True)
        return sessions[:limit]

    def transition(
        self,
        session_id: str,
        status: ResearchSessionStatus,
        *,
        updates: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if status not in SESSION_STATUSES:
            raise ResearchSessionValidationError(f"unsupported session status: {status}")
        safe_id = self._validate_session_id(session_id)
        with _STORE_LOCK:
            session = self._required(safe_id)
            previous = str(session["status"])
            if status == previous:
                return session
            if status not in _TRANSITIONS[previous]:
                raise InvalidResearchSessionTransitionError(
                    f"cannot transition {previous} to {status}"
                )
            now = _now_iso()
            session["status"] = status
            session["updated_at"] = now
            if status == "running" and session.get("started_at") is None:
                session["started_at"] = now
            if status in TERMINAL_SESSION_STATUSES:
                session["finished_at"] = now
                session["active_run_id"] = None
            if updates:
                self._apply_updates(session, updates)
            self._write_session(safe_id, session)
            return session

    def update(self, session_id: str, updates: Mapping[str, Any]) -> dict[str, Any]:
        safe_id = self._validate_session_id(session_id)
        with _STORE_LOCK:
            session = self._required(safe_id)
            self._apply_updates(session, updates)
            session["updated_at"] = _now_iso()
            self._write_session(safe_id, session)
            return session

    def append_event(
        self,
        session_id: str,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        safe_id = self._validate_session_id(session_id)
        clean_type = str(event_type).strip()
        if not clean_type or len(clean_type) > 64:
            raise ResearchSessionValidationError("event type must contain 1 to 64 characters")
        clean_payload = canonicalize_request(payload or {})
        encoded = json.dumps(clean_payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(encoded) > MAX_SESSION_EVENT_BYTES:
            raise ResearchSessionValidationError("session event payload exceeds its size limit")
        with _STORE_LOCK:
            self._required(safe_id)
            path = self._session_dir(safe_id) / "events.jsonl"
            events = self._read_events(path)
            event = {
                "id": max((int(item["id"]) for item in events), default=0) + 1,
                "timestamp": _now_iso(),
                "type": clean_type,
                "payload": clean_payload,
            }
            events.append(event)
            events = events[-MAX_SESSION_EVENTS:]
            _atomic_write_text(
                path,
                "".join(
                    json.dumps(item, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                    + "\n"
                    for item in events
                ),
            )
            return event

    def read_events(self, session_id: str, *, after_id: int = 0) -> list[dict[str, Any]]:
        if isinstance(after_id, bool) or not isinstance(after_id, int) or after_id < 0:
            raise ResearchSessionValidationError("after_id must be a non-negative integer")
        safe_id = self._validate_session_id(session_id)
        with _STORE_LOCK:
            self._required(safe_id)
            events = self._read_events(self._session_dir(safe_id) / "events.jsonl")
        return [item for item in events if int(item["id"]) > after_id]

    def recover_interrupted(self) -> int:
        recovered = 0
        for session in self.list(limit=200, statuses=ACTIVE_SESSION_STATUSES):
            session_id = str(session["session_id"])
            self.transition(
                session_id,
                "interrupted",
                updates={"stop_reason": "application_restarted"},
            )
            self.append_event(
                session_id,
                "interrupted",
                {"status": "interrupted", "reason": "application_restarted"},
            )
            recovered += 1
        return recovered

    def _required(self, session_id: str) -> dict[str, Any]:
        path = self._session_dir(session_id) / "session.json"
        if not path.is_file():
            raise KeyError(session_id)
        return self._validate_session(_read_json(path), session_id)

    @staticmethod
    def _apply_updates(session: dict[str, Any], updates: Mapping[str, Any]) -> None:
        clean = canonicalize_request(updates)
        forbidden = {"schema_version", "session_id", "status", "created_at"} & set(clean)
        if forbidden:
            raise ResearchSessionValidationError(f"session updates contain protected fields: {sorted(forbidden)}")
        unknown = set(clean) - set(session)
        if unknown:
            raise ResearchSessionValidationError(f"session updates contain unknown fields: {sorted(unknown)}")
        session.update(clean)

    def _write_session(self, session_id: str, session: Mapping[str, Any]) -> None:
        clean = self._validate_session(canonicalize_request(session), session_id)
        _atomic_write_json(self._session_dir(session_id) / "session.json", clean)

    @staticmethod
    def _validate_session(value: Any, session_id: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ResearchSessionError(f"invalid research session: {session_id}")
        if value.get("session_id") != session_id or value.get("status") not in SESSION_STATUSES:
            raise ResearchSessionError(f"invalid research session: {session_id}")
        if not isinstance(value.get("trials"), list) or not isinstance(value.get("leaderboard"), list):
            raise ResearchSessionError(f"invalid research session collections: {session_id}")
        return cast(dict[str, Any], value)

    def _session_dir(self, session_id: str) -> Path:
        path = (self.sessions_root / session_id).resolve()
        if path.parent != self.sessions_root:
            raise ResearchSessionValidationError("session path escapes sessions root")
        return path

    @staticmethod
    def _validate_session_id(session_id: str) -> str:
        if not isinstance(session_id, str) or not _SESSION_ID_PATTERN.fullmatch(session_id):
            raise ResearchSessionValidationError("invalid research session id")
        return session_id

    @staticmethod
    def _read_events(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        events: list[dict[str, Any]] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                value = json.loads(line)
                if isinstance(value, dict) and isinstance(value.get("id"), int):
                    events.append(value)
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchSessionError("failed to read research session events") from exc
        return events


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchSessionError(f"failed to read {path.name}") from exc


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")),
    )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise ResearchSessionError(f"failed to write {path.name}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()
