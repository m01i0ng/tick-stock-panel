"""Serial catalog autoresearch orchestration over persisted mining runs."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import threading
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from app.backtest.factor import FACTOR_COLUMNS
from app.backtest.mining import evaluate_candidate_gate, evaluate_holdout_gate
from app.services.mining_jobs import (
    SUCCESS_RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    MiningRunStore,
    canonicalize_request,
)
from app.services.mining_preflight import (
    MiningPreflightError,
    require_mining_availability,
    reserve_final_holdout,
)
from app.services.mining_schedule import build_data_fingerprint
from app.services.research_sessions import (
    TERMINAL_SESSION_STATUSES,
    InvalidResearchSessionTransitionError,
    ResearchSessionConflictError,
    ResearchSessionStore,
    ResearchSessionValidationError,
    catalog_proposal_digest,
    catalog_strategies,
    generate_catalog_proposal,
    validate_followup_step,
)

ProposalRunner = Callable[[Mapping[str, Any], list[dict[str, Any]]], dict[str, Any]]
FingerprintBuilder = Callable[[dict[str, Any]], Any]
EvidenceLoader = Callable[[MiningRunStore, str], dict[str, Any]]
HoldoutReserver = Callable[..., dict[str, Any]]
HoldoutRunner = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]

_POLL_SECONDS = 0.1
_SHUTDOWN_JOIN_SECONDS = 5.0
_MAX_CANDIDATE_ARTIFACT_BYTES = 8 * 1024 * 1024


class ResearchSessionManager:
    """Run one mining trial at a time and retain a compact, auditable lineage."""

    def __init__(
        self,
        data_dir: Path | str,
        mining_manager: Any,
        repo: Any,
        app_state: Any,
        *,
        proposal_runner: ProposalRunner | None = None,
        fingerprint_builder: FingerprintBuilder | None = None,
        evidence_loader: EvidenceLoader | None = None,
        holdout_reserver: HoldoutReserver | None = None,
        holdout_runner: HoldoutRunner | None = None,
        poll_seconds: float = _POLL_SECONDS,
    ) -> None:
        self.store = ResearchSessionStore(data_dir)
        self._mining_manager = mining_manager
        self._repo = repo
        self._app_state = app_state
        self._proposal_runner = proposal_runner
        self._fingerprint_builder = fingerprint_builder or (
            lambda request: build_data_fingerprint(self._repo, self._app_state, request)
        )
        self._evidence_loader = evidence_loader or load_compact_mining_evidence
        self._holdout_reserver = holdout_reserver or self._default_holdout_reserver
        self._holdout_runner = holdout_runner
        self._poll_seconds = poll_seconds
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._seal_threads: dict[str, threading.Thread] = {}
        self._stop_events: dict[str, threading.Event] = {}
        self._creation_reserved = False
        self._shutdown = False

    def reserve_creation(self) -> None:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("research session manager is shut down")
            if self._creation_reserved or self.active_session() is not None:
                raise ResearchSessionConflictError("another autoresearch session is active")
            self._creation_reserved = True

    def release_creation(self) -> None:
        with self._lock:
            self._creation_reserved = False

    def create_session(
        self,
        spec: Mapping[str, Any],
        initial_proposal: Mapping[str, Any],
        *,
        session_id: str | None = None,
        holdout: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._validate_spec(spec)
        reservation = dict(holdout) if holdout is not None else self._reserve_holdout(spec)
        session_spec = {
            **dict(spec),
            "adaptive_end": reservation["adaptive_end"],
            "holdout": _holdout_record(reservation),
        }
        with self._lock:
            if self._shutdown:
                raise RuntimeError("research session manager is shut down")
            active = self.store.list(
                limit=1,
                statuses={"awaiting_approval", "running", "paused", "stopping"},
            )
            if active:
                raise ResearchSessionConflictError(
                    f"another autoresearch session is active: {active[0]['session_id']}"
                )
            session = self.store.create(session_spec, initial_proposal, session_id=session_id)
            self.store.append_event(
                session["session_id"],
                "created",
                {"status": "awaiting_approval"},
            )
            return session

    def active_session(self) -> dict[str, Any] | None:
        active = self.store.list(
            limit=1,
            statuses={"awaiting_approval", "running", "paused", "stopping"},
        )
        return active[0] if active else None

    def approve(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._required(session_id)
            if session["status"] != "awaiting_approval":
                raise InvalidResearchSessionTransitionError(
                    f"cannot approve session in status {session['status']}"
                )
            self._validate_proposal(session, session["initial_proposal"])
            self._require_readiness(session)
            now = datetime.now(UTC).isoformat()
            session = self.store.transition(
                session_id,
                "running",
                updates={"approved_at": now},
            )
            self.store.append_event(session_id, "approved", {"status": "running"})
            self._start_thread_locked(session_id)
            return session

    def pause(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._required(session_id)
            if session["status"] != "running":
                raise InvalidResearchSessionTransitionError(
                    f"cannot pause session in status {session['status']}"
                )
            session = self.store.transition(session_id, "paused")
            self.store.append_event(session_id, "paused", {"status": "paused"})
            return session

    def resume(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._required(session_id)
            if session["status"] != "paused":
                raise InvalidResearchSessionTransitionError(
                    f"cannot resume session in status {session['status']}"
                )
            session = self.store.transition(session_id, "running")
            self.store.append_event(session_id, "resumed", {"status": "running"})
            self._start_thread_locked(session_id)
            return session

    def stop(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._required(session_id)
            status = str(session["status"])
            if status in TERMINAL_SESSION_STATUSES:
                return session
            if status == "awaiting_approval":
                stopped = self.store.transition(
                    session_id,
                    "stopped",
                    updates={"stop_reason": "user_requested"},
                )
                self.store.append_event(session_id, "stopped", {"status": "stopped"})
                return stopped
            if status != "stopping":
                session = self.store.transition(
                    session_id,
                    "stopping",
                    updates={"stop_reason": "user_requested"},
                )
                self.store.append_event(session_id, "stopping", {"status": "stopping"})
            stop_event = self._stop_events.get(session_id)
            if stop_event is not None:
                stop_event.set()
            active_run_id = session.get("active_run_id")
            if (
                isinstance(active_run_id, str)
                and active_run_id
                and session_id not in self._threads
            ):
                self._mining_manager.cancel(active_run_id)
            if session_id not in self._threads:
                return self._finish_stopped(session_id)
            return session

    def seal(self, session_id: str, trial_index: int | None = None) -> dict[str, Any]:
        with self._lock:
            if self._shutdown:
                raise RuntimeError("research session manager is shut down")
            session = self._required(session_id)
            if session["status"] not in TERMINAL_SESSION_STATUSES:
                raise InvalidResearchSessionTransitionError(
                    "session must be finished before sealing the final holdout"
                )
            holdout = dict(session.get("holdout") or {})
            status = str(holdout.get("status") or "")
            if status == "sealed":
                raise ResearchSessionConflictError("final holdout is already sealed")
            if status == "sealing":
                raise ResearchSessionConflictError("final holdout is already being sealed")
            if status not in {"reserved", "failed"}:
                raise ResearchSessionValidationError("final holdout is not reserved")
            trial = self._trial_for_seal(session, trial_index)
            run_id = str(trial["mining_run_id"])
            definition, signature, confidence = self._frozen_candidate(run_id)
            holdout.update({
                "status": "sealing",
                "mining_run_id": run_id,
                "signature": signature,
                "error": None,
            })
            session = self.store.update(session_id, {"holdout": holdout})
            self.store.append_event(
                session_id,
                "holdout_sealing",
                {"trial_index": trial["index"], "mining_run_id": run_id},
            )
            self._start_seal_thread_locked(session_id, definition, signature, confidence)
            return session

    def recover_interrupted(self) -> int:
        return self.store.recover_interrupted() + self._fail_orphaned_seals()

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
            session_ids = list(self._threads)
        for session_id in session_ids:
            try:
                self.stop(session_id)
            except (KeyError, ResearchSessionValidationError):
                continue
        deadline = time.monotonic() + _SHUTDOWN_JOIN_SECONDS
        current = threading.current_thread()
        with self._lock:
            threads = [thread for thread in self._threads.values() if thread is not current]
            threads.extend(thread for thread in self._seal_threads.values() if thread is not current)
        for thread in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(timeout=remaining)

    def _start_thread_locked(self, session_id: str) -> None:
        thread = self._threads.get(session_id)
        if thread is not None and thread.is_alive():
            return
        stop_event = self._stop_events.get(session_id) or threading.Event()
        thread = threading.Thread(
            target=self._run_session,
            args=(session_id, stop_event),
            name=f"autoresearch-{session_id}",
            daemon=True,
        )
        self._stop_events[session_id] = stop_event
        self._threads[session_id] = thread
        thread.start()

    def _start_seal_thread_locked(
        self,
        session_id: str,
        definition: Mapping[str, Any],
        signature: str,
        confidence: str | None,
    ) -> None:
        thread = self._seal_threads.get(session_id)
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(
            target=self._run_seal,
            args=(session_id, definition, signature, confidence),
            name=f"autoresearch-seal-{session_id}",
            daemon=True,
        )
        self._seal_threads[session_id] = thread
        thread.start()

    def _run_seal(
        self,
        session_id: str,
        definition: Mapping[str, Any],
        signature: str,
        confidence: str | None,
    ) -> None:
        try:
            self._finish_seal(session_id, definition, signature, confidence)
        finally:
            with self._lock:
                self._seal_threads.pop(session_id, None)

    def _fail_orphaned_seals(self) -> int:
        recovered = 0
        # ponytail: latest 200 sessions; raise list limit if orphaned seals exceed that
        for session in self.store.list(limit=200):
            holdout = session.get("holdout") or {}
            if holdout.get("status") != "sealing":
                continue
            session_id = str(session["session_id"])
            with self._lock:
                thread = self._seal_threads.get(session_id)
                if thread is not None and thread.is_alive():
                    continue
            holdout = dict(holdout)
            holdout["status"] = "failed"
            holdout["error"] = "application_restarted"
            self.store.update(session_id, {"holdout": holdout})
            self.store.append_event(
                session_id,
                "holdout_failed",
                {"message": "application_restarted"},
            )
            recovered += 1
        return recovered

    def _run_session(self, session_id: str, stop_event: threading.Event) -> None:
        try:
            while True:
                session = self._required(session_id)
                if stop_event.is_set() or session["status"] == "stopping":
                    self._finish_stopped(session_id)
                    return
                if session["status"] == "paused":
                    stop_event.wait(self._poll_seconds)
                    continue
                if session["status"] != "running":
                    return
                stop_reason = self._budget_stop_reason(session)
                if stop_reason is not None:
                    self._finish_completed(session_id, stop_reason)
                    return

                try:
                    proposal = self._proposal_for_next_trial(session, stop_event)
                except _ResearchWallTimeExceededError:
                    self._finish_completed(session_id, "max_wall_minutes")
                    return
                digest = str(proposal.get("digest") or "")
                if not digest or digest != catalog_proposal_digest(proposal):
                    raise ValueError("catalog proposal digest is invalid")
                if digest in set(session.get("proposal_digests") or []):
                    self.store.append_event(
                        session_id,
                        "proposal_rejected",
                        {"reason": "duplicate_proposal", "proposal_digest": digest},
                    )
                    self._finish_completed(session_id, "duplicate_proposal")
                    return
                self._validate_proposal(session, proposal)
                previous_trials = session.get("trials") or []
                if previous_trials:
                    try:
                        validate_followup_step(previous_trials[-1]["proposal"], proposal)
                    except ValueError as exc:
                        self.store.append_event(
                            session_id,
                            "proposal_rejected",
                            {
                                "reason": "followup_step",
                                "message": str(exc)[:500],
                                "proposal_digest": digest,
                            },
                        )
                        stop_event.wait(self._poll_seconds)
                        continue
                trial_index = len(session["trials"]) + 1
                request = self._mining_request(session, proposal)
                self._require_readiness(session)
                base_fingerprint = self._fingerprint_builder(request)
                if not isinstance(base_fingerprint, Mapping):
                    raise ValueError("research data fingerprint must be an object")
                fingerprint = {
                    **base_fingerprint,
                    "source": "autoresearch",
                    "source_session_id": session_id,
                    "final_holdout_sealed": False,
                }
                snapshot_digest = _research_snapshot_digest(fingerprint)
                current = self._required(session_id)
                frozen_snapshot = current.get("data_snapshot_digest")
                if frozen_snapshot is None:
                    self.store.update(
                        session_id,
                        {"data_snapshot_digest": snapshot_digest},
                    )
                elif frozen_snapshot != snapshot_digest:
                    self.store.append_event(
                        session_id,
                        "data_changed",
                        {"reason": "research_data_snapshot_changed"},
                    )
                    raise RuntimeError("research data snapshot changed during session")
                manifest = None
                while manifest is None:
                    with self._lock:
                        current = self._required(session_id)
                        if stop_event.is_set() or current["status"] == "stopping":
                            self._finish_stopped(session_id)
                            return
                        if self._wall_time_expired(current):
                            self._finish_completed(session_id, "max_wall_minutes")
                            return
                        if current["status"] == "running":
                            manifest = self._mining_manager.start(
                                request,
                                fingerprint,
                                # Each session owns its leaf lifecycle. Reusing an active manual
                                # run would let stop/timeout cancel work owned by another caller.
                                force=True,
                                source=f"autoresearch:{session_id}",
                            )
                    if manifest is None:
                        stop_event.wait(self._poll_seconds)
                run_id = str(manifest["run_id"])
                trial = {
                    "index": trial_index,
                    "proposal": proposal,
                    "mining_run_id": run_id,
                    "status": "running",
                    "started_at": datetime.now(UTC).isoformat(),
                    "finished_at": None,
                    "evidence": None,
                    "improved": None,
                    "error": None,
                }
                trials = [*session["trials"], trial]
                digests = [*session.get("proposal_digests", []), digest]
                self.store.update(
                    session_id,
                    {
                        "trials": trials,
                        "proposal_digests": digests,
                        "current_trial": trial_index,
                        "active_run_id": run_id,
                    },
                )
                self.store.append_event(
                    session_id,
                    "trial_started",
                    {"status": "running", "trial_index": trial_index, "mining_run_id": run_id},
                )
                terminal, budget_reason = self._wait_for_mining(
                    session_id,
                    run_id,
                    stop_event,
                )
                if terminal["status"] in SUCCESS_RUN_STATUSES:
                    evidence = self._evidence_loader(self._mining_manager.store, run_id)
                    self._finish_trial_success(
                        session_id,
                        trial_index,
                        run_id,
                        proposal,
                        evidence,
                    )
                    if stop_event.is_set() or self._required(session_id)["status"] == "stopping":
                        self._finish_stopped(session_id)
                        return
                    if budget_reason is not None:
                        self._finish_completed(session_id, budget_reason)
                        return
                    continue
                if stop_event.is_set() or self._required(session_id)["status"] == "stopping":
                    self._finish_trial_stopped(session_id, trial_index)
                    self._finish_stopped(session_id)
                    return
                if budget_reason is not None:
                    self._finish_trial_stopped(session_id, trial_index, reason=budget_reason)
                    self._finish_completed(session_id, budget_reason)
                    return
                if terminal["status"] not in SUCCESS_RUN_STATUSES:
                    message = str(terminal.get("error") or f"mining run ended as {terminal['status']}")
                    self._finish_trial_failed(session_id, trial_index, message)
                    raise RuntimeError(message)
        except Exception as exc:
            self._finish_failed(session_id, exc)
        finally:
            with self._lock:
                self._threads.pop(session_id, None)
                self._stop_events.pop(session_id, None)

    def _proposal_for_next_trial(
        self,
        session: Mapping[str, Any],
        stop_event: threading.Event,
    ) -> dict[str, Any]:
        if not session["trials"]:
            return dict(session["initial_proposal"])
        history = [
            {
                "proposal": trial.get("proposal"),
                "evidence": trial.get("evidence"),
                "improved": trial.get("improved"),
            }
            for trial in session["trials"]
        ]
        proposal = (
            self._proposal_runner(session, history)
            if self._proposal_runner is not None
            else self._generate_next_proposal(session, history, stop_event)
        )
        if not isinstance(proposal, dict):
            raise TypeError("proposal runner must return a proposal object")
        return proposal

    def _generate_next_proposal(
        self,
        session: Mapping[str, Any],
        history: list[dict[str, Any]],
        stop_event: threading.Event,
    ) -> dict[str, Any]:
        strategies = catalog_strategies(self._app_state.strategy_engine, str(session["asset_type"]))
        async def generate() -> dict[str, Any]:
            remaining = self._remaining_wall_seconds(session)
            if remaining <= 0:
                raise _ResearchWallTimeExceededError
            task = asyncio.create_task(generate_catalog_proposal(
                goal=str(session["goal"]),
                factors=FACTOR_COLUMNS,
                strategies=strategies,
                asset_type=str(session["asset_type"]),
                budget_profile=str(session["budget_profile"]),
                history=history,
                timeout=remaining,
            ))
            try:
                while True:
                    if stop_event.is_set():
                        raise RuntimeError("autoresearch proposal cancelled")
                    remaining = self._remaining_wall_seconds(session)
                    if remaining <= 0:
                        raise _ResearchWallTimeExceededError
                    done, _ = await asyncio.wait({task}, timeout=min(0.25, remaining))
                    if task in done:
                        return await task
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        return asyncio.run(generate())

    def _wait_for_mining(
        self,
        session_id: str,
        run_id: str,
        stop_event: threading.Event,
    ) -> tuple[dict[str, Any], str | None]:
        cancellation_sent = False
        budget_reason: str | None = None
        while True:
            session = self._required(session_id)
            if budget_reason is None and self._wall_time_expired(session):
                budget_reason = "max_wall_minutes"
                self.store.append_event(
                    session_id,
                    "budget_exhausted",
                    {"reason": budget_reason, "mining_run_id": run_id},
                )
            if (
                not cancellation_sent
                and (
                    stop_event.is_set()
                    or session["status"] == "stopping"
                    or budget_reason is not None
                )
            ):
                self._mining_manager.cancel(run_id)
                cancellation_sent = True
            manifest = self._mining_manager.store.get(run_id)
            if manifest is None:
                raise RuntimeError(f"mining run disappeared: {run_id}")
            if manifest["status"] in TERMINAL_RUN_STATUSES:
                return manifest, budget_reason
            stop_event.wait(self._poll_seconds)

    def _finish_trial_success(
        self,
        session_id: str,
        trial_index: int,
        run_id: str,
        proposal: Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> None:
        session = self._required(session_id)
        trials = list(session["trials"])
        trial = dict(trials[trial_index - 1])
        score = _finite(evidence.get("oos_sharpe"))
        best = _finite(session.get("best_score"))
        improved = score is not None and (best is None or score > best)
        trial.update(
            {
                "status": "completed",
                "finished_at": datetime.now(UTC).isoformat(),
                "evidence": dict(evidence),
                "improved": improved,
            }
        )
        trials[trial_index - 1] = trial
        leaderboard = list(session["leaderboard"])
        leaderboard.append(
            {
                "trial_index": trial_index,
                "mining_run_id": run_id,
                "proposal_digest": proposal["digest"],
                "title": proposal["title"],
                **dict(evidence),
            }
        )
        leaderboard.sort(key=_leaderboard_sort_key)
        no_improvement = 0 if improved else int(session["no_improvement_trials"]) + 1
        self.store.update(
            session_id,
            {
                "trials": trials,
                "leaderboard": leaderboard,
                "completed_trials": int(session["completed_trials"]) + 1,
                "no_improvement_trials": no_improvement,
                "best_score": score if improved else best,
                "active_run_id": None,
            },
        )
        self.store.append_event(
            session_id,
            "trial_completed",
            {
                "status": "completed",
                "trial_index": trial_index,
                "mining_run_id": run_id,
                "improved": improved,
            },
        )

    def _finish_trial_failed(self, session_id: str, trial_index: int, message: str) -> None:
        session = self._required(session_id)
        trials = list(session["trials"])
        trial = dict(trials[trial_index - 1])
        trial.update(
            {
                "status": "failed",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": message[:2000],
            }
        )
        trials[trial_index - 1] = trial
        self.store.update(session_id, {"trials": trials, "active_run_id": None})

    def _finish_trial_stopped(
        self,
        session_id: str,
        trial_index: int,
        *,
        reason: str = "session stopped",
    ) -> None:
        session = self._required(session_id)
        trials = list(session["trials"])
        trial = dict(trials[trial_index - 1])
        trial.update(
            {
                "status": "stopped",
                "finished_at": datetime.now(UTC).isoformat(),
                "error": reason,
            }
        )
        trials[trial_index - 1] = trial
        self.store.update(session_id, {"trials": trials, "active_run_id": None})

    def _budget_stop_reason(self, session: Mapping[str, Any]) -> str | None:
        if len(session["trials"]) >= int(session["max_trials"]):
            return "max_trials"
        if int(session["no_improvement_trials"]) >= int(session["patience"]):
            return "patience"
        if self._wall_time_expired(session):
            return "max_wall_minutes"
        return None

    @staticmethod
    def _wall_time_expired(session: Mapping[str, Any]) -> bool:
        return ResearchSessionManager._remaining_wall_seconds(session) <= 0

    @staticmethod
    def _remaining_wall_seconds(session: Mapping[str, Any]) -> float:
        started_at = session.get("started_at")
        if not isinstance(started_at, str):
            return math.inf
        elapsed = datetime.now(UTC) - datetime.fromisoformat(started_at)
        return int(session["max_wall_minutes"]) * 60 - elapsed.total_seconds()

    def _finish_completed(self, session_id: str, reason: str) -> dict[str, Any]:
        session = self._required(session_id)
        if session["status"] in TERMINAL_SESSION_STATUSES:
            return session
        completed = self.store.transition(
            session_id,
            "completed",
            updates={"stop_reason": reason, "active_run_id": None},
        )
        self.store.append_event(
            session_id,
            "completed",
            {"status": "completed", "reason": reason},
        )
        return completed

    def _finish_stopped(self, session_id: str) -> dict[str, Any]:
        session = self._required(session_id)
        if session["status"] in TERMINAL_SESSION_STATUSES:
            return session
        if session["status"] != "stopping":
            session = self.store.transition(
                session_id,
                "stopping",
                updates={"stop_reason": "user_requested"},
            )
        stopped = self.store.transition(
            session_id,
            "stopped",
            updates={"stop_reason": session.get("stop_reason") or "user_requested"},
        )
        self.store.append_event(session_id, "stopped", {"status": "stopped"})
        return stopped

    def _finish_failed(self, session_id: str, exc: Exception) -> None:
        message = str(exc)[:2000]
        with self._lock:
            try:
                session = self._required(session_id)
            except (KeyError, ResearchSessionValidationError):
                return
            if session["status"] in TERMINAL_SESSION_STATUSES:
                return
            if session["status"] == "stopping":
                self._finish_stopped(session_id)
                return
            self.store.transition(
                session_id,
                "failed",
                updates={"error": message, "active_run_id": None},
            )
            self.store.append_event(
                session_id,
                "error",
                {"status": "failed", "message": message},
            )

    def _validate_proposal(
        self,
        session: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> None:
        factor_ids = {str(item["id"]) for item in FACTOR_COLUMNS}
        factor_names = proposal.get("factor_names")
        if (
            not isinstance(factor_names, list)
            or not 2 <= len(factor_names) <= 12
            or any(not isinstance(value, str) or not value for value in factor_names)
            or len(set(factor_names)) != len(factor_names)
        ):
            raise ValueError("catalog proposal must contain 2 to 12 unique factors")
        unknown_factors = sorted(set(factor_names) - factor_ids)
        if unknown_factors:
            raise ValueError(f"catalog proposal contains unknown factors: {unknown_factors}")
        strategy_ids = {
            str(item["id"])
            for item in catalog_strategies(
                self._app_state.strategy_engine,
                str(session["asset_type"]),
            )
        }
        selected_strategies = proposal.get("strategy_ids")
        if (
            not isinstance(selected_strategies, list)
            or len(selected_strategies) > 3
            or any(not isinstance(value, str) or not value for value in selected_strategies)
            or len(set(selected_strategies)) != len(selected_strategies)
        ):
            raise ValueError("catalog proposal must contain at most 3 unique strategies")
        unknown_strategies = sorted(set(selected_strategies) - strategy_ids)
        if unknown_strategies:
            raise ValueError(f"catalog proposal contains unknown strategies: {unknown_strategies}")

    def _mining_request(
        self,
        session: Mapping[str, Any],
        proposal: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "factor_names": sorted(str(value) for value in proposal["factor_names"]),
            "strategy_ids": sorted(str(value) for value in proposal["strategy_ids"]),
            "symbols": None,
            "asset_type": session["asset_type"],
            "start": session.get("start"),
            "end": session.get("adaptive_end") or session.get("end"),
            "budget_profile": session["budget_profile"],
            "commission_pct": session["commission_pct"],
            "stamp_tax_pct": session["stamp_tax_pct"],
            "slippage_bps": session["slippage_bps"],
            "correlation_threshold": session["correlation_threshold"],
            "max_combination_factors": session["max_combination_factors"],
            "beam_width": session["beam_width"],
            "max_finalists": session["max_finalists"],
        }

    def _require_readiness(self, session: Mapping[str, Any]) -> None:
        require_mining_availability(
            self._repo.store.data_dir,
            asset_type=str(session["asset_type"]),
            budget_profile=str(session["budget_profile"]),
            start=_optional_date(session.get("start")),
            end=_optional_date(session.get("adaptive_end") or session.get("end")),
        )
        current = session.get("holdout") or {}
        fresh = self._reserve_holdout(session)
        if (
            fresh.get("start") != current.get("start")
            or fresh.get("end") != current.get("end")
            or fresh.get("bars") != current.get("bars")
        ):
            raise MiningPreflightError(
                "holdout_changed",
                "最终留出集窗口已变化, 请停止后新建会话",
            )

    def _reserve_holdout(self, spec: Mapping[str, Any]) -> dict[str, Any]:
        return self._holdout_reserver(
            self._repo.store.data_dir,
            asset_type=str(spec["asset_type"]),
            budget_profile=str(spec["budget_profile"]),
            start=_optional_date(spec.get("start")),
            end=_optional_date(spec.get("end")),
        )

    def _default_holdout_reserver(self, data_dir, **kwargs) -> dict[str, Any]:
        return reserve_final_holdout(data_dir, **kwargs)

    def _default_holdout_runner(
        self,
        session: Mapping[str, Any],
        definition: Mapping[str, Any],
    ) -> dict[str, Any]:
        from app.backtest.mining_runtime import evaluate_frozen_candidate

        holdout = session.get("holdout") or {}
        run_id = str(holdout.get("mining_run_id") or "")
        generation = None
        if run_id:
            manifest = self._mining_manager.store.get(run_id)
            fingerprint = (manifest or {}).get("data_fingerprint")
            if isinstance(fingerprint, Mapping):
                generation = fingerprint.get("generation")
        return evaluate_frozen_candidate(
            repo=self._repo,
            strategy_engine=self._app_state.strategy_engine,
            data_dir=self._repo.store.data_dir,
            definition=definition,
            asset_type=str(session["asset_type"]),
            start=date.fromisoformat(str(holdout["start"])),
            end=date.fromisoformat(str(holdout["end"])),
            commission_pct=float(session["commission_pct"]),
            stamp_tax_pct=float(session["stamp_tax_pct"]),
            slippage_bps=float(session["slippage_bps"]),
            expected_generation=str(generation) if generation else None,
        )

    def _trial_for_seal(
        self,
        session: Mapping[str, Any],
        trial_index: int | None,
    ) -> dict[str, Any]:
        if trial_index is None:
            leaderboard = session.get("leaderboard") or []
            if not leaderboard:
                raise ValueError("no successful trial is available to seal")
            trial_index = int(leaderboard[0]["trial_index"])
        trials = [
            trial
            for trial in session.get("trials") or []
            if int(trial.get("index") or 0) == trial_index
        ]
        if not trials:
            raise ValueError("requested trial does not exist")
        trial = trials[0]
        if trial.get("status") != "completed" or not trial.get("mining_run_id"):
            raise ValueError("requested trial has no successful mining run")
        return trial

    def _frozen_candidate(self, run_id: str) -> tuple[dict[str, Any], str, str | None]:
        store = self._mining_manager.store
        manifest = store.get(run_id)
        if manifest is None or manifest.get("status") not in SUCCESS_RUN_STATUSES:
            raise ValueError("holdout source mining run is not successful")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, Mapping) or artifacts.get("candidates") != "candidates.parquet":
            raise ValueError("registered mining candidates artifact is unavailable")
        path = store.artifact_path(run_id, "candidates")
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_CANDIDATE_ARTIFACT_BYTES:
            raise ValueError("registered mining candidates artifact is unavailable")
        frame = pl.read_parquet(path, columns=["kind", "signature", "definition_json", "confidence", "oos_sharpe"])
        rows = [
            row
            for row in frame.to_dicts()
            if row.get("kind") == "factor_combination"
        ]
        if not rows:
            raise ValueError("mining run produced no factor combination to seal")
        rows.sort(key=lambda row: _score_sort_key(row.get("oos_sharpe")))
        best = rows[0]
        raw = best.get("definition_json")
        definition = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(definition, dict):
            raise ValueError("frozen candidate definition is invalid")
        return definition, str(best.get("signature") or ""), best.get("confidence")

    def _finish_seal(
        self,
        session_id: str,
        definition: Mapping[str, Any],
        signature: str,
        confidence: str | None,
    ) -> dict[str, Any]:
        session = self._required(session_id)
        holdout = dict(session.get("holdout") or {})
        runner = self._holdout_runner or self._default_holdout_runner
        try:
            raw = runner(session, definition)
            if not isinstance(raw, Mapping):
                raise TypeError("holdout runner must return a mapping")
            metrics = dict(raw)
            if metrics.get("error"):
                raise RuntimeError(str(metrics["error"]))
            gate = evaluate_holdout_gate(
                confidence=confidence,
                sharpe=_finite(metrics.get("sharpe")),
                max_drawdown=_finite(metrics.get("max_drawdown")),
                n_trades=metrics.get("n_trades"),
            )
            holdout.update({
                "status": "sealed",
                "signature": signature,
                "sealed_at": datetime.now(UTC).isoformat(),
                "sharpe": _finite(metrics.get("sharpe")),
                "max_drawdown": _finite(metrics.get("max_drawdown")),
                "n_trades": (
                    int(metrics["n_trades"])
                    if isinstance(metrics.get("n_trades"), (int, float))
                    and not isinstance(metrics.get("n_trades"), bool)
                    else None
                ),
                "total_return": _finite(metrics.get("total_return")),
                "qualified": gate.qualified,
                "gate_reasons": list(gate.reasons),
                "error": None,
            })
            self._mining_manager.store.patch_fingerprint(
                str(holdout["mining_run_id"]),
                {
                    "final_holdout_sealed": True,
                    "final_holdout": {
                        "signature": signature,
                        "start": holdout.get("start"),
                        "end": holdout.get("end"),
                        "bars": holdout.get("bars"),
                        "sharpe": holdout["sharpe"],
                        "max_drawdown": holdout["max_drawdown"],
                        "n_trades": holdout["n_trades"],
                        "total_return": holdout["total_return"],
                        "qualified": gate.qualified,
                        "gate_reasons": list(gate.reasons),
                    },
                },
            )
            session = self.store.update(session_id, {"holdout": holdout})
            self.store.append_event(
                session_id,
                "holdout_sealed",
                {
                    "qualified": gate.qualified,
                    "signature": signature,
                    "mining_run_id": holdout.get("mining_run_id"),
                },
            )
            return session
        except Exception as exc:
            holdout.update({
                "status": "failed",
                "error": str(exc)[:2000],
            })
            session = self.store.update(session_id, {"holdout": holdout})
            self.store.append_event(
                session_id,
                "holdout_failed",
                {"message": str(exc)[:2000]},
            )
            return session

    @staticmethod
    def _validate_spec(spec: Mapping[str, Any]) -> None:
        max_trials = spec.get("max_trials")
        wall = spec.get("max_wall_minutes")
        patience = spec.get("patience")
        if isinstance(max_trials, bool) or not isinstance(max_trials, int) or not 1 <= max_trials <= 10:
            raise ResearchSessionValidationError("max_trials must be between 1 and 10")
        if isinstance(wall, bool) or not isinstance(wall, int) or not 1 <= wall <= 360:
            raise ResearchSessionValidationError("max_wall_minutes must be between 1 and 360")
        if (
            isinstance(patience, bool)
            or not isinstance(patience, int)
            or not 1 <= patience <= max_trials
        ):
            raise ResearchSessionValidationError("patience must be between 1 and max_trials")

    def _required(self, session_id: str) -> dict[str, Any]:
        session = self.store.get(session_id)
        if session is None:
            raise KeyError(session_id)
        return session


def load_compact_mining_evidence(store: MiningRunStore, run_id: str) -> dict[str, Any]:
    """Read only the registered best candidate evidence needed by the next proposal."""
    manifest = store.get(run_id)
    if manifest is None or manifest.get("status") not in SUCCESS_RUN_STATUSES:
        raise ValueError("compact evidence requires a successful mining run")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get("candidates") != "candidates.parquet":
        raise ValueError("registered mining candidates artifact is unavailable")
    path = store.artifact_path(run_id, "candidates")
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_CANDIDATE_ARTIFACT_BYTES:
        raise ValueError("registered mining candidates artifact is unavailable")
    required = {
        "kind",
        "oos_sharpe",
        "oos_max_drawdown",
        "oos_positive_fold_ratio",
        "oos_n_trades",
        "valid_folds",
        "confidence",
    }
    schema = pl.read_parquet_schema(path)
    if not required.issubset(schema):
        raise ValueError("mining candidates artifact schema is invalid")
    extra = {"factor_names_json", "signature"} & set(schema)
    frame = pl.read_parquet(path, columns=sorted(required | extra))
    summary = store.read_summary(run_id)
    rows = frame.to_dicts()
    factor_rows = [row for row in rows if row.get("kind") == "factor_combination"]
    if not factor_rows:
        return {
            "data_as_of": summary.get("data_as_of"),
            "oos_sharpe": None,
            "oos_max_drawdown": None,
            "oos_positive_fold_ratio": None,
            "oos_n_trades": None,
            "valid_folds": 0,
            "qualified": False,
            "gate_reasons": ["mining run produced no candidates"],
        }
    factor_rows.sort(key=lambda row: _score_sort_key(row.get("oos_sharpe")))
    best = factor_rows[0]
    gate = evaluate_candidate_gate(
        confidence=best.get("confidence"),
        valid_folds=best.get("valid_folds"),
        positive_fold_ratio=_finite(best.get("oos_positive_fold_ratio")),
        sharpe=_finite(best.get("oos_sharpe")),
        max_drawdown=_finite(best.get("oos_max_drawdown")),
        n_trades=best.get("oos_n_trades"),
    )
    factor_names = _json_string_list(best.get("factor_names_json"))
    max_abs_correlation, max_corr_pair = _strongest_selected_correlation(
        store, run_id, artifacts, factor_names
    )
    strategy_rows = [row for row in rows if row.get("kind") == "existing_strategy"]
    strategy_rows.sort(key=lambda row: _score_sort_key(row.get("oos_sharpe")))
    evidence = {
        "data_as_of": summary.get("data_as_of"),
        "oos_sharpe": _finite(best.get("oos_sharpe")),
        "oos_max_drawdown": _finite(best.get("oos_max_drawdown")),
        "oos_positive_fold_ratio": _finite(best.get("oos_positive_fold_ratio")),
        "oos_n_trades": int(best["oos_n_trades"]) if best.get("oos_n_trades") is not None else None,
        "valid_folds": int(best.get("valid_folds") or 0),
        "qualified": gate.qualified,
        "gate_reasons": list(gate.reasons),
        "factor_names": factor_names,
        "benchmark_sharpe": (
            _finite(strategy_rows[0].get("oos_sharpe")) if strategy_rows else None
        ),
        "max_abs_correlation": max_abs_correlation,
        "max_corr_pair": max_corr_pair,
        "regime_sharpe": _regime_sharpe_summary(
            store, run_id, artifacts, best.get("signature")
        ),
    }
    return evidence


def _json_string_list(value: Any) -> list[str] | None:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [str(item) for item in parsed]


def _strongest_selected_correlation(
    store: MiningRunStore,
    run_id: str,
    artifacts: Mapping[str, Any],
    factor_names: list[str] | None,
) -> tuple[float | None, list[str] | None]:
    if not factor_names or len(factor_names) < 2:
        return None, None
    if artifacts.get("correlation") != "correlation.parquet":
        return None, None
    path = store.artifact_path(run_id, "correlation")
    if path.is_symlink() or not path.is_file():
        return None, None
    try:
        frame = pl.read_parquet(path, columns=["factor_x", "factor_y", "rho"])
    except Exception:
        return None, None
    selected = set(factor_names)
    strongest: float | None = None
    pair: list[str] | None = None
    for row in frame.to_dicts():
        left = str(row.get("factor_x") or "")
        right = str(row.get("factor_y") or "")
        if left == right or left not in selected or right not in selected:
            continue
        rho = _finite(row.get("rho"))
        if rho is None:
            continue
        magnitude = abs(rho)
        if strongest is None or magnitude > strongest:
            strongest = magnitude
            pair = sorted((left, right))
    return strongest, pair


def _regime_sharpe_summary(
    store: MiningRunStore,
    run_id: str,
    artifacts: Mapping[str, Any],
    signature: Any,
) -> dict[str, float] | None:
    if not signature or artifacts.get("folds") != "folds.parquet":
        return None
    path = store.artifact_path(run_id, "folds")
    if path.is_symlink() or not path.is_file():
        return None
    try:
        frame = pl.read_parquet(
            path, columns=["candidate_signature", "regime_state", "sharpe", "skipped"]
        )
    except Exception:
        return None
    summary: dict[str, float] = {}
    for state in ("strong", "range", "weak"):
        values = [
            _finite(row.get("sharpe"))
            for row in frame.to_dicts()
            if str(row.get("candidate_signature") or "") == str(signature)
            and row.get("regime_state") == state
            and not row.get("skipped")
        ]
        finite = [value for value in values if value is not None]
        if finite:
            summary[state] = round(sum(finite) / len(finite), 4)
    return summary or None


def _holdout_record(reservation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": reservation.get("status") or "reserved",
        "start": reservation.get("start"),
        "end": reservation.get("end"),
        "bars": reservation.get("bars"),
        "mining_run_id": reservation.get("mining_run_id"),
        "signature": reservation.get("signature"),
        "sealed_at": reservation.get("sealed_at"),
        "sharpe": reservation.get("sharpe"),
        "max_drawdown": reservation.get("max_drawdown"),
        "n_trades": reservation.get("n_trades"),
        "total_return": reservation.get("total_return"),
        "qualified": reservation.get("qualified"),
        "gate_reasons": reservation.get("gate_reasons"),
        "error": reservation.get("error"),
    }


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


class _ResearchWallTimeExceededError(RuntimeError):
    pass


def _research_snapshot_digest(fingerprint: Any) -> str:
    if not isinstance(fingerprint, Mapping):
        raise ValueError("research data fingerprint must be an object")
    payload = {
        key: fingerprint.get(key)
        for key in (
            "version",
            "asset_type",
            "generation",
            "latest_enriched_date",
            "enriched",
            "instruments",
            "regime",
            "algorithm_version",
            "methodology_version",
            "implementation",
        )
    }
    encoded = json.dumps(
        canonicalize_request(payload),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _score_sort_key(value: Any) -> tuple[bool, float]:
    score = _finite(value)
    return score is None, -score if score is not None else 0.0


def _leaderboard_sort_key(item: Mapping[str, Any]) -> tuple[bool, float, int]:
    missing, score = _score_sort_key(item.get("oos_sharpe"))
    return missing, score, int(item["trial_index"])


def _optional_date(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise ValueError("research session dates must use ISO YYYY-MM-DD format")
