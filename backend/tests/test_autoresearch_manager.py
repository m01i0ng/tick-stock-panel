from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

import app.services.autoresearch_manager as manager_module
from app.backtest.mining import compute_candidate_signature
from app.services.autoresearch_manager import (
    ResearchSessionManager,
    load_compact_mining_evidence,
)
from app.services.mining_jobs import MiningRunStore
from app.services.research_sessions import (
    InvalidResearchSessionTransitionError,
    ResearchSessionConflictError,
    ResearchSessionValidationError,
    catalog_proposal_digest,
)


class _StrategyEngine:
    @staticmethod
    def list_strategies():
        return [
            {
                "id": "benchmark",
                "name": "基准",
                "execution_backend": "matrix_native",
                "timeframes": ["1d"],
                "asset_types": ["stock"],
            }
        ]


class _InstantMiningManager:
    def __init__(self, data_dir: Path) -> None:
        self.store = MiningRunStore(data_dir)
        self.requests = []
        self.cancelled = []
        self.forces = []

    def start(self, request, fingerprint, force=False, source="manual"):
        self.forces.append(force)
        self.requests.append((request, fingerprint, source))
        manifest = self.store.create(request, fingerprint)
        run_id = manifest["run_id"]
        self.store.transition_status(run_id, "running")
        self.store.write_summary(run_id, {"candidate_count": 1})
        return self.store.transition_status(run_id, "succeeded")

    def cancel(self, run_id):
        self.cancelled.append(run_id)
        manifest = self.store.get(run_id)
        if manifest and manifest["status"] not in {"cancelled", "succeeded", "failed"}:
            return self.store.transition_status(run_id, "cancelled")
        return manifest


class _BlockingMiningManager(_InstantMiningManager):
    def start(self, request, fingerprint, force=False, source="manual"):
        self.forces.append(force)
        self.requests.append((request, fingerprint, source))
        manifest = self.store.create(request, fingerprint)
        return self.store.transition_status(manifest["run_id"], "running")


def _proposal(factors, strategy_ids=None):
    value = {
        "title": "候选",
        "hypothesis": "可检验假设",
        "rationale": "互补因子",
        "factor_names": factors,
        "strategy_ids": list(strategy_ids or []),
        "expected_outcome": "样本外改善",
        "risks": ["过拟合"],
    }
    value["digest"] = catalog_proposal_digest(value)
    return value


def _fake_holdout(data_dir, **kwargs):
    del data_dir
    end = kwargs.get("end")
    end_s = end.isoformat() if hasattr(end, "isoformat") and not isinstance(end, str) else (end or "2025-01-01")
    return {
        "adaptive_end": "2024-09-30",
        "start": "2024-10-08",
        "end": end_s,
        "bars": 63,
        "status": "reserved",
        "mining_run_id": None,
        "signature": None,
        "sealed_at": None,
        "sharpe": None,
        "max_drawdown": None,
        "n_trades": None,
        "total_return": None,
        "qualified": None,
        "gate_reasons": None,
        "error": None,
    }


def _spec(**updates):
    value = {
        "goal": "自动研究量价组合",
        "asset_type": "stock",
        "start": "2024-01-01",
        "end": "2025-01-01",
        "budget_profile": "exploratory",
        "commission_pct": 0.0002,
        "stamp_tax_pct": 0.0005,
        "slippage_bps": 5.0,
        "correlation_threshold": 0.75,
        "max_combination_factors": 4,
        "beam_width": 12,
        "max_finalists": 8,
        "max_trials": 2,
        "max_wall_minutes": 60,
        "patience": 2,
    }
    value.update(updates)
    return value


def _manager(tmp_path, mining, proposals=(), evidence=None):
    queued = list(proposals)
    app_state = SimpleNamespace(strategy_engine=_StrategyEngine())
    return ResearchSessionManager(
        tmp_path,
        mining,
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        app_state,
        proposal_runner=lambda session, history: queued.pop(0),
        fingerprint_builder=lambda request: {"generation": "test"},
        evidence_loader=evidence or (
            lambda store, run_id: {
                "oos_sharpe": 1.0,
                "oos_max_drawdown": -0.1,
                "oos_positive_fold_ratio": 0.75,
                "oos_n_trades": 80,
                "valid_folds": 3,
                "qualified": True,
                "gate_reasons": [],
            }
        ),
        holdout_reserver=_fake_holdout,
        poll_seconds=0.005,
    )


def _wait(manager, session_id, status, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = manager.store.get(session_id)
        assert session is not None
        if session["status"] == status:
            return session
        time.sleep(0.005)
    pytest.fail(f"session did not reach {status}")


def _wait_holdout(manager, session_id, status, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        session = manager.store.get(session_id)
        assert session is not None
        if (session.get("holdout") or {}).get("status") == status:
            return session
        time.sleep(0.005)
    pytest.fail(f"holdout did not reach {status}")


@pytest.fixture(autouse=True)
def _skip_real_preflight(monkeypatch):
    monkeypatch.setattr(manager_module, "require_mining_availability", lambda *args, **kwargs: None)


def test_runs_trials_serially_until_max_trials(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    manager = _manager(tmp_path, mining, [_proposal(["momentum_10d", "turnover_rate"])])
    created = manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="serial",
    )
    assert created["status"] == "awaiting_approval"
    assert created["adaptive_end"] == "2024-09-30"
    assert created["holdout"]["status"] == "reserved"
    assert created["holdout"]["start"] == "2024-10-08"

    manager.approve("serial")
    completed = _wait(manager, "serial", "completed")
    assert completed["stop_reason"] == "max_trials"
    assert completed["completed_trials"] == 2
    assert [trial["index"] for trial in completed["trials"]] == [1, 2]
    assert all(trial["status"] == "completed" for trial in completed["trials"])
    assert len(mining.requests) == 2
    assert mining.forces == [True, True]
    assert all(
        fingerprint["source"] == "autoresearch"
        and fingerprint["source_session_id"] == "serial"
        and fingerprint["final_holdout_sealed"] is False
        for _, fingerprint, _ in mining.requests
    )
    assert all(request["end"] == "2024-09-30" for request, _, _ in mining.requests)
    assert all(source == "autoresearch:serial" for _, _, source in mining.requests)


def test_followup_step_rejects_multi_axis_then_accepts_one_factor_swap(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    accepted = _proposal(["momentum_10d", "turnover_rate"])
    manager = _manager(
        tmp_path,
        mining,
        [
            _proposal(["momentum_10d", "turnover_rate"], ["benchmark"]),
            accepted,
        ],
    )
    manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="followup",
    )
    manager.approve("followup")
    completed = _wait(manager, "followup", "completed")
    assert completed["stop_reason"] == "max_trials"
    assert completed["completed_trials"] == 2
    assert completed["trials"][1]["proposal"]["factor_names"] == ["momentum_10d", "turnover_rate"]
    assert completed["trials"][1]["proposal"]["strategy_ids"] == []
    rejected = [
        event
        for event in manager.store.read_events("followup")
        if event["type"] == "proposal_rejected"
    ]
    assert rejected[0]["payload"]["reason"] == "followup_step"
    assert len(mining.requests) == 2


def test_duplicate_proposal_stops_without_duplicate_mining(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    initial = _proposal(["momentum_5d", "turnover_rate"])
    manager = _manager(tmp_path, mining, [dict(initial)])
    manager.create_session(_spec(max_trials=3), initial, session_id="duplicate")
    manager.approve("duplicate")

    completed = _wait(manager, "duplicate", "completed")
    assert completed["stop_reason"] == "duplicate_proposal"
    assert len(completed["trials"]) == 1
    assert len(mining.requests) == 1
    rejected = [
        event
        for event in manager.store.read_events("duplicate")
        if event["type"] == "proposal_rejected"
    ]
    assert rejected[0]["payload"] == {
        "reason": "duplicate_proposal",
        "proposal_digest": initial["digest"],
    }


def test_patience_and_invalid_budget_are_enforced(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    scores = iter([1.0, 0.5])
    manager = _manager(
        tmp_path,
        mining,
        [_proposal(["momentum_10d", "turnover_rate"])],
        evidence=lambda store, run_id: {
            "oos_sharpe": next(scores),
            "oos_max_drawdown": -0.1,
            "oos_positive_fold_ratio": 0.75,
            "oos_n_trades": 80,
            "valid_folds": 3,
            "qualified": True,
            "gate_reasons": [],
        },
    )
    with pytest.raises(ResearchSessionValidationError, match="patience"):
        manager.create_session(_spec(max_trials=1, patience=2), _proposal(["rsi_6", "turnover_rate"]))

    manager.create_session(
        _spec(max_trials=3, patience=1),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="patience",
    )
    manager.approve("patience")
    completed = _wait(manager, "patience", "completed")
    assert completed["stop_reason"] == "patience"
    assert completed["completed_trials"] == 2


def test_stop_propagates_to_active_mining_run(tmp_path: Path) -> None:
    mining = _BlockingMiningManager(tmp_path)
    manager = _manager(tmp_path, mining)
    manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="cancel",
    )
    manager.approve("cancel")
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        session = manager.store.get("cancel")
        if session and session["active_run_id"]:
            break
        time.sleep(0.005)
    else:
        pytest.fail("mining trial did not start")

    manager.stop("cancel")
    stopped = _wait(manager, "cancel", "stopped")
    assert mining.cancelled == [stopped["trials"][0]["mining_run_id"]]
    assert stopped["trials"][0]["status"] == "stopped"


def test_stop_after_leaf_success_keeps_completed_evidence(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    holder: dict[str, ResearchSessionManager] = {}

    def evidence(store, run_id):
        holder["manager"].stop("stop_after_success")
        return {
            "oos_sharpe": 0.8,
            "oos_max_drawdown": -0.1,
            "oos_positive_fold_ratio": 0.75,
            "oos_n_trades": 80,
            "valid_folds": 3,
            "qualified": True,
            "gate_reasons": [],
        }

    manager = _manager(tmp_path, mining, evidence=evidence)
    holder["manager"] = manager
    manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="stop_after_success",
    )
    manager.approve("stop_after_success")

    stopped = _wait(manager, "stop_after_success", "stopped")
    assert stopped["completed_trials"] == 1
    assert stopped["trials"][0]["status"] == "completed"
    assert stopped["trials"][0]["evidence"]["oos_sharpe"] == 0.8


def test_pause_resume_and_illegal_approval(tmp_path: Path) -> None:
    mining = _BlockingMiningManager(tmp_path)
    manager = _manager(tmp_path, mining)
    manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="pause",
    )
    manager.approve("pause")
    paused = manager.pause("pause")
    assert paused["status"] == "paused"
    resumed = manager.resume("pause")
    assert resumed["status"] == "running"
    with pytest.raises(InvalidResearchSessionTransitionError):
        manager.approve("pause")
    manager.stop("pause")
    _wait(manager, "pause", "stopped")


def test_rejects_second_non_terminal_session(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    manager = _manager(tmp_path, mining)
    manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="first_active",
    )
    with pytest.raises(ResearchSessionConflictError, match="first_active"):
        manager.create_session(
            _spec(),
            _proposal(["momentum_10d", "turnover_rate"]),
            session_id="second_active",
        )


def test_creation_reservation_blocks_duplicate_ai_planning(tmp_path: Path) -> None:
    manager = _manager(tmp_path, _InstantMiningManager(tmp_path))

    manager.reserve_creation()
    with pytest.raises(ResearchSessionConflictError):
        manager.reserve_creation()
    manager.release_creation()
    manager.reserve_creation()
    manager.release_creation()


def test_approve_rechecks_readiness_before_transitioning_or_starting_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mining = _InstantMiningManager(tmp_path)
    manager = _manager(tmp_path, mining)
    manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="preflight",
    )

    def fail_preflight(*_args, **_kwargs):
        raise ValueError("市场环境数据不可用")

    monkeypatch.setattr(
        manager_module,
        "require_mining_availability",
        fail_preflight,
    )

    with pytest.raises(ValueError, match="市场环境"):
        manager.approve("preflight")

    assert manager.store.get("preflight")["status"] == "awaiting_approval"
    assert mining.requests == []


def test_wall_budget_cancels_active_leaf_and_completes_deterministically(
    tmp_path: Path,
) -> None:
    mining = _BlockingMiningManager(tmp_path)
    manager = _manager(tmp_path, mining)
    manager.create_session(
        _spec(max_wall_minutes=1),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="wall",
    )
    manager.approve("wall")
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        session = manager.store.get("wall")
        if session and session["active_run_id"]:
            break
        time.sleep(0.005)
    else:
        pytest.fail("mining trial did not start")
    manager.store.update(
        "wall",
        {"started_at": (datetime.now(UTC) - timedelta(minutes=2)).isoformat()},
    )

    completed = _wait(manager, "wall", "completed")
    assert completed["stop_reason"] == "max_wall_minutes"
    assert completed["trials"][0]["status"] == "stopped"
    assert completed["trials"][0]["error"] == "max_wall_minutes"
    assert mining.cancelled == [completed["trials"][0]["mining_run_id"]]


def test_pause_during_fingerprint_never_starts_a_leaf_until_resume(tmp_path: Path) -> None:
    mining = _BlockingMiningManager(tmp_path)
    fingerprint_started = threading.Event()
    release_fingerprint = threading.Event()
    app_state = SimpleNamespace(strategy_engine=_StrategyEngine())

    def fingerprint_builder(request):
        fingerprint_started.set()
        assert release_fingerprint.wait(1)
        return {"generation": "test"}

    manager = ResearchSessionManager(
        tmp_path,
        mining,
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        app_state,
        proposal_runner=lambda session, history: _proposal(
            ["momentum_10d", "turnover_rate"]
        ),
        fingerprint_builder=fingerprint_builder,
        evidence_loader=lambda store, run_id: {},
        holdout_reserver=_fake_holdout,
        poll_seconds=0.005,
    )
    manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="pause_race",
    )
    manager.approve("pause_race")
    assert fingerprint_started.wait(1)
    manager.pause("pause_race")
    release_fingerprint.set()
    time.sleep(0.03)
    assert mining.requests == []

    manager.resume("pause_race")
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline and not mining.requests:
        time.sleep(0.005)
    assert len(mining.requests) == 1
    manager.stop("pause_race")
    _wait(manager, "pause_race", "stopped")


def test_data_snapshot_change_stops_before_starting_another_leaf(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    fingerprints = iter([{"generation": "first"}, {"generation": "second"}])
    manager = ResearchSessionManager(
        tmp_path,
        mining,
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        SimpleNamespace(strategy_engine=_StrategyEngine()),
        proposal_runner=lambda session, history: _proposal(
            ["momentum_10d", "turnover_rate"]
        ),
        fingerprint_builder=lambda request: next(fingerprints),
        evidence_loader=lambda store, run_id: {
            "oos_sharpe": 1.0,
            "oos_max_drawdown": -0.1,
            "oos_positive_fold_ratio": 0.75,
            "oos_n_trades": 80,
            "valid_folds": 3,
            "qualified": True,
            "gate_reasons": [],
        },
        holdout_reserver=_fake_holdout,
        poll_seconds=0.005,
    )
    manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="data_changed",
    )
    manager.approve("data_changed")

    failed = _wait(manager, "data_changed", "failed")
    assert failed["completed_trials"] == 1
    assert failed["data_snapshot_digest"]
    assert failed["error"] == "research data snapshot changed during session"
    assert len(mining.requests) == 1


def test_compact_evidence_excludes_existing_strategy_benchmarks(tmp_path: Path) -> None:
    store = MiningRunStore(tmp_path)
    run_id = "factor_evidence"
    store.create({}, {"generation": "test"}, run_id=run_id)
    store.transition_status(run_id, "running")
    pl.DataFrame({
        "kind": ["factor_combination", "existing_strategy"],
        "oos_sharpe": [0.8, 3.0],
        "oos_max_drawdown": [-0.1, -0.02],
        "oos_positive_fold_ratio": [0.75, 1.0],
        "oos_n_trades": [80, 200],
        "valid_folds": [3, 3],
        "confidence": ["standard", "standard"],
    }).write_parquet(store.artifact_path(run_id, "candidates"))
    store.register_artifact(run_id, "candidates")
    store.write_summary(run_id, {"data_as_of": "2026-08-25"})
    store.transition_status(run_id, "succeeded")

    evidence = load_compact_mining_evidence(store, run_id)

    assert evidence["oos_sharpe"] == 0.8
    assert evidence["oos_n_trades"] == 80
    assert evidence["benchmark_sharpe"] == 3.0
    assert evidence["factor_names"] is None
    assert evidence["max_corr_pair"] is None


def _write_seal_candidate(store: MiningRunStore, run_id: str) -> str:
    definition = {
        "kind": "factor_rank",
        "factor_names": ["momentum_5d", "turnover_rate"],
        "scoring": {"momentum_5d": 1.0, "turnover_rate": 1.0},
        "directions": {"momentum_5d": "high", "turnover_rate": "high"},
    }
    signature = compute_candidate_signature(definition)
    pl.DataFrame({
        "kind": ["factor_combination"],
        "signature": [signature],
        "definition_json": [json.dumps(definition, sort_keys=True)],
        "factor_names_json": [json.dumps(["momentum_5d", "turnover_rate"])],
        "oos_sharpe": [0.8],
        "oos_max_drawdown": [-0.1],
        "oos_positive_fold_ratio": [0.75],
        "oos_n_trades": [80],
        "valid_folds": [3],
        "confidence": ["standard"],
    }).write_parquet(store.artifact_path(run_id, "candidates"))
    store.register_artifact(run_id, "candidates")
    return signature


def test_seal_writes_holdout_and_patches_origin_fingerprint(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    manager = ResearchSessionManager(
        tmp_path,
        mining,
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        SimpleNamespace(strategy_engine=_StrategyEngine()),
        proposal_runner=lambda session, history: _proposal(["rsi_14", "turnover_rate"]),
        fingerprint_builder=lambda request: {"generation": "test"},
        evidence_loader=lambda store, run_id: {
            "oos_sharpe": 0.8,
            "oos_max_drawdown": -0.1,
            "oos_positive_fold_ratio": 0.75,
            "oos_n_trades": 80,
            "valid_folds": 3,
            "qualified": True,
            "gate_reasons": [],
        },
        holdout_reserver=_fake_holdout,
        holdout_runner=lambda session, definition: {
            "sharpe": 0.9,
            "max_drawdown": -0.08,
            "n_trades": 70,
            "total_return": 0.12,
        },
        poll_seconds=0.005,
    )
    manager.create_session(
        _spec(max_trials=1, patience=1),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="seal",
    )
    manager.approve("seal")
    completed = _wait(manager, "seal", "completed")
    run_id = completed["trials"][0]["mining_run_id"]
    signature = _write_seal_candidate(mining.store, run_id)

    sealed_start = manager.seal("seal")
    assert sealed_start["status"] == "completed"
    assert sealed_start["holdout"]["status"] in {"sealing", "sealed"}
    sealed = _wait_holdout(manager, "seal", "sealed")
    assert sealed["holdout"]["qualified"] is True
    assert sealed["holdout"]["sharpe"] == 0.9
    assert sealed["holdout"]["signature"] == signature
    fingerprint = mining.store.get(run_id)["data_fingerprint"]
    assert fingerprint["final_holdout_sealed"] is True
    assert fingerprint["final_holdout"]["signature"] == signature
    assert fingerprint["final_holdout"]["sharpe"] == 0.9

    with pytest.raises(ResearchSessionConflictError, match="already sealed"):
        manager.seal("seal")
    manager.shutdown()


def test_seal_rejects_unfinished_session(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    manager = _manager(tmp_path, mining)
    manager.create_session(
        _spec(),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="unfinished",
    )
    with pytest.raises(InvalidResearchSessionTransitionError, match="finished"):
        manager.seal("unfinished")


def test_seal_rejects_in_progress_and_recover_fails_orphaned_sealing(tmp_path: Path) -> None:
    mining = _InstantMiningManager(tmp_path)
    started = threading.Event()
    release = threading.Event()

    def slow_runner(_session, _definition):
        started.set()
        assert release.wait(2)
        return {
            "sharpe": 0.9,
            "max_drawdown": -0.08,
            "n_trades": 70,
            "total_return": 0.12,
        }

    manager = ResearchSessionManager(
        tmp_path,
        mining,
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        SimpleNamespace(strategy_engine=_StrategyEngine()),
        proposal_runner=lambda session, history: _proposal(["rsi_14", "turnover_rate"]),
        fingerprint_builder=lambda request: {"generation": "test"},
        evidence_loader=lambda store, run_id: {
            "oos_sharpe": 0.8,
            "oos_max_drawdown": -0.1,
            "oos_positive_fold_ratio": 0.75,
            "oos_n_trades": 80,
            "valid_folds": 3,
            "qualified": True,
            "gate_reasons": [],
        },
        holdout_reserver=_fake_holdout,
        holdout_runner=slow_runner,
        poll_seconds=0.005,
    )
    manager.create_session(
        _spec(max_trials=1, patience=1),
        _proposal(["momentum_5d", "turnover_rate"]),
        session_id="seal-async",
    )
    manager.approve("seal-async")
    completed = _wait(manager, "seal-async", "completed")
    _write_seal_candidate(mining.store, completed["trials"][0]["mining_run_id"])

    first = manager.seal("seal-async")
    assert first["holdout"]["status"] == "sealing"
    assert started.wait(1)
    with pytest.raises(ResearchSessionConflictError, match="already being sealed"):
        manager.seal("seal-async")
    release.set()
    _wait_holdout(manager, "seal-async", "sealed")
    manager.shutdown()

    orphan = ResearchSessionManager(
        tmp_path,
        mining,
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        SimpleNamespace(strategy_engine=_StrategyEngine()),
        holdout_reserver=_fake_holdout,
        poll_seconds=0.005,
    )
    holdout = dict(orphan.store.get("seal-async")["holdout"])
    holdout["status"] = "sealing"
    orphan.store.update("seal-async", {"holdout": holdout})
    assert orphan.recover_interrupted() == 1
    recovered = orphan.store.get("seal-async")
    assert recovered["holdout"]["status"] == "failed"
    assert recovered["holdout"]["error"] == "application_restarted"
    orphan.shutdown()


def test_compact_evidence_adds_correlation_and_regime_summaries(tmp_path: Path) -> None:
    store = MiningRunStore(tmp_path)
    run_id = "thick_evidence"
    store.create({}, {"generation": "test"}, run_id=run_id)
    store.transition_status(run_id, "running")
    pl.DataFrame({
        "kind": ["factor_combination", "existing_strategy"],
        "signature": ["sig-a", "sig-b"],
        "factor_names_json": [
            json.dumps(["momentum_5d", "turnover_rate"]),
            json.dumps([]),
        ],
        "oos_sharpe": [0.6, 0.2],
        "oos_max_drawdown": [-0.1, -0.05],
        "oos_positive_fold_ratio": [0.67, 0.5],
        "oos_n_trades": [80, 40],
        "valid_folds": [3, 3],
        "confidence": ["standard", "standard"],
    }).write_parquet(store.artifact_path(run_id, "candidates"))
    store.register_artifact(run_id, "candidates")
    pl.DataFrame({
        "factor_x": ["momentum_5d", "momentum_5d"],
        "factor_y": ["momentum_5d", "turnover_rate"],
        "rho": [1.0, -0.42],
        "pair_count": [10, 10],
    }).write_parquet(store.artifact_path(run_id, "correlation"))
    store.register_artifact(run_id, "correlation")
    pl.DataFrame({
        "candidate_signature": ["sig-a", "sig-a", "sig-a"],
        "regime_state": ["strong", "range", "weak"],
        "sharpe": [0.5, -0.1, 0.2],
        "skipped": [False, False, False],
    }).write_parquet(store.artifact_path(run_id, "folds"))
    store.register_artifact(run_id, "folds")
    store.write_summary(run_id, {"data_as_of": "2026-08-25"})
    store.transition_status(run_id, "succeeded")

    evidence = load_compact_mining_evidence(store, run_id)
    assert evidence["factor_names"] == ["momentum_5d", "turnover_rate"]
    assert evidence["benchmark_sharpe"] == 0.2
    assert evidence["max_abs_correlation"] == 0.42
    assert evidence["max_corr_pair"] == ["momentum_5d", "turnover_rate"]
    assert evidence["regime_sharpe"] == {"strong": 0.5, "range": -0.1, "weak": 0.2}
