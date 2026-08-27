from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import polars as pl
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.autoresearch as api_module
import app.services.ai_provider as ai_provider
import app.services.autoresearch_manager as manager_module
from app.api.autoresearch import router
from app.api.mining import router as mining_router
from app.backtest.mining import compute_candidate_signature
from app.services.autoresearch_manager import ResearchSessionManager
from app.services.mining_jobs import MiningRunStore
from app.services.mining_preflight import (
    MiningPreflightError,
    reserve_final_holdout as real_holdout_reserve,
)
from app.services.research_sessions import catalog_proposal_digest


class _StrategyEngine:
    @staticmethod
    def list_strategies() -> list[dict]:
        return [{
            "id": "benchmark",
            "name": "基准策略",
            "execution_backend": "matrix_native",
            "timeframes": ["1d"],
            "asset_types": ["stock"],
        }]


class _InstantMiningManager:
    def __init__(self, data_dir: Path) -> None:
        self.store = MiningRunStore(data_dir)
        self.requests: list[tuple[dict, dict, str]] = []
        self.cancelled: list[str] = []

    def start(self, request, fingerprint, force=False, source="manual"):
        assert force is True
        self.requests.append((request, fingerprint, source))
        manifest = self.store.create(request, fingerprint)
        run_id = manifest["run_id"]
        self.store.append_event(run_id, "queued", {"status": "queued", "source": source})
        self.store.transition_status(run_id, "running")
        _write_artifacts(self.store, run_id, request["factor_names"][0])
        return self.store.transition_status(run_id, "succeeded")

    def cancel(self, run_id):
        self.cancelled.append(run_id)
        manifest = self.store.get(run_id)
        if manifest and manifest["status"] not in {"cancelled", "succeeded", "failed"}:
            return self.store.transition_status(run_id, "cancelled")
        return manifest


class _BlockingMiningManager(_InstantMiningManager):
    def start(self, request, fingerprint, force=False, source="manual"):
        assert force is True
        self.requests.append((request, fingerprint, source))
        manifest = self.store.create(request, fingerprint)
        run_id = manifest["run_id"]
        self.store.append_event(run_id, "queued", {"status": "queued", "source": source})
        return self.store.transition_status(run_id, "running")


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


def _proposal(factors: list[str], title: str) -> dict:
    value = {
        "title": title,
        "hypothesis": "目录内因子可能具有互补性",
        "rationale": "用真实 mining leaf 比较 compact evidence",
        "factor_names": factors,
        "strategy_ids": ["benchmark"],
        "expected_outcome": "样本外 Sharpe 改善且回撤受控",
        "risks": ["过拟合"],
        "ai_provider": "test",
        "ai_model": "test-model",
    }
    value["digest"] = catalog_proposal_digest(value)
    return value


def _write_artifacts(store: MiningRunStore, run_id: str, factor_name: str) -> str:
    definition = {
        "kind": "factor_rank",
        "factor_names": [factor_name],
        "scoring": {factor_name: 1.0},
        "directions": {factor_name: "high"},
    }
    signature = compute_candidate_signature(definition)
    frames = {
        "factors": pl.DataFrame({
            "factor_name": [factor_name],
            "label": [factor_name],
            "direction": [1],
            "score": [0.8],
            "ic_mean": [0.05],
            "ir": [0.6],
            "coverage": [1.0],
            "turnover": [0.2],
            "spread_return": [None],
            "spread_sharpe": [None],
            "selected": [True],
            "excluded_reason": [None],
        }),
        "correlation": pl.DataFrame({
            "factor_x": [factor_name],
            "factor_y": [factor_name],
            "rho": [1.0],
            "pair_count": [3],
        }),
        "candidates": pl.DataFrame({
            "signature": [signature],
            "name": [f"因子组合 · {factor_name}"],
            "kind": ["factor_combination"],
            "factor_names_json": [json.dumps([factor_name])],
            "strategy_id": [None],
            "definition_json": [json.dumps(definition, sort_keys=True)],
            "regime_state": ["overall"],
            "score": [0.8],
            "oos_return": [0.06],
            "oos_sharpe": [0.8],
            "oos_max_drawdown": [-0.12],
            "oos_positive_fold_ratio": [0.67],
            "oos_n_trades": [80],
            "confidence": ["standard"],
            "valid_folds": [3],
            "skipped_folds": [0],
            "promoted_candidate_id": [None],
            "published_strategy_id": [None],
        }, schema_overrides={
            "strategy_id": pl.String,
            "promoted_candidate_id": pl.String,
            "published_strategy_id": pl.String,
        }),
        "folds": pl.DataFrame({
            "candidate_signature": [signature],
            "fold": [0],
            "label": ["OOS 1"],
            "regime_state": ["overall"],
            "n_dates": [63],
            "train_start": ["2025-01-01"],
            "train_end": ["2025-12-31"],
            "test_start": ["2026-01-01"],
            "test_end": ["2026-03-31"],
            "selected_factors_json": [json.dumps([factor_name])],
            "evaluation_kind": ["selected"],
            "total_return": [0.06],
            "sharpe": [0.8],
            "max_drawdown": [-0.12],
            "n_trades": [80],
            "skipped": [False],
            "reason": [None],
        }),
    }
    for name, frame in frames.items():
        frame.write_parquet(store.artifact_path(run_id, name))
        store.register_artifact(run_id, name)
    store.write_summary(run_id, {
        "factor_count": 1,
        "selected_factor_count": 1,
        "candidate_count": 1,
        "valid_fold_count": 3,
        "skipped_fold_count": 0,
        "confidence": "standard",
        "budget_exhausted": False,
        "elapsed_ms": 10.0,
        "data_as_of": "2026-08-26",
        "methodology_version": "factor_v2",
        "algorithm_version": "e2e",
    })
    return signature


def _payload(**updates) -> dict:
    value = {
        "goal": "研究量价因子稳定性",
        "asset_type": "stock",
        "budget_profile": "exploratory",
        "max_trials": 2,
        "max_wall_minutes": 30,
        "patience": 2,
    }
    value.update(updates)
    return value


def _client(tmp_path: Path, monkeypatch, mining, *, proposals: list[dict], evidence=(), holdout_runner=None) -> tuple[TestClient, ResearchSessionManager]:
    next_proposals = list(proposals[1:])
    evidence_values = iter(evidence)
    engine = _StrategyEngine()
    manager = ResearchSessionManager(
        tmp_path,
        mining,
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        SimpleNamespace(strategy_engine=engine),
        proposal_runner=lambda _session, _history: next_proposals.pop(0),
        fingerprint_builder=lambda _request: {"generation": "e2e-snapshot"},
        evidence_loader=lambda _store, _run_id: next(evidence_values),
        holdout_reserver=_fake_holdout,
        holdout_runner=holdout_runner,
        poll_seconds=0.002,
    )

    async def generate_initial(**_kwargs):
        return proposals[0]

    monkeypatch.setattr(ai_provider, "ai_configured", lambda: True)
    monkeypatch.setattr(api_module, "generate_catalog_proposal", generate_initial)
    monkeypatch.setattr(api_module, "reserve_final_holdout", _fake_holdout)
    monkeypatch.setattr(manager_module, "require_mining_availability", lambda *args, **kwargs: None)

    app = FastAPI()
    app.include_router(router)
    app.include_router(mining_router)
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.state.strategy_engine = engine
    app.state.mining_manager = mining
    app.state.autoresearch_manager = manager
    return TestClient(app), manager


def _wait(client: TestClient, session_id: str, status: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/backtest/autoresearch/sessions/{session_id}")
        assert response.status_code == 200
        session = response.json()
        if session["status"] == status:
            return session
        time.sleep(0.005)
    pytest.fail(f"session {session_id} did not reach {status}")


def _wait_holdout(client: TestClient, session_id: str, status: str, timeout: float = 2.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/backtest/autoresearch/sessions/{session_id}")
        assert response.status_code == 200
        session = response.json()
        if (session.get("holdout") or {}).get("status") == status:
            return session
        time.sleep(0.005)
    pytest.fail(f"holdout {session_id} did not reach {status}")


def test_catalog_research_http_e2e_runs_trials_streams_events_and_survives_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    proposals = [
        _proposal(["momentum_5d", "turnover_rate"], "首轮量价假设"),
        _proposal(["rsi_14", "turnover_rate"], "修正后的反转假设"),
    ]
    evidence = [
        {
            "oos_sharpe": -0.2,
            "oos_max_drawdown": -0.2,
            "oos_positive_fold_ratio": 0.33,
            "oos_n_trades": 40,
            "valid_folds": 3,
            "qualified": False,
            "gate_reasons": ["样本外 Sharpe 低于 0.5"],
        },
        {
            "oos_sharpe": 0.8,
            "oos_max_drawdown": -0.12,
            "oos_positive_fold_ratio": 0.67,
            "oos_n_trades": 80,
            "valid_folds": 3,
            "qualified": True,
            "gate_reasons": [],
        },
    ]
    mining = _InstantMiningManager(tmp_path)
    client, manager = _client(
        tmp_path,
        monkeypatch,
        mining,
        proposals=proposals,
        evidence=evidence,
    )
    try:
        created = client.post("/api/backtest/autoresearch/sessions", json=_payload())
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        assert created.json()["status"] == "awaiting_approval"
        assert created.json()["adaptive_end"] == "2024-09-30"
        assert created.json()["holdout"]["start"] == "2024-10-08"

        approved = client.post(f"/api/backtest/autoresearch/sessions/{session_id}/approve")
        assert approved.status_code == 200
        assert approved.json()["status"] == "running"

        completed = _wait(client, session_id, "completed")
        assert completed["stop_reason"] == "max_trials"
        assert [trial["status"] for trial in completed["trials"]] == ["completed", "completed"]
        assert [row["oos_sharpe"] for row in completed["leaderboard"]] == [0.8, -0.2]
        assert completed["data_snapshot_digest"]
        assert len(mining.requests) == 2
        assert all(source == f"autoresearch:{session_id}" for _, _, source in mining.requests)
        assert all(
            fingerprint["source"] == "autoresearch"
            and fingerprint["source_session_id"] == session_id
            and fingerprint["final_holdout_sealed"] is False
            for _, fingerprint, _ in mining.requests
        )
        assert all(request["end"] == "2024-09-30" for request, _, _ in mining.requests)

        listed = client.get(
            "/api/backtest/autoresearch/sessions",
            params=[("status", "completed")],
        )
        assert [item["session_id"] for item in listed.json()["items"]] == [session_id]
        stream = client.get(f"/api/backtest/autoresearch/sessions/{session_id}/events")
        assert stream.status_code == 200
        assert "event: snapshot" in stream.text
        assert "event: trial_completed" in stream.text
        assert "event: completed" in stream.text

        first_run_id = completed["trials"][0]["mining_run_id"]
        first_candidate = client.get(
            f"/api/backtest/mining/runs/{first_run_id}/result"
        ).json()["candidates"][0]
        assert first_candidate["publishable"] is False
        assert "最终留出集" in first_candidate["publish_block_reason"]
        signature = first_candidate["signature"]
        promoted = client.post(
            f"/api/backtest/mining/runs/{first_run_id}/candidates/"
            f"{quote(signature, safe='')}/promote"
        )
        assert promoted.status_code == 200
        assert promoted.json()["status"] == "pending"
        blocked = client.post(
            f"/api/backtest/mining/runs/{first_run_id}/candidates/"
            f"{quote(signature, safe='')}/publish"
        )
        assert blocked.status_code == 400
        assert "sealed final holdout" in blocked.json()["detail"]

        interrupted = client.post(
            "/api/backtest/autoresearch/sessions",
            json=_payload(max_trials=1, patience=1),
        ).json()
        manager.store.transition(interrupted["session_id"], "running")
        assert manager.recover_interrupted() == 1
        recovered = client.get(
            f"/api/backtest/autoresearch/sessions/{interrupted['session_id']}"
        ).json()
        assert recovered["status"] == "interrupted"
        assert recovered["stop_reason"] == "application_restarted"
    finally:
        manager.shutdown()


def test_catalog_research_http_e2e_controls_pause_resume_and_stop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mining = _BlockingMiningManager(tmp_path)
    client, manager = _client(
        tmp_path,
        monkeypatch,
        mining,
        proposals=[_proposal(["momentum_5d", "turnover_rate"], "可控制研究")],
    )
    try:
        created = client.post(
            "/api/backtest/autoresearch/sessions",
            json=_payload(max_trials=1, patience=1),
        ).json()
        session_id = created["session_id"]
        assert client.post(
            f"/api/backtest/autoresearch/sessions/{session_id}/approve"
        ).status_code == 200

        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            running = client.get(
                f"/api/backtest/autoresearch/sessions/{session_id}"
            ).json()
            if running["active_run_id"]:
                break
            time.sleep(0.005)
        else:
            pytest.fail("mining leaf did not start")

        paused = client.post(
            f"/api/backtest/autoresearch/sessions/{session_id}/pause"
        )
        assert paused.status_code == 200
        assert paused.json()["status"] == "paused"
        resumed = client.post(
            f"/api/backtest/autoresearch/sessions/{session_id}/resume"
        )
        assert resumed.status_code == 200
        assert resumed.json()["status"] == "running"
        stopping = client.post(
            f"/api/backtest/autoresearch/sessions/{session_id}/stop"
        )
        assert stopping.status_code == 200

        stopped = _wait(client, session_id, "stopped")
        assert stopped["stop_reason"] == "user_requested"
        assert stopped["trials"][0]["status"] == "stopped"
        assert mining.cancelled == [stopped["trials"][0]["mining_run_id"]]
    finally:
        manager.shutdown()


def test_catalog_research_http_e2e_preflights_before_ai_and_again_on_approval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mining = _InstantMiningManager(tmp_path)
    client, manager = _client(
        tmp_path,
        monkeypatch,
        mining,
        proposals=[_proposal(["momentum_5d", "turnover_rate"], "预检研究")],
    )
    try:
        ai_calls: list[str] = []

        async def generate_initial(**_kwargs):
            ai_calls.append("called")
            return _proposal(["momentum_5d", "turnover_rate"], "预检研究")

        monkeypatch.setattr(api_module, "generate_catalog_proposal", generate_initial)
        monkeypatch.setattr(api_module, "reserve_final_holdout", real_holdout_reserve)
        missing_data = client.post(
            "/api/backtest/autoresearch/sessions",
            json=_payload(max_trials=1, patience=1),
        )
        assert missing_data.status_code == 400
        assert missing_data.headers["X-Mining-Preflight-Code"] == "enriched_insufficient"
        assert ai_calls == []
        assert manager.store.list() == []

        monkeypatch.setattr(api_module, "reserve_final_holdout", _fake_holdout)
        created = client.post(
            "/api/backtest/autoresearch/sessions",
            json=_payload(max_trials=1, patience=1),
        ).json()
        assert ai_calls == ["called"]

        def fail_approval(*_args, **_kwargs):
            raise MiningPreflightError("regime_unavailable", "市场环境数据不可用")

        monkeypatch.setattr(manager_module, "require_mining_availability", fail_approval)
        rejected = client.post(
            f"/api/backtest/autoresearch/sessions/{created['session_id']}/approve"
        )
        assert rejected.status_code == 400
        assert rejected.headers["X-Mining-Preflight-Code"] == "regime_unavailable"
        assert client.get(
            f"/api/backtest/autoresearch/sessions/{created['session_id']}"
        ).json()["status"] == "awaiting_approval"
        assert mining.requests == []
    finally:
        manager.shutdown()


def test_catalog_research_http_e2e_seals_holdout_and_updates_publication_block(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mining = _InstantMiningManager(tmp_path)
    client, manager = _client(
        tmp_path,
        monkeypatch,
        mining,
        proposals=[_proposal(["momentum_5d", "turnover_rate"], "密封研究")],
        evidence=[{
            "oos_sharpe": 0.8,
            "oos_max_drawdown": -0.12,
            "oos_positive_fold_ratio": 0.67,
            "oos_n_trades": 80,
            "valid_folds": 3,
            "qualified": True,
            "gate_reasons": [],
        }],
        holdout_runner=lambda _session, _definition: {
            "sharpe": 0.9,
            "max_drawdown": -0.08,
            "n_trades": 70,
            "total_return": 0.1,
        },
    )
    try:
        created = client.post(
            "/api/backtest/autoresearch/sessions",
            json=_payload(max_trials=1, patience=1),
        )
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        assert client.post(
            f"/api/backtest/autoresearch/sessions/{session_id}/approve"
        ).status_code == 200
        completed = _wait(client, session_id, "completed")
        run_id = completed["trials"][0]["mining_run_id"]

        unfinished = client.post("/api/backtest/autoresearch/sessions/missing/seal")
        assert unfinished.status_code == 404

        posted = client.post(f"/api/backtest/autoresearch/sessions/{session_id}/seal")
        assert posted.status_code == 200
        assert posted.json()["status"] == "completed"
        assert posted.json()["holdout"]["status"] in {"sealing", "sealed"}
        body = _wait_holdout(client, session_id, "sealed")
        assert body["holdout"]["qualified"] is True
        assert body["holdout"]["sharpe"] == 0.9

        conflict = client.post(f"/api/backtest/autoresearch/sessions/{session_id}/seal")
        assert conflict.status_code == 409

        candidate = client.get(
            f"/api/backtest/mining/runs/{run_id}/result"
        ).json()["candidates"][0]
        assert candidate["publishable"] is True
        assert candidate["publish_block_reason"] is None
        fingerprint = mining.store.get(run_id)["data_fingerprint"]
        assert fingerprint["final_holdout_sealed"] is True
        assert fingerprint["final_holdout"]["sharpe"] == 0.9
    finally:
        manager.shutdown()
