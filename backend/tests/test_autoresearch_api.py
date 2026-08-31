from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.api.autoresearch as api_module
import app.services.ai_provider as ai_provider
from app.api.autoresearch import router
from app.services.autoresearch_manager import ResearchSessionManager
from app.services.mining_jobs import MiningRunStore
from app.services.research_sessions import catalog_proposal_digest


class _StrategyEngine:
    @staticmethod
    def list_strategies():
        return []


class _MiningManager:
    def __init__(self, data_dir: Path) -> None:
        self.store = MiningRunStore(data_dir)

    def start(self, request, fingerprint, force=False, source="manual"):
        raise AssertionError("approval is not part of this API fixture")

    def cancel(self, run_id):
        raise AssertionError("no leaf run exists")


def _proposal():
    value = {
        "title": "量价计划",
        "hypothesis": "量价因子存在互补性",
        "rationale": "选择两个现有因子",
        "factor_names": ["momentum_5d", "turnover_rate"],
        "strategy_ids": [],
        "expected_outcome": "样本外指标改善",
        "risks": ["过拟合"],
        "ai_provider": "test",
        "ai_model": "test-model",
    }
    value["digest"] = catalog_proposal_digest(value)
    return value


def _client(tmp_path: Path, monkeypatch):
    engine = _StrategyEngine()
    mining = _MiningManager(tmp_path)
    state = SimpleNamespace(strategy_engine=engine)
    manager = ResearchSessionManager(
        tmp_path,
        mining,
        SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path)),
        state,
        proposal_runner=lambda session, history: _proposal(),
        fingerprint_builder=lambda request: {"generation": "test"},
    )
    app = FastAPI()
    app.include_router(router)
    app.state.repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    app.state.strategy_engine = engine
    app.state.autoresearch_manager = manager
    monkeypatch.setattr(ai_provider, "ai_configured", lambda: True)
    monkeypatch.setattr(
        api_module,
        "reserve_final_holdout",
        lambda *args, **kwargs: {
            "adaptive_end": "2024-09-30",
            "start": "2024-10-08",
            "end": "2025-01-01",
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
        },
    )
    monkeypatch.setattr(
        api_module,
        "generate_catalog_proposal",
        lambda **kwargs: _AsyncValue(_proposal()),
    )
    return TestClient(app), manager


class _AsyncValue:
    def __init__(self, value):
        self.value = value

    def __await__(self):
        async def resolve():
            return self.value

        return resolve().__await__()


def _payload():
    return {
        "goal": "研究量价因子稳定性",
        "asset_type": "stock",
        "budget_profile": "exploratory",
        "max_trials": 2,
        "max_wall_minutes": 30,
        "patience": 1,
    }


def test_create_list_get_stop_and_snapshot_contract(tmp_path: Path, monkeypatch) -> None:
    client, _manager = _client(tmp_path, monkeypatch)
    created = client.post("/api/backtest/autoresearch/sessions", json=_payload())
    assert created.status_code == 200
    session = created.json()
    assert session["status"] == "awaiting_approval"
    assert session["initial_proposal"]["digest"]
    assert session["trials"] == []
    assert session["commission_pct"] == 0.0002
    assert session["adaptive_end"] == "2024-09-30"
    assert session["holdout"]["status"] == "reserved"
    assert session["holdout"]["bars"] == 63

    session_id = session["session_id"]
    listed = client.get("/api/backtest/autoresearch/sessions").json()
    assert [item["session_id"] for item in listed["items"]] == [session_id]
    assert client.get(f"/api/backtest/autoresearch/sessions/{session_id}").json() == session

    stopped = client.post(f"/api/backtest/autoresearch/sessions/{session_id}/stop")
    assert stopped.status_code == 200
    assert stopped.json()["status"] == "stopped"
    invalid = client.post(f"/api/backtest/autoresearch/sessions/{session_id}/approve")
    assert invalid.status_code == 409

    stream = client.get(f"/api/backtest/autoresearch/sessions/{session_id}/events")
    assert stream.status_code == 200
    assert "event: snapshot" in stream.text
    assert '"status": "stopped"' in stream.text


def test_create_rejects_second_active_session_with_409(tmp_path: Path, monkeypatch) -> None:
    client, _manager = _client(tmp_path, monkeypatch)
    assert client.post("/api/backtest/autoresearch/sessions", json=_payload()).status_code == 200
    conflict = client.post("/api/backtest/autoresearch/sessions", json=_payload())
    assert conflict.status_code == 409


def test_create_maps_invalid_ai_catalog_plan_to_502(tmp_path: Path, monkeypatch) -> None:
    client, _manager = _client(tmp_path, monkeypatch)
    monkeypatch.setattr(
        api_module,
        "generate_catalog_proposal",
        lambda **kwargs: _AsyncFailure(ValueError("研究计划包含未知因子")),
    )
    response = client.post("/api/backtest/autoresearch/sessions", json=_payload())
    assert response.status_code == 502
    assert "未知因子" in response.json()["detail"]


class _AsyncFailure:
    def __init__(self, error):
        self.error = error

    def __await__(self):
        async def fail():
            raise self.error

        return fail().__await__()


def test_create_request_is_strict_and_validates_patience(tmp_path: Path, monkeypatch) -> None:
    client, _manager = _client(tmp_path, monkeypatch)
    payload = {**_payload(), "unknown": True}
    assert client.post("/api/backtest/autoresearch/sessions", json=payload).status_code == 422
    payload = {**_payload(), "max_trials": 1, "patience": 2}
    assert client.post("/api/backtest/autoresearch/sessions", json=payload).status_code == 422
