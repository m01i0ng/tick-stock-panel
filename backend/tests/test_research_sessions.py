from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.services.research_sessions import (
    InvalidResearchSessionTransitionError,
    ResearchSessionStore,
    ResearchSessionValidationError,
    catalog_proposal_digest,
    generate_catalog_proposal,
    parse_catalog_proposal,
    validate_followup_step,
)


def _spec(**updates):
    value = {
        "goal": "研究量价因子的样本外稳定性",
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
        "max_trials": 3,
        "max_wall_minutes": 60,
        "patience": 2,
    }
    value.update(updates)
    return value


def _proposal(factors=None, strategies=None):
    value = {
        "title": "量价组合",
        "hypothesis": "量价组合具有稳定样本外表现",
        "rationale": "组合互补因子",
        "factor_names": factors or ["momentum_5d", "turnover_rate"],
        "strategy_ids": strategies or [],
        "expected_outcome": "样本外指标改善",
        "risks": ["过拟合"],
    }
    value["digest"] = catalog_proposal_digest(value)
    return value


def test_store_persists_session_events_and_strict_transitions(tmp_path: Path) -> None:
    store = ResearchSessionStore(tmp_path)
    created = store.create(_spec(), _proposal(), session_id="catalog_session")

    assert created["status"] == "awaiting_approval"
    assert ResearchSessionStore(tmp_path).get("catalog_session") == created
    store.append_event("catalog_session", "created", {"status": "awaiting_approval"})
    assert store.read_events("catalog_session")[0]["type"] == "created"

    running = store.transition("catalog_session", "running", updates={"approved_at": "now"})
    assert running["started_at"] is not None
    paused = store.transition("catalog_session", "paused")
    assert paused["status"] == "paused"
    store.transition("catalog_session", "running")
    completed = store.transition("catalog_session", "completed")
    assert completed["finished_at"] is not None
    with pytest.raises(InvalidResearchSessionTransitionError):
        store.transition("catalog_session", "running")


def test_store_rejects_bad_digest_and_protected_updates(tmp_path: Path) -> None:
    store = ResearchSessionStore(tmp_path)
    bad = _proposal()
    bad["factor_names"] = ["momentum_10d", "turnover_rate"]
    with pytest.raises(ResearchSessionValidationError, match="digest"):
        store.create(_spec(), bad)

    store.create(_spec(), _proposal(), session_id="protected")
    with pytest.raises(ResearchSessionValidationError, match="protected"):
        store.update("protected", {"status": "completed"})


def test_recover_marks_only_active_sessions_interrupted(tmp_path: Path) -> None:
    store = ResearchSessionStore(tmp_path)
    for session_id, status in (
        ("running_before_restart", "running"),
        ("paused_before_restart", "paused"),
        ("waiting_for_approval", "awaiting_approval"),
    ):
        store.create(_spec(), _proposal(), session_id=session_id)
        if status != "awaiting_approval":
            store.transition(session_id, "running")
        if status == "paused":
            store.transition(session_id, "paused")

    assert store.recover_interrupted() == 2
    assert store.get("running_before_restart")["status"] == "interrupted"  # type: ignore[index]
    assert store.get("paused_before_restart")["status"] == "interrupted"  # type: ignore[index]
    assert store.get("waiting_for_approval")["status"] == "awaiting_approval"  # type: ignore[index]


def test_catalog_proposal_digest_deduplicates_execution_choices() -> None:
    first = _proposal()
    renamed = {**first, "title": "另一个标题", "hypothesis": "另一段解释"}
    reordered = _proposal(factors=["turnover_rate", "momentum_5d"])

    assert catalog_proposal_digest(first) == catalog_proposal_digest(renamed)
    assert catalog_proposal_digest(first) == catalog_proposal_digest(reordered)


def test_validate_followup_step_allows_one_factor_swap_and_rejects_multi_axis() -> None:
    previous = _proposal(["momentum_5d", "turnover_rate"])
    validate_followup_step(previous, _proposal(["momentum_10d", "turnover_rate"]))
    validate_followup_step(previous, _proposal(["momentum_5d", "turnover_rate"], ["benchmark"]))
    with pytest.raises(ValueError, match="factors and strategies"):
        validate_followup_step(
            previous,
            _proposal(["momentum_10d", "turnover_rate"], ["benchmark"]),
        )
    with pytest.raises(ValueError, match="at most one factor"):
        validate_followup_step(previous, _proposal(["rsi_14", "volume_ratio"]))


def test_parse_catalog_proposal_rejects_ai_ids_outside_whitelist() -> None:
    raw = json.dumps(
        {
            "title": "非法计划",
            "hypothesis": "测试未知因子",
            "rationale": "测试",
            "factor_names": ["momentum_5d", "ai_invented_factor"],
            "strategy_ids": [],
            "expected_outcome": "无",
            "risks": [],
        }
    )
    with pytest.raises(ValueError, match="未知因子"):
        parse_catalog_proposal(
            raw,
            factor_ids={"momentum_5d", "turnover_rate"},
            strategy_ids=set(),
        )


@pytest.mark.asyncio
async def test_generate_catalog_proposal_leaves_reasoning_output_unbounded(
    monkeypatch,
) -> None:
    import app.services.ai_provider as ai_provider

    captured = {}

    async def fake_generate(*_args, **kwargs):
        captured.update(kwargs)
        return json.dumps(
            {key: value for key, value in _proposal().items() if key != "digest"}
        )

    monkeypatch.setattr(ai_provider, "generate_ai_text", fake_generate)
    monkeypatch.setattr(ai_provider, "current_ai_provider", lambda: "test-provider")
    monkeypatch.setattr(ai_provider, "current_ai_model", lambda: "test-model")

    proposal = await generate_catalog_proposal(
        goal="研究量价互补性",
        factors=[
            {"id": "momentum_5d", "label": "五日动量"},
            {"id": "turnover_rate", "label": "换手率"},
        ],
        strategies=[],
        asset_type="stock",
        budget_profile="exploratory",
        timeout=12.5,
    )

    assert proposal["ai_model"] == "test-model"
    assert captured["max_tokens"] is None
    assert captured["timeout"] == 12.5


@pytest.mark.asyncio
async def test_generate_catalog_proposal_followup_prompt_requires_evidence_and_one_axis(
    monkeypatch,
) -> None:
    import app.services.ai_provider as ai_provider

    captured: dict = {}

    async def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return json.dumps(
            {key: value for key, value in _proposal().items() if key != "digest"}
        )

    monkeypatch.setattr(ai_provider, "generate_ai_text", fake_generate)
    monkeypatch.setattr(ai_provider, "current_ai_provider", lambda: "test-provider")
    monkeypatch.setattr(ai_provider, "current_ai_model", lambda: "test-model")

    await generate_catalog_proposal(
        goal="研究量价互补性",
        factors=[
            {"id": "momentum_5d", "label": "五日动量"},
            {"id": "turnover_rate", "label": "换手率"},
            {"id": "momentum_10d", "label": "十日动量"},
        ],
        strategies=[],
        asset_type="stock",
        budget_profile="exploratory",
        history=[{
            "proposal": _proposal(),
            "evidence": {
                "benchmark_sharpe": 0.2,
                "max_corr_pair": ["momentum_5d", "turnover_rate"],
                "regime_sharpe": {"weak": -0.1},
            },
            "improved": False,
        }],
    )

    prompt = captured["messages"][0]["content"]
    assert "benchmark_sharpe" in prompt
    assert "max_corr_pair" in prompt
    assert "regime_sharpe" in prompt
    assert "只改一个轴" in prompt
    assert "历史试验(JSON)" in captured["messages"][1]["content"]


def test_session_file_is_atomic_json(tmp_path: Path) -> None:
    store = ResearchSessionStore(tmp_path)
    store.create(_spec(), _proposal(), session_id="atomic")
    old = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
    store.update("atomic", {"started_at": old})
    path = tmp_path / "research" / "autoresearch" / "sessions" / "atomic" / "session.json"
    assert json.loads(path.read_text(encoding="utf-8"))["started_at"] == old
    assert not list(path.parent.glob("*.tmp"))
    assert not list(path.parent.glob(".*.tmp"))
