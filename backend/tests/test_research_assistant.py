from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.mining import (
    MiningAssistantPlanRequest,
    MiningAssistantReportRequest,
    create_assistant_plan,
    create_assistant_report,
)
from app.services.mining_jobs import compute_run_signature
from app.services.research_assistant import (
    build_plan_messages,
    build_report_messages,
    parse_plan,
)

FACTORS = [
    {"id": "momentum_20d", "label": "20日动量", "desc": "20日收益"},
    {"id": "annual_vol_20d", "label": "20日波动", "desc": "年化波动"},
]
STRATEGIES = [
    {
        "id": "trend_breakout",
        "name": "趋势突破",
        "description": "趋势策略",
        "execution_backend": "matrix_native",
        "asset_types": ["stock"],
        "timeframes": ["1d"],
    }
]


def _plan_text(**overrides) -> str:
    value = {
        "title": "动量与低波研究",
        "hypothesis": "动量叠加低波动能改善样本外稳定性",
        "rationale": "两个因子刻画收益趋势与风险",
        "factor_names": ["momentum_20d", "annual_vol_20d"],
        "strategy_ids": ["trend_breakout"],
        "expected_outcome": "样本外指标优于对照且回撤不恶化",
        "risks": ["参数过拟合"],
    }
    value.update(overrides)
    return json.dumps(value, ensure_ascii=False)


def test_plan_prompt_contains_only_supplied_catalog() -> None:
    messages = build_plan_messages(
        "研究动量和波动",
        FACTORS,
        STRATEGIES,
        asset_type="stock",
        budget_profile="balanced",
    )
    prompt = messages[0]["content"]
    assert "momentum_20d" in prompt
    assert "annual_vol_20d" in prompt
    assert "trend_breakout" in prompt
    assert "roe_latest" not in prompt


def test_parse_plan_accepts_fenced_json_and_validates_whitelists() -> None:
    plan = parse_plan(
        f"```json\n{_plan_text()}\n```",
        factor_ids={item["id"] for item in FACTORS},
        strategy_ids={item["id"] for item in STRATEGIES},
    )
    assert plan["factor_names"] == ["momentum_20d", "annual_vol_20d"]
    assert plan["strategy_ids"] == ["trend_breakout"]


def test_parse_plan_rejects_unknown_factor() -> None:
    with pytest.raises(ValueError, match="未知因子"):
        parse_plan(
            _plan_text(factor_names=["momentum_20d", "invented_factor"]),
            factor_ids={item["id"] for item in FACTORS},
            strategy_ids={item["id"] for item in STRATEGIES},
        )


def test_parse_plan_rejects_extra_execution_fields() -> None:
    with pytest.raises(ValueError, match="不支持的字段"):
        parse_plan(
            _plan_text(code="print('unsafe')"),
            factor_ids={item["id"] for item in FACTORS},
            strategy_ids={item["id"] for item in STRATEGIES},
        )


def test_research_context_does_not_change_execution_signature() -> None:
    request = {"factor_names": ["momentum_20d"]}
    with_context = {
        **request,
        "research_context": {"goal": "研究动量", "title": "AI 生成的标题"},
    }
    assert compute_run_signature(request, {"data": "same"}) == compute_run_signature(
        with_context,
        {"data": "same"},
    )


@pytest.mark.asyncio
async def test_plan_endpoint_is_gone() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await create_assistant_plan(
            MiningAssistantPlanRequest(goal="研究动量和低波动的互补性"),
        )
    assert exc_info.value.status_code == 410
    assert "/api/backtest/autoresearch/sessions" in str(exc_info.value.detail)


def test_report_prompt_contains_only_provided_evidence() -> None:
    messages = build_report_messages(
        {
            "data_as_of": "2026-08-25",
            "summary": {"confidence": "standard"},
            "request_summary": {"budget_profile": "balanced"},
            "factors": [{"factor_name": "momentum_20d", "selected": True}],
            "candidates": [{"oos_sharpe": 0.7, "gate": {"qualified": True}}],
            "regimes": [],
            "folds": [],
        },
        {"goal": "测试目标"},
    )
    evidence = json.loads(messages[1]["content"])
    assert evidence["candidates"][0]["oos_sharpe"] == 0.7
    assert evidence["research_context"]["goal"] == "测试目标"


@pytest.mark.asyncio
async def test_report_endpoint_uses_projected_result(monkeypatch) -> None:
    import app.api.mining as mining_api
    import app.services.ai_provider as ai_provider

    captured: dict = {}

    async def fake_generate(messages, **_kwargs):
        captured.update(json.loads(messages[1]["content"]))
        return "结论: 证据不足。"

    monkeypatch.setattr(ai_provider, "ai_configured", lambda: True)
    monkeypatch.setattr(ai_provider, "current_ai_provider", lambda: "test-provider")
    monkeypatch.setattr(ai_provider, "current_ai_model", lambda: "test-model")
    monkeypatch.setattr(ai_provider, "generate_ai_text", fake_generate)
    monkeypatch.setattr(mining_api, "get_result", lambda _run_id, _request: {
        "run_id": "run-1",
        "data_as_of": "2026-08-25",
        "summary": {"confidence": "low"},
        "request_summary": {"budget_profile": "exploratory"},
        "factors": [],
        "candidates": [],
        "regimes": [],
        "folds": [],
    })
    monkeypatch.setattr(mining_api, "_required_manifest", lambda _store, _run_id: {
        "request": {"research_context": {"goal": "测试目标"}},
    })
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(
        mining_manager=SimpleNamespace(store=object()),
    )))

    result = await create_assistant_report(
        MiningAssistantReportRequest(run_id="run-1"),
        request,
    )

    assert result["report"] == "结论: 证据不足。"
    assert captured["summary"]["confidence"] == "low"
    assert captured["research_context"]["goal"] == "测试目标"
