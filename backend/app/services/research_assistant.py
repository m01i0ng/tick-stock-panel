"""AI research planning and result reporting over the mining contract."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from app.strategy.custom_signals_ai import _extract_json_object

MAX_ASSISTANT_FACTORS = 12


def build_plan_messages(
    goal: str,
    factors: Sequence[Mapping[str, Any]],
    strategies: Sequence[Mapping[str, Any]],
    *,
    asset_type: str,
    budget_profile: str,
) -> list[dict[str, str]]:
    factor_lines = "\n".join(
        f"- {item['id']}: {item.get('label', item['id'])}; {item.get('desc', '')}"
        for item in factors
    )
    strategy_lines = "\n".join(
        f"- {item['id']}: {item.get('name', item['id'])}; {item.get('description', '')}"
        for item in strategies
    ) or "- 无可用对照策略"
    system = f"""你是A股量化研究规划助手。你只负责提出一个受控研究计划, 不能创造因子、公式、策略ID或回测结果。

资产类型: {asset_type}
验证档位: {budget_profile}

可用因子白名单:
{factor_lines}

可用对照策略白名单:
{strategy_lines}

只输出一个 JSON 对象, 不要 markdown 或解释。结构固定为:
{{
  "title": "简短研究标题",
  "hypothesis": "可证伪的研究假设",
  "rationale": "为什么选择这些因子",
  "factor_names": ["白名单因子ID"],
  "strategy_ids": ["可选的白名单策略ID"],
  "expected_outcome": "什么结果支持或否定假设",
  "risks": ["关键数据或过拟合风险"]
}}

约束:
1. 选择 2 到 {MAX_ASSISTANT_FACTORS} 个互补因子, 宁少勿多。
2. 对照策略最多 3 个; 没有合适对照时返回空数组。
3. 不预测收益, 不声称策略有效, 不输出阈值或任意代码。
4. hypothesis 和 expected_outcome 必须可由样本外 Sharpe、最大回撤、正收益折比例、交易数和市场环境结果验证。
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": goal},
    ]


def parse_plan(
    text: str,
    *,
    factor_ids: set[str] | frozenset[str],
    strategy_ids: set[str] | frozenset[str],
) -> dict[str, Any]:
    raw = _extract_json_object(text)
    if not isinstance(raw, dict):
        raise ValueError("AI 返回的研究计划不是 JSON 对象")
    allowed_fields = {
        "title", "hypothesis", "rationale", "factor_names", "strategy_ids",
        "expected_outcome", "risks",
    }
    extra_fields = sorted(set(raw) - allowed_fields)
    if extra_fields:
        raise ValueError(f"研究计划包含不支持的字段: {extra_fields}")

    factor_names = _unique_string_list(raw.get("factor_names"), "factor_names")
    if not 2 <= len(factor_names) <= MAX_ASSISTANT_FACTORS:
        raise ValueError(f"研究计划必须选择 2 到 {MAX_ASSISTANT_FACTORS} 个因子")
    unknown_factors = sorted(set(factor_names) - set(factor_ids))
    if unknown_factors:
        raise ValueError(f"研究计划包含未知因子: {unknown_factors}")

    selected_strategies = _unique_string_list(
        raw.get("strategy_ids", []), "strategy_ids", allow_empty=True
    )
    if len(selected_strategies) > 3:
        raise ValueError("研究计划最多选择 3 个对照策略")
    unknown_strategies = sorted(set(selected_strategies) - set(strategy_ids))
    if unknown_strategies:
        raise ValueError(f"研究计划包含未知策略: {unknown_strategies}")

    risks = _unique_string_list(raw.get("risks", []), "risks", allow_empty=True)
    return {
        "title": _required_text(raw, "title", 80),
        "hypothesis": _required_text(raw, "hypothesis", 600),
        "rationale": _required_text(raw, "rationale", 1000),
        "factor_names": factor_names,
        "strategy_ids": selected_strategies,
        "expected_outcome": _required_text(raw, "expected_outcome", 600),
        "risks": [value[:300] for value in risks[:5]],
    }


def build_report_messages(result: Mapping[str, Any], context: Mapping[str, Any] | None) -> list[dict[str, str]]:
    evidence = {
        "research_context": dict(context or {}),
        "data_as_of": result.get("data_as_of"),
        "summary": result.get("summary"),
        "request_summary": result.get("request_summary"),
        "selected_factors": [
            item for item in result.get("factors", []) if item.get("selected")
        ][:MAX_ASSISTANT_FACTORS],
        "candidates": result.get("candidates", [])[:8],
        "regimes": result.get("regimes", []),
        "folds": result.get("folds", [])[:40],
    }
    system = """你是A股量化研究审阅助手。请只依据用户提供的真实挖掘证据生成简洁中文报告。

要求:
1. 先给结论: 支持假设、否定假设或证据不足。
2. 引用样本外 Sharpe、最大回撤、正收益折比例、交易数、有效折和市场环境差异; 缺失数据写明缺失, 禁止补造。
3. 明确区分 exploratory 与已验证结果, 发布门槛不等于未来有效。
4. 列出主要风险和下一步; 如果候选未达 gate, 明确建议不发布。
5. 不提供个股推荐、收益承诺或实盘指令。输出纯文本, 最多 700 字。
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, allow_nan=False)},
    ]


def _required_text(raw: Mapping[str, Any], field: str, max_length: int) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"研究计划缺少 {field}")
    value = value.strip()
    if len(value) > max_length:
        raise ValueError(f"研究计划字段 {field} 不能超过 {max_length} 个字符")
    return value


def _unique_string_list(value: Any, field: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"研究计划字段 {field} 必须是字符串数组")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"研究计划字段 {field} 包含非法值")
    cleaned = [item.strip() for item in value]
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"研究计划字段 {field} 不能重复")
    return cleaned
