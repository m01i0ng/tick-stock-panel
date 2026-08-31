"""Lightweight mining date availability checks shared by API and workers."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from app.backtest.mining import (
    nested_fold_count,
    required_holdout_bars,
    required_outer_folds,
    required_trading_bars,
    validation_config_for_profile,
)
from app.services.regime_builder import load_regime_history
from app.tickflow.repository import enriched_dirname


class MiningPreflightError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class MiningAvailability:
    asset_type: str
    budget_profile: str
    trading_bars: int
    required_bars: int
    outer_folds: int
    required_outer_folds: int
    eligible: bool
    available_start: date | None
    available_end: date | None
    effective_start: date | None
    effective_end: date | None
    suggested_start: date | None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value.isoformat() if isinstance(value, date) else value
            for key, value in asdict(self).items()
        }


def enriched_partition_dates(
    data_dir: Path,
    asset_type: str,
    start: date | None = None,
    end: date | None = None,
) -> list[date]:
    root = data_dir / enriched_dirname(asset_type)
    values: set[date] = set()
    for partition in root.glob("date=*"):
        try:
            value = date.fromisoformat(partition.name.removeprefix("date="))
        except ValueError:
            continue
        if start is not None and value < start:
            continue
        if end is not None and value > end:
            continue
        if (partition / "part.parquet").is_file():
            values.add(value)
    return sorted(values)


def mining_availability(
    data_dir: Path,
    *,
    asset_type: str,
    budget_profile: str,
    start: date | None = None,
    end: date | None = None,
) -> MiningAvailability:
    if asset_type not in {"stock", "etf"}:
        raise ValueError(f"unsupported mining asset type: {asset_type}")
    if start is not None and end is not None and start > end:
        raise ValueError("mining start must not be after end")

    config = validation_config_for_profile(budget_profile)
    required_folds = required_outer_folds(budget_profile)
    required_bars = required_trading_bars(config, required_folds)
    all_dates = enriched_partition_dates(data_dir, asset_type)
    scoped = [
        value
        for value in all_dates
        if (start is None or value >= start) and (end is None or value <= end)
    ]
    dates_through_end = [
        value for value in all_dates if end is None or value <= end
    ]
    suggested_start = (
        dates_through_end[-required_bars]
        if len(dates_through_end) >= required_bars
        else None
    )
    trading_bars = len(scoped)
    return MiningAvailability(
        asset_type=asset_type,
        budget_profile=budget_profile,
        trading_bars=trading_bars,
        required_bars=required_bars,
        outer_folds=nested_fold_count(trading_bars, config),
        required_outer_folds=required_folds,
        eligible=trading_bars >= required_bars,
        available_start=all_dates[0] if all_dates else None,
        available_end=all_dates[-1] if all_dates else None,
        effective_start=scoped[0] if scoped else None,
        effective_end=scoped[-1] if scoped else None,
        suggested_start=suggested_start,
    )


def require_mining_availability(
    data_dir: Path,
    *,
    asset_type: str,
    budget_profile: str,
    start: date | None = None,
    end: date | None = None,
) -> MiningAvailability:
    availability = mining_availability(
        data_dir,
        asset_type=asset_type,
        budget_profile=budget_profile,
        start=start,
        end=end,
    )
    if availability.eligible:
        regime = load_regime_history(data_dir)
        if regime.is_empty() or "date" not in regime.columns:
            raise MiningPreflightError(
                "regime_unavailable",
                "市场环境数据不可用, 请先在数据页完成市场环境计算后再研究",
            )
        regime_dates = {str(value)[:10] for value in regime["date"].to_list()}
        covered_dates = enriched_partition_dates(
            data_dir,
            asset_type,
            start,
            end,
        )
        missing = [
            value.isoformat()
            for value in covered_dates[:-1]
            if value.isoformat() not in regime_dates
        ]
        if missing:
            raise MiningPreflightError(
                "regime_incomplete",
                "市场环境数据覆盖不完整: "
                f"T-1 对齐缺少 {len(missing)} 个交易日, 首个为 {missing[0]}, "
                "请先补算对应区间",
            )
        return availability

    if availability.effective_start is None:
        effective_range = "contains no enriched data"
    else:
        effective_range = (
            f"{availability.effective_start.isoformat()} to "
            f"{availability.effective_end.isoformat()}"
        )
    fold_label = "outer fold" if availability.required_outer_folds == 1 else "outer folds"
    raise MiningPreflightError(
        "enriched_insufficient",
        f"{budget_profile} mining requires at least {availability.required_bars} "
        f"enriched trading bars for {availability.required_outer_folds} {fold_label}; "
        f"effective range {effective_range} has {availability.trading_bars}",
    )


def reserve_final_holdout(
    data_dir: Path,
    *,
    asset_type: str,
    budget_profile: str,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, Any]:
    """Split the experiment range into an adaptive window and a sealed tail holdout."""
    holdout_bars = required_holdout_bars(budget_profile)
    config = validation_config_for_profile(budget_profile)
    required_adaptive = required_trading_bars(config, required_outer_folds(budget_profile))
    dates = enriched_partition_dates(data_dir, asset_type, start, end)
    needed = required_adaptive + holdout_bars
    if len(dates) < needed:
        if not dates:
            effective_range = "contains no enriched data"
        else:
            effective_range = f"{dates[0].isoformat()} to {dates[-1].isoformat()}"
        raise MiningPreflightError(
            "enriched_insufficient",
            f"{budget_profile} autoresearch requires at least {needed} "
            f"enriched trading bars ({required_adaptive} adaptive + "
            f"{holdout_bars} final holdout); effective range {effective_range} "
            f"has {len(dates)}",
        )
    adaptive_dates = dates[:-holdout_bars]
    holdout_dates = dates[-holdout_bars:]
    require_mining_availability(
        data_dir,
        asset_type=asset_type,
        budget_profile=budget_profile,
        start=start,
        end=adaptive_dates[-1],
    )
    require_mining_availability(
        data_dir,
        asset_type=asset_type,
        budget_profile=budget_profile,
        start=start,
        end=holdout_dates[-1],
    )
    return {
        "adaptive_end": adaptive_dates[-1].isoformat(),
        "start": holdout_dates[0].isoformat(),
        "end": holdout_dates[-1].isoformat(),
        "bars": holdout_bars,
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
