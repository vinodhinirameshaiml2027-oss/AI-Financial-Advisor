"""
Financial goal planning calculations: months required to reach a target,
estimated completion date, and required monthly contribution for a
target date.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional

from dateutil.relativedelta import relativedelta

from src.utils import safe_float, safe_divide


@dataclass
class GoalPlanResult:
    remaining_amount: float
    months_required: Optional[int]
    estimated_completion_date: Optional[date]
    is_already_achieved: bool
    is_unreachable: bool
    message: str


def plan_goal(
    target_amount: float,
    current_amount: float,
    monthly_contribution: float,
    start_date: Optional[date] = None,
) -> GoalPlanResult:
    """Calculate months required and completion date for a savings goal."""
    target_amount = safe_float(target_amount)
    current_amount = safe_float(current_amount)
    monthly_contribution = safe_float(monthly_contribution)
    start_date = start_date or date.today()

    remaining = max(0.0, target_amount - current_amount)

    if current_amount >= target_amount and target_amount > 0:
        return GoalPlanResult(
            remaining_amount=0.0,
            months_required=0,
            estimated_completion_date=start_date,
            is_already_achieved=True,
            is_unreachable=False,
            message="You have already reached this financial goal.",
        )

    if monthly_contribution <= 0:
        return GoalPlanResult(
            remaining_amount=round(remaining, 2),
            months_required=None,
            estimated_completion_date=None,
            is_already_achieved=False,
            is_unreachable=True,
            message="A positive monthly contribution is required to reach this goal.",
        )

    months_required = math.ceil(safe_divide(remaining, monthly_contribution))
    completion_date = start_date + relativedelta(months=months_required)

    return GoalPlanResult(
        remaining_amount=round(remaining, 2),
        months_required=int(months_required),
        estimated_completion_date=completion_date,
        is_already_achieved=False,
        is_unreachable=False,
        message=f"At your current contribution rate, you will reach this goal in approximately {months_required} month(s).",
    )


def required_monthly_contribution(
    target_amount: float,
    current_amount: float,
    target_date: date,
    start_date: Optional[date] = None,
) -> float:
    """Calculate the monthly contribution required to hit a target amount by a target date."""
    target_amount = safe_float(target_amount)
    current_amount = safe_float(current_amount)
    start_date = start_date or date.today()

    remaining = max(0.0, target_amount - current_amount)

    months_available = (target_date.year - start_date.year) * 12 + (target_date.month - start_date.month)
    months_available = max(1, months_available)

    return round(safe_divide(remaining, months_available), 2)


def goal_progress_percent(target_amount: float, current_amount: float) -> float:
    target_amount = safe_float(target_amount)
    current_amount = safe_float(current_amount)
    if target_amount <= 0:
        return 0.0
    pct = safe_divide(current_amount, target_amount) * 100
    return max(0.0, min(100.0, pct))
