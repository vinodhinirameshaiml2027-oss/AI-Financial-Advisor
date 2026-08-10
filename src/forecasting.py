"""
Savings forecasting module. Projects future savings based on current
savings, monthly income, monthly expenses, and monthly investment
contributions, optionally applying a modest expense growth trend
estimated from historical data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.utils import safe_float


@dataclass
class ForecastPoint:
    month_index: int
    label: str
    projected_savings: float
    projected_income: float
    projected_expense: float


def estimate_monthly_expense_growth(monthly_df: pd.DataFrame) -> float:
    """
    Estimate an average month-over-month expense growth rate from history.
    Returns a fractional growth rate (e.g. 0.01 = 1% per month), clamped to
    a sane range to avoid runaway forecasts.
    """
    if monthly_df is None or len(monthly_df) < 3:
        return 0.0

    expenses = monthly_df["Expense"].values
    expenses = expenses[expenses > 0]
    if len(expenses) < 3:
        return 0.0

    growth_rates = np.diff(expenses) / expenses[:-1]
    growth_rates = growth_rates[np.isfinite(growth_rates)]
    if len(growth_rates) == 0:
        return 0.0

    avg_growth = float(np.median(growth_rates))
    return float(np.clip(avg_growth, -0.03, 0.03))


def forecast_savings(
    current_savings: float,
    monthly_income: float,
    monthly_expenses: float,
    monthly_investment: float,
    months: int,
    expense_growth_rate: float = 0.0,
    start_label_index: int = 1,
) -> List[ForecastPoint]:
    """
    Project savings month-by-month.
    Savings_t = Savings_(t-1) + Income - Expenses_t - Investment
    (Investment is treated as money set aside, still counted toward savings/net worth.)
    """
    current_savings = safe_float(current_savings)
    monthly_income = safe_float(monthly_income)
    monthly_expenses = safe_float(monthly_expenses)
    monthly_investment = safe_float(monthly_investment)
    months = max(1, int(months))

    points: List[ForecastPoint] = []
    running_savings = current_savings
    running_expense = monthly_expenses

    for m in range(1, months + 1):
        if m > 1:
            running_expense = running_expense * (1 + expense_growth_rate)

        net_change = monthly_income - running_expense
        running_savings += net_change
        # monthly_investment is assumed to be drawn from income already reflected in expenses
        # budget; here we treat it as additional deliberate saving on top of net cash flow.
        running_savings += 0  # investment already implicitly part of savings; kept explicit for clarity

        points.append(
            ForecastPoint(
                month_index=m,
                label=f"Month {m}",
                projected_savings=round(running_savings, 2),
                projected_income=round(monthly_income, 2),
                projected_expense=round(running_expense, 2),
            )
        )

    return points


def forecast_to_dataframe(points: List[ForecastPoint]) -> pd.DataFrame:
    if not points:
        return pd.DataFrame(columns=["Month", "ProjectedSavings", "ProjectedIncome", "ProjectedExpense"])
    return pd.DataFrame(
        [
            {
                "Month": p.label,
                "ProjectedSavings": p.projected_savings,
                "ProjectedIncome": p.projected_income,
                "ProjectedExpense": p.projected_expense,
            }
            for p in points
        ]
    )
