"""
Core financial analysis: ratios, financial health score, and
human-readable insights derived from the user's actual numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.utils import safe_divide, safe_float, clamp


@dataclass
class FinancialProfile:
    monthly_income: float = 0.0
    monthly_fixed_expenses: float = 0.0
    monthly_variable_expenses: float = 0.0
    current_savings: float = 0.0
    monthly_investment: float = 0.0
    monthly_debt: float = 0.0
    goal_name: str = ""
    target_amount: float = 0.0
    risk_tolerance: str = "Moderate"
    investment_horizon_years: float = 5.0

    @property
    def total_monthly_expenses(self) -> float:
        return safe_float(self.monthly_fixed_expenses) + safe_float(self.monthly_variable_expenses)

    @property
    def monthly_savings(self) -> float:
        return safe_float(self.monthly_income) - self.total_monthly_expenses - safe_float(self.monthly_debt)

    @property
    def savings_rate(self) -> float:
        return clamp(safe_divide(self.monthly_savings, self.monthly_income) * 100, -1000, 100)


@dataclass
class HealthScoreBreakdown:
    total_score: float
    components: Dict[str, float] = field(default_factory=dict)
    max_components: Dict[str, float] = field(default_factory=dict)


def compute_financial_health_score(profile: FinancialProfile, emergency_fund_months: float = None) -> HealthScoreBreakdown:
    """
    Compute an educational 0-100 financial health score from five weighted
    components. This is a transparent heuristic, not a validated financial
    model, and is clearly labeled as such in the UI.
    """
    income = safe_float(profile.monthly_income)
    expenses = profile.total_monthly_expenses
    debt = safe_float(profile.monthly_debt)
    savings = profile.monthly_savings
    current_savings = safe_float(profile.current_savings)

    components: Dict[str, float] = {}
    max_components = {
        "Savings Rate": 30.0,
        "Debt-to-Income": 25.0,
        "Emergency Fund": 20.0,
        "Expense-to-Income": 15.0,
        "Goal Progress": 10.0,
    }

    # 1. Savings rate (0-30): 20%+ savings rate = full marks
    savings_rate = safe_divide(savings, income) * 100
    components["Savings Rate"] = clamp(safe_divide(savings_rate, 20) * 30, 0, 30)

    # 2. Debt-to-income ratio (0-25): lower is better, 0% debt = full marks, 40%+ = 0
    dti = safe_divide(debt, income) * 100
    components["Debt-to-Income"] = clamp(25 - safe_divide(dti, 40) * 25, 0, 25)

    # 3. Emergency fund coverage (0-20): 6 months of expenses = full marks
    if emergency_fund_months is None:
        emergency_fund_months = safe_divide(current_savings, expenses) if expenses > 0 else 0
    components["Emergency Fund"] = clamp(safe_divide(emergency_fund_months, 6) * 20, 0, 20)

    # 4. Expense-to-income ratio (0-15): lower is better, 50% or less = full marks
    eti = safe_divide(expenses, income) * 100
    components["Expense-to-Income"] = clamp(15 - safe_divide(max(eti - 50, 0), 50) * 15, 0, 15)

    # 5. Goal progress (0-10): based on current savings vs target amount
    if profile.target_amount and profile.target_amount > 0:
        progress = safe_divide(current_savings, profile.target_amount) * 100
        components["Goal Progress"] = clamp(safe_divide(progress, 100) * 10, 0, 10)
    else:
        components["Goal Progress"] = 5.0  # neutral score when no goal is set

    total = sum(components.values())
    total = clamp(total, 0, 100)

    return HealthScoreBreakdown(total_score=round(total, 1), components=components, max_components=max_components)


def score_label(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 45:
        return "Fair"
    return "Needs Attention"


def generate_insights(profile: FinancialProfile, health: HealthScoreBreakdown,
                       top_category: str = None, top_category_amount: float = None,
                       predicted_next_expense: float = None) -> List[str]:
    """Generate a bulleted list of plain-language insights from real numbers."""
    insights: List[str] = []

    insights.append(f"Your current savings rate is {round(profile.savings_rate, 1)}%.")

    if top_category:
        amt_txt = f" (₹{top_category_amount:,.0f})" if top_category_amount is not None else ""
        insights.append(f"'{top_category}' is your highest expense category{amt_txt}.")

    if predicted_next_expense is not None:
        insights.append(f"Your predicted next-month expense is ₹{predicted_next_expense:,.0f}.")

    if profile.target_amount and profile.target_amount > 0:
        progress = clamp(safe_divide(profile.current_savings, profile.target_amount) * 100, 0, 100)
        insights.append(f"You are approximately {round(progress, 1)}% toward your '{profile.goal_name or 'financial'}' goal.")

    insights.append(f"Your financial health score is {health.total_score}/100 ({score_label(health.total_score)}).")

    dti = safe_divide(profile.monthly_debt, profile.monthly_income) * 100
    if dti > 0:
        insights.append(f"Your debt-to-income ratio is {round(dti, 1)}%.")

    return insights
