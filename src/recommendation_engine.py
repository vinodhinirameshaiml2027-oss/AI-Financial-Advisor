"""
Local, rule-based recommendation engine. Generates personalized financial
recommendations purely from the user's own calculated numbers — no LLM
or external API required.
"""
from __future__ import annotations

from typing import List, Optional

from src.financial_analysis import FinancialProfile, HealthScoreBreakdown
from src.utils import safe_divide


def generate_recommendations(
    profile: FinancialProfile,
    health: HealthScoreBreakdown,
    risk_category: Optional[str] = None,
    top_expense_category: Optional[str] = None,
    goal_progress_pct: Optional[float] = None,
) -> List[str]:
    """Generate a list of personalized, rule-based recommendations."""
    recs: List[str] = []

    income = profile.monthly_income
    expenses = profile.total_monthly_expenses
    savings_rate = profile.savings_rate
    debt = profile.monthly_debt
    dti = safe_divide(debt, income) * 100 if income else 0

    # Expense-to-income
    eti = safe_divide(expenses, income) * 100 if income else 0
    if eti > 75:
        recs.append(
            "Your expenses are relatively high compared with your income "
            f"({round(eti, 1)}% of income). Review discretionary categories such as "
            "shopping and entertainment to identify potential savings."
        )
    elif eti < 50 and income > 0:
        recs.append(
            f"Your expenses are well-managed at {round(eti, 1)}% of income. "
            "This gives you room to increase investments or accelerate goal progress."
        )

    # Savings rate
    if savings_rate < 10:
        recs.append(
            "Your current savings rate is low. Consider setting a monthly savings target "
            "of at least 15-20% of income and automating transfers right after payday."
        )
    elif savings_rate >= 25:
        recs.append(
            f"Great job — your savings rate of {round(savings_rate, 1)}% is strong. "
            "Consider directing a portion of this toward long-term investment goals."
        )

    # Debt-to-income
    if dti > 30:
        recs.append(
            f"Your debt-to-income ratio is elevated at {round(dti, 1)}%. "
            "Consider prioritizing debt reduction and maintaining an emergency reserve "
            "before taking on additional financial commitments."
        )
    elif dti > 0:
        recs.append(
            f"Your debt-to-income ratio of {round(dti, 1)}% is within a manageable range, "
            "but continue monitoring it as your income or obligations change."
        )

    # Emergency fund
    emergency_component = health.components.get("Emergency Fund", 0)
    emergency_max = health.max_components.get("Emergency Fund", 20)
    if emergency_component < emergency_max * 0.5:
        recs.append(
            "Your emergency fund appears insufficient to cover several months of expenses. "
            "Aim to build a reserve covering at least 3-6 months of essential expenses."
        )

    # Goal progress
    if goal_progress_pct is not None:
        if goal_progress_pct >= 75:
            recs.append(f"You are currently on track toward your financial goal ({round(goal_progress_pct,1)}% complete).")
        elif goal_progress_pct < 25:
            recs.append(
                f"You are early in your journey toward your financial goal ({round(goal_progress_pct,1)}% complete). "
                "Consider reviewing your monthly contribution to accelerate progress if possible."
            )

    # Top expense category
    if top_expense_category:
        recs.append(
            f"'{top_expense_category}' is your largest expense category. "
            "A focused review of this category could yield the biggest savings impact."
        )

    # Risk-based general guidance (no specific securities/returns)
    if risk_category == "LOW":
        recs.append(
            "Given your LOW risk tolerance, prioritize capital protection and stable, "
            "highly liquid options for your savings, alongside a solid emergency fund."
        )
    elif risk_category == "MODERATE":
        recs.append(
            "Given your MODERATE risk tolerance, a balanced approach with diversification "
            "across asset types may align with your comfort level. Consult a licensed "
            "financial advisor before making investment decisions."
        )
    elif risk_category == "HIGH":
        recs.append(
            "Given your HIGH risk tolerance and longer horizon, you may be comfortable with "
            "growth-oriented strategies, though diversification and periodic review remain "
            "important. Consult a licensed financial advisor before making investment decisions."
        )

    if not recs:
        recs.append("Your financial indicators look balanced. Continue monitoring your income, expenses, and goals regularly.")

    return recs
