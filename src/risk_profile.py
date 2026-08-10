"""
Simple, transparent risk-tolerance questionnaire and scoring.

This module intentionally does NOT recommend specific stocks, crypto,
securities, or promise investment returns. It only classifies a general
risk appetite (LOW / MODERATE / HIGH) for educational purposes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


# Each question maps an answer option to a numeric score contribution.
RISK_QUESTIONS = {
    "age_range": {
        "label": "What is your age range?",
        "options": {
            "Under 25": 4,
            "25-35": 4,
            "36-45": 3,
            "46-55": 2,
            "56 and above": 1,
        },
    },
    "investment_horizon": {
        "label": "What is your investment horizon?",
        "options": {
            "Less than 1 year": 1,
            "1-3 years": 2,
            "3-7 years": 3,
            "7+ years": 4,
        },
    },
    "income_stability": {
        "label": "How stable is your income?",
        "options": {
            "Very stable (permanent job/pension)": 4,
            "Stable (salaried)": 3,
            "Variable (freelance/business)": 2,
            "Uncertain / irregular": 1,
        },
    },
    "emergency_savings": {
        "label": "Do you have an emergency fund (3-6 months of expenses)?",
        "options": {
            "Yes, more than 6 months covered": 4,
            "Yes, 3-6 months covered": 3,
            "Some, but less than 3 months": 2,
            "No emergency fund": 1,
        },
    },
    "debt_level": {
        "label": "How would you describe your current debt level?",
        "options": {
            "No debt": 4,
            "Low (manageable EMIs)": 3,
            "Moderate": 2,
            "High": 1,
        },
    },
    "loss_reaction": {
        "label": "If your investments dropped 20% in a few months, you would:",
        "options": {
            "Invest more, seeing it as an opportunity": 4,
            "Hold and wait it out": 3,
            "Feel anxious but do nothing": 2,
            "Sell to prevent further loss": 1,
        },
    },
    "goal_type": {
        "label": "What best describes your primary financial goal?",
        "options": {
            "Long-term wealth building": 4,
            "Buying a large asset (home, car) in a few years": 3,
            "Medium-term goal (education, travel)": 2,
            "Capital protection / short-term need": 1,
        },
    },
}


@dataclass
class RiskResult:
    total_score: int
    max_score: int
    percentage: float
    category: str
    explanation: str


def score_answers(answers: Dict[str, str]) -> RiskResult:
    """Score questionnaire answers and classify overall risk tolerance."""
    total = 0
    max_total = 0
    reasons: List[str] = []

    for key, question in RISK_QUESTIONS.items():
        max_total += max(question["options"].values())
        answer = answers.get(key)
        if answer and answer in question["options"]:
            total += question["options"][answer]

    percentage = (total / max_total * 100) if max_total > 0 else 0.0

    if percentage >= 70:
        category = "HIGH"
        reasons.append("Your answers indicate a longer horizon, stable finances, and comfort with market fluctuations.")
    elif percentage >= 40:
        category = "MODERATE"
        reasons.append("Your answers reflect a balanced approach — some comfort with risk, but with an eye on stability.")
    else:
        category = "LOW"
        reasons.append("Your answers indicate a preference for capital protection and lower tolerance for short-term losses.")

    explanation = (
        f"Based on your responses (score: {total}/{max_total}, {round(percentage,1)}%), "
        f"you were classified as {category} risk tolerance. {reasons[0]}"
    )

    return RiskResult(
        total_score=total,
        max_score=max_total,
        percentage=round(percentage, 1),
        category=category,
        explanation=explanation,
    )
