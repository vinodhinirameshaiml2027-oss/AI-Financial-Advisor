"""
Transforms cleaned transaction-level data into monthly aggregates and
feature tables that the analysis, ML, and forecasting modules consume.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_month_period(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'Month' column (Period[M]) derived from Date."""
    df = df.copy()
    df["Month"] = df["Date"].dt.to_period("M")
    return df


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a month-by-month summary with total income, total expense,
    net savings, and savings rate.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Month", "Income", "Expense", "Savings", "SavingsRate"])

    df = add_month_period(df)

    income = df[df["Type"] == "Income"].groupby("Month")["Amount"].sum()
    expense = df[df["Type"] == "Expense"].groupby("Month")["Amount"].sum()

    all_months = sorted(set(income.index) | set(expense.index))
    summary = pd.DataFrame(index=all_months)
    summary["Income"] = income.reindex(all_months, fill_value=0.0)
    summary["Expense"] = expense.reindex(all_months, fill_value=0.0)
    summary["Savings"] = summary["Income"] - summary["Expense"]
    summary["SavingsRate"] = np.where(
        summary["Income"] > 0, (summary["Savings"] / summary["Income"]) * 100, 0.0
    )
    summary = summary.reset_index().rename(columns={"index": "Month"})
    summary["Month"] = summary["Month"].astype(str)
    return summary


def category_summary(df: pd.DataFrame, txn_type: str = "Expense") -> pd.DataFrame:
    """Aggregate total amount and transaction count per category for a given type."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Category", "Total", "Count", "Average"])

    subset = df[df["Type"] == txn_type]
    if subset.empty:
        return pd.DataFrame(columns=["Category", "Total", "Count", "Average"])

    grouped = subset.groupby("Category")["Amount"].agg(Total="sum", Count="count", Average="mean")
    grouped = grouped.reset_index().sort_values("Total", ascending=False).reset_index(drop=True)
    return grouped


def monthly_category_matrix(df: pd.DataFrame, txn_type: str = "Expense") -> pd.DataFrame:
    """Pivot table: rows = months, columns = categories, values = summed amount."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = add_month_period(df)
    subset = df[df["Type"] == txn_type]
    if subset.empty:
        return pd.DataFrame()

    pivot = subset.pivot_table(
        index="Month", columns="Category", values="Amount", aggfunc="sum", fill_value=0.0
    )
    pivot.index = pivot.index.astype(str)
    return pivot.sort_index()


# Categories commonly considered "essential" vs "discretionary" for a simple
# heuristic breakdown. This is an approximation for educational purposes.
ESSENTIAL_CATEGORIES = {
    "rent", "utilities", "food", "groceries", "insurance", "healthcare",
    "transport", "education", "emi", "loan", "bills", "medical",
}


def essential_vs_discretionary(df: pd.DataFrame) -> pd.DataFrame:
    """Classify expense categories into Essential vs Discretionary buckets."""
    cat_summary = category_summary(df, "Expense")
    if cat_summary.empty:
        return pd.DataFrame(columns=["Bucket", "Total"])

    def classify(cat: str) -> str:
        return "Essential" if cat.strip().lower() in ESSENTIAL_CATEGORIES else "Discretionary"

    cat_summary["Bucket"] = cat_summary["Category"].apply(classify)
    bucket_summary = cat_summary.groupby("Bucket")["Total"].sum().reset_index()
    return bucket_summary


def build_ml_feature_table(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a supervised-learning feature table for expense prediction.
    Features use lagged expense values and a time index; target is the
    current month's expense.
    """
    if monthly_df is None or monthly_df.empty or len(monthly_df) < 2:
        return pd.DataFrame()

    data = monthly_df.copy().reset_index(drop=True)
    data["time_idx"] = np.arange(len(data))
    data["expense_lag1"] = data["Expense"].shift(1)
    data["expense_lag2"] = data["Expense"].shift(2)
    data["income_lag1"] = data["Income"].shift(1)
    data["rolling_mean_3"] = data["Expense"].shift(1).rolling(window=3, min_periods=1).mean()

    data = data.dropna().reset_index(drop=True)
    return data
