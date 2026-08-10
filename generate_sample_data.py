"""
One-off script used to generate data/sample_transactions.csv.
Not required to run the app (the CSV is already included), but kept
in the project so the data generation process is fully transparent
and reproducible.
"""
import numpy as np
import pandas as pd
from datetime import date

np.random.seed(42)

START = date(2024, 8, 1)
MONTHS = 18  # 1.5 years of synthetic history

EXPENSE_CATEGORIES = {
    "Rent": (15000, 15000, 0),        # fixed
    "Food": (7000, 1500, 1),          # variable, mean, std
    "Transport": (3500, 900, 1),
    "Utilities": (2500, 500, 1),
    "Entertainment": (2800, 1200, 1),
    "Shopping": (4200, 2200, 1),
    "Healthcare": (1500, 1300, 1),
    "Education": (2000, 500, 1),
    "Insurance": (1800, 1800, 0),
    "Miscellaneous": (1200, 700, 1),
}

INCOME_CATEGORIES = {
    "Salary": (60000, 1500),
    "Freelance": (4000, 3500),
}

rows = []
month_starts = pd.date_range(START.isoformat(), periods=MONTHS, freq="MS")

for m_idx, month_start in enumerate(month_starts):
    days_in_month = (month_start + pd.offsets.MonthEnd(0)).day
    # slight upward drift over time to make data interesting for ML
    drift = 1.0 + (m_idx * 0.006)

    # Income - salary on the 1st, occasional freelance mid-month
    salary_mean, salary_std = INCOME_CATEGORIES["Salary"]
    salary_amt = max(0, np.random.normal(salary_mean, salary_std) * drift)
    rows.append([month_start + pd.Timedelta(days=0), "Salary", round(salary_amt, 2), "Income"])

    if np.random.rand() < 0.55:
        free_mean, free_std = INCOME_CATEGORIES["Freelance"]
        free_amt = max(0, np.random.normal(free_mean, free_std))
        free_day = np.random.randint(10, min(25, days_in_month))
        rows.append([month_start + pd.Timedelta(days=free_day), "Freelance", round(free_amt, 2), "Income"])

    # Expenses
    for cat, (mean, std, is_variable) in EXPENSE_CATEGORIES.items():
        if is_variable:
            n_transactions = np.random.randint(1, 5)
            for _ in range(n_transactions):
                amt = max(50, np.random.normal(mean / n_transactions, std / max(n_transactions, 1)) * drift)
                day = np.random.randint(1, days_in_month)
                rows.append([month_start + pd.Timedelta(days=day), cat, round(amt, 2), "Expense"])
        else:
            amt = max(0, mean * drift + np.random.normal(0, mean * 0.02))
            day = np.random.randint(1, 5)
            rows.append([month_start + pd.Timedelta(days=day), cat, round(amt, 2), "Expense"])

df = pd.DataFrame(rows, columns=["Date", "Category", "Amount", "Type"])
df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
df = df.sort_values("Date").reset_index(drop=True)
df.to_csv("data/sample_transactions.csv", index=False)
print(f"Generated {len(df)} transactions across {MONTHS} months")
print(df.head(10))
