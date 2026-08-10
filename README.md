# 💰 AI Financial Advisor

**End-to-End Data Science & Machine Learning Application**

A fully local, offline personal-finance analysis app built with Streamlit,
Pandas, and Scikit-learn. Enter your financial details (or upload your own
transaction CSV) and get financial health scoring, expense analysis, ML-based
expense prediction, savings forecasting, goal planning, risk profiling, and
personalized recommendations — all running on your own machine, with no
external APIs, no API keys, and no internet connection required.

---

## Table of Contents

1. [Features](#features)
2. [Tech Stack](#tech-stack)
3. [Project Architecture](#project-architecture)
4. [Folder Structure](#folder-structure)
5. [Installation (Windows / VS Code)](#installation-windows--vs-code)
6. [How to Run](#how-to-run)
7. [How to Upload a CSV](#how-to-upload-a-csv)
8. [Expected CSV Format](#expected-csv-format)
9. [Machine Learning Explanation](#machine-learning-explanation)
10. [Forecasting Explanation](#forecasting-explanation)
11. [Risk Profiling Explanation](#risk-profiling-explanation)
12. [Financial Health Score Explanation](#financial-health-score-explanation)
13. [Troubleshooting](#troubleshooting)
14. [Future Improvements](#future-improvements)
15. [Disclaimer](#disclaimer)

---

## Features

- 🏠 **Dashboard** — income, expenses, savings, savings rate, debt, and health score at a glance
- 👤 **Financial Profile** — input form with validation for income, expenses, goals, and risk tolerance
- 📤 **CSV Upload** — upload your own transaction history with automatic cleaning and error handling
- 📊 **Expense Analysis** — category breakdown, monthly trends, essential vs discretionary spending
- 🤖 **ML Expense Prediction** — trains and compares Linear Regression, Random Forest, and Gradient
  Boosting models on your own data, with a safe statistical fallback for small datasets
- 📈 **Savings Forecast** — 3/6/12/24/36-month projections with an optional historical growth trend
- 🎯 **Goal Planner** — months-to-goal calculator, completion date estimate, and reverse "required
  monthly contribution" calculator
- ⚖️ **Risk Profiling** — a 7-question questionnaire classifying you as LOW / MODERATE / HIGH risk
  tolerance (no specific investment recommendations)
- 💡 **AI Recommendations** — personalized, rule-based recommendations generated from your real numbers
- 📉 **Interactive Visualizations** — Plotly charts throughout the app
- 🇮🇳 **Indian Rupee formatting** (₹ with lakh/crore grouping) throughout

---

## Tech Stack

| Layer            | Technology                                  |
|-------------------|----------------------------------------------|
| UI / App          | Streamlit                                    |
| Data processing   | Pandas, NumPy                                |
| Machine learning  | Scikit-learn (Linear Regression, Random Forest, Gradient Boosting) |
| Visualization     | Plotly, Matplotlib                           |
| Storage           | Local CSV files (no database required)       |

No paid APIs. No OpenAI/Gemini/Claude API keys. No cloud deployment. 100% local.

---

## Project Architecture

The app follows a simple layered architecture:

```
Streamlit UI (app.py)
        │
        ▼
src/ (business logic — pure Python, no Streamlit imports)
   ├── data_loader.py          → CSV loading, validation, cleaning
   ├── preprocessing.py        → monthly aggregation, feature engineering
   ├── financial_analysis.py   → ratios, financial health score, insights
   ├── ml_models.py            → expense prediction (real scikit-learn models)
   ├── forecasting.py          → multi-month savings forecast
   ├── goal_planner.py         → goal math (months required, completion date)
   ├── risk_profile.py         → risk questionnaire scoring
   ├── recommendation_engine.py→ rule-based personalized recommendations
   └── utils.py                → currency formatting & safe numeric helpers
```

Keeping business logic separate from the UI means every calculation can be
(and was) independently tested outside of Streamlit.

---

## Folder Structure

```
AI-Financial-Advisor/
│
├── app.py                     # Main Streamlit application (entry point)
├── generate_sample_data.py    # Script used to generate the sample dataset
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
│   └── config.toml            # Theme + telemetry settings
│
├── data/
│   └── sample_transactions.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── financial_analysis.py
│   ├── ml_models.py
│   ├── forecasting.py
│   ├── goal_planner.py
│   ├── risk_profile.py
│   ├── recommendation_engine.py
│   └── utils.py
│
├── models/
│   └── .gitkeep                # ML models are trained fresh each session
│
└── assets/
    └── README.md
```

---

## Installation (Windows / VS Code)

### Prerequisites
- [Python 3.11+](https://www.python.org/downloads/) installed and available on PATH
- [VS Code](https://code.visualstudio.com/) with the Python extension

### Step-by-step (Windows PowerShell)

1. **Extract** the `AI-Financial-Advisor.zip` file to a folder of your choice.
2. **Open the folder in VS Code**: `File → Open Folder...` → select `AI-Financial-Advisor`.
3. **Open a terminal in VS Code**: `Terminal → New Terminal` (make sure it's PowerShell).
4. **Create a virtual environment**:
   ```powershell
   python -m venv venv
   ```
5. **Activate the virtual environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
   > If you get an execution policy error, run this once, then retry step 5:
   > ```powershell
   > Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   > ```
6. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```
7. **Run the app** (see [How to Run](#how-to-run) below).

### Step-by-step (Windows Command Prompt)

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### macOS / Linux (for reference)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

With the virtual environment activated and dependencies installed, run:

```powershell
streamlit run app.py
```

This launches ONE local web server. Streamlit will automatically open your
default browser to:

```
http://localhost:8501
```

If it doesn't open automatically, copy that URL into your browser manually.

Press `Ctrl+C` in the terminal to stop the app.

The app loads with the bundled **sample dataset** by default, so you can
explore every feature immediately without uploading anything.

---

## How to Upload a CSV

1. Go to the **📤 Upload Transactions** page in the sidebar.
2. Click **Browse files** and select your `.csv` file.
3. The app will validate and clean the data automatically, showing any
   warnings (e.g. rows removed due to bad dates or amounts).
4. Once loaded, every other page (Expense Analysis, ML Predictions, Savings
   Forecast, etc.) automatically uses your uploaded data.
5. Click **🔄 Reset to Sample Data** at any time to go back to the bundled
   sample dataset.

You can also click **⬇️ Download Sample CSV** to see exactly what format is
expected, or use it as a starting template.

---

## Expected CSV Format

Your CSV must contain exactly these four columns (header names must match):

| Column   | Type                        | Example        |
|----------|-----------------------------|----------------|
| Date     | Date (any standard format)  | `2026-01-01`   |
| Category | Text                        | `Food`         |
| Amount   | Positive number             | `500`          |
| Type     | `Income` or `Expense`       | `Expense`      |

Example:

```csv
Date,Category,Amount,Type
2026-01-01,Food,500,Expense
2026-01-02,Transport,200,Expense
2026-01-03,Salary,60000,Income
```

The app automatically handles: missing columns (clear error shown), invalid
dates (row removed, warning shown), non-numeric amounts (row removed,
warning shown), negative amounts (converted to positive), zero amounts (row
removed), missing categories (labeled "Uncategorized"), and duplicate rows
(removed).

---

## Machine Learning Explanation

The **ML Predictions** page trains real scikit-learn regression models on
your monthly expense history to predict next month's expense:

- **Features used**: a time index, 1-month and 2-month lagged expenses,
  1-month lagged income, and a 3-month rolling average of expenses.
- **Models compared**: Linear Regression, Random Forest Regressor, and
  Gradient Boosting Regressor.
- **Evaluation**: models are trained on a train/test split of your monthly
  history and compared using MAE (Mean Absolute Error), RMSE (Root Mean
  Squared Error), and R² where the test set is large enough. The
  lowest-MAE model is selected and then refit on all available data to
  produce the final prediction.
- **Fallback**: if there isn't enough monthly history for meaningful ML
  training (fewer than ~6 usable months), the app clearly labels this and
  falls back to a transparent weighted moving average of recent months
  instead of faking a model result.

---

## Forecasting Explanation

The **Savings Forecast** page projects your savings forward month-by-month
using:

```
Savings(t) = Savings(t-1) + Monthly Income - Monthly Expenses(t)
```

Optionally, it applies a modest expense growth trend estimated from the
median month-over-month change in your historical expenses (clamped to
±3% per month to avoid unrealistic runaway projections). All forecasts are
clearly labeled as estimates based on simplified assumptions.

---

## Risk Profiling Explanation

The **Risk Profile** page asks 7 questions covering age range, investment
horizon, income stability, emergency savings, debt level, reaction to
investment losses, and goal type. Each answer contributes points toward a
total score, which is converted to a percentage and classified as:

- **LOW** (< 40%)
- **MODERATE** (40–70%)
- **HIGH** (≥ 70%)

This tool intentionally does **not** recommend specific stocks, funds,
cryptocurrencies, or securities, and does not promise any investment
returns — it only provides a general, educational risk-tolerance
classification with a plain-language explanation.

---

## Financial Health Score Explanation

The **Financial Health Score** (0–100) is a transparent, educational
heuristic combining five weighted components:

| Component            | Weight | What it measures                                  |
|-----------------------|--------|-----------------------------------------------------|
| Savings Rate          | 30 pts | Savings as a % of income (20%+ = full marks)        |
| Debt-to-Income        | 25 pts | Monthly debt as a % of income (lower is better)     |
| Emergency Fund        | 20 pts | Current savings ÷ monthly expenses (6+ months = full marks) |
| Expense-to-Income     | 15 pts | Expenses as a % of income (≤50% = full marks)       |
| Goal Progress         | 10 pts | Current savings vs. target goal amount              |

This score is **not** a scientifically validated financial metric — it's a
simplified, explainable model built for this educational project. Every
component is shown individually so you can see exactly how the score is
calculated.

---

## Troubleshooting

**`streamlit: command not found` or similar**
Make sure your virtual environment is activated (you should see `(venv)` in
your terminal prompt) and that `pip install -r requirements.txt` completed
without errors.

**PowerShell blocks the activation script**
Run this once in PowerShell (as your normal user, not admin):
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Port 8501 already in use**
Run Streamlit on a different port:
```powershell
streamlit run app.py --server.port 8502
```

**CSV upload shows an error**
Check that your file has the exact columns `Date, Category, Amount, Type`
(case-sensitive) and at least one valid row. Use the **Download Sample CSV**
button as a template.

**Charts look empty**
This usually means the active dataset has no `Income` or `Expense` rows of
the type the chart needs — check the **Upload Transactions** page to confirm
your data loaded correctly.

**App shows "Something went wrong while rendering this page"**
The app is designed to catch errors gracefully and show this message
instead of crashing. Expand "Technical details" for more information, and
double-check your Financial Profile and uploaded data for unusual values.

---

## Future Improvements

Ideas for extending this portfolio project further:

- Persist user profiles and transaction history across sessions using a
  local SQLite database
- Add more sophisticated time-series forecasting (e.g. ARIMA/Prophet) with
  confidence intervals
- Add budget-vs-actual tracking with monthly alerts
- Support multiple currencies with live-independent conversion tables
- Add unit tests with `pytest` and a CI pipeline
- Export dashboard summaries to PDF

---

## Disclaimer

This application is provided for **educational and portfolio purposes
only**. It does not constitute professional financial, investment, tax, or
legal advice. All scores, predictions, and forecasts are simplified,
transparent heuristics and statistical estimates intended to demonstrate
data science and machine learning techniques — they are not guarantees of
future performance. Always consult a qualified, licensed financial advisor
before making real financial decisions.
