"""
AI Financial Advisor
End-to-End Data Science & Machine Learning Application

Run with:  streamlit run app.py

This is a local, offline, educational personal-finance analysis tool.
It is NOT professional investment advice.
"""
from __future__ import annotations

import traceback
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import (
    get_sample_csv_bytes,
    load_sample_data,
    read_uploaded_csv,
    validate_and_clean,
)
from src.financial_analysis import (
    FinancialProfile,
    compute_financial_health_score,
    generate_insights,
    score_label,
)
from src.forecasting import estimate_monthly_expense_growth, forecast_savings, forecast_to_dataframe
from src.goal_planner import goal_progress_percent, plan_goal, required_monthly_contribution
from src.ml_models import predict_next_month_expense
from src.preprocessing import (
    category_summary,
    essential_vs_discretionary,
    monthly_category_matrix,
    monthly_summary,
)
from src.recommendation_engine import generate_recommendations
from src.risk_profile import RISK_QUESTIONS, score_answers
from src.utils import format_inr, safe_divide, safe_float

# --------------------------------------------------------------------------
# Page configuration
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#2E7D32"
ACCENT_COLOR = "#1565C0"
WARN_COLOR = "#EF6C00"
DANGER_COLOR = "#C62828"

CUSTOM_CSS = f"""
<style>
    .main {{ padding-top: 1rem; }}
    div[data-testid="stMetric"] {{
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 14px 16px 10px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ font-weight: 600; }}
    .app-header {{
        padding: 0.5rem 0 1rem 0;
    }}
    .disclaimer-box {{
        background-color: #FFF8E1;
        border-left: 4px solid {WARN_COLOR};
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-top: 1rem;
    }}
    .insight-item {{
        padding: 6px 0;
        border-bottom: 1px solid #eee;
    }}
    section[data-testid="stSidebar"] .stRadio label {{
        font-size: 0.95rem;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "profile": FinancialProfile(
            monthly_income=60000.0,
            monthly_fixed_expenses=20000.0,
            monthly_variable_expenses=15000.0,
            current_savings=150000.0,
            monthly_investment=5000.0,
            monthly_debt=5000.0,
            goal_name="Emergency Fund",
            target_amount=500000.0,
            risk_tolerance="Moderate",
            investment_horizon_years=5.0,
        ),
        "transactions_df": None,
        "data_source": "Sample Data",
        "risk_answers": {},
        "risk_result": None,
        "upload_warnings": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


def get_active_transactions() -> pd.DataFrame:
    """Return the currently active transaction dataframe (uploaded or sample)."""
    if st.session_state["transactions_df"] is not None and not st.session_state["transactions_df"].empty:
        return st.session_state["transactions_df"]
    try:
        return load_sample_data()
    except Exception:
        return pd.DataFrame(columns=["Date", "Category", "Amount", "Type"])


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
st.sidebar.markdown("## 💰 AI Financial Advisor")
st.sidebar.caption("End-to-End Data Science & ML Application")
st.sidebar.markdown("---")

PAGES = [
    "🏠 Home",
    "👤 Financial Profile",
    "📤 Upload Transactions",
    "📊 Expense Analysis",
    "🤖 ML Predictions",
    "📈 Savings Forecast",
    "🎯 Goal Planner",
    "⚖️ Risk Profile",
    "💡 AI Recommendations",
    "ℹ️ About",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption(f"Data source: **{st.session_state['data_source']}**")
st.sidebar.markdown(
    '<div class="disclaimer-box">📌 Educational project only. Not professional investment advice.</div>',
    unsafe_allow_html=True,
)


# ==========================================================================
# PAGE: HOME / DASHBOARD
# ==========================================================================
def page_home():
    st.title("🏠 Financial Dashboard")
    st.caption("Your complete financial snapshot at a glance")

    profile: FinancialProfile = st.session_state["profile"]
    health = compute_financial_health_score(profile)

    df = get_active_transactions()
    m_summary = monthly_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Monthly Income", format_inr(profile.monthly_income))
    col2.metric("Monthly Expenses", format_inr(profile.total_monthly_expenses))
    col3.metric("Monthly Savings", format_inr(profile.monthly_savings))
    col4.metric("Savings Rate", f"{round(profile.savings_rate, 1)}%")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Current Savings", format_inr(profile.current_savings))
    col6.metric("Monthly Debt/EMI", format_inr(profile.monthly_debt))
    col7.metric("Monthly Investment", format_inr(profile.monthly_investment))
    score_delta = "Excellent" if health.total_score >= 80 else None
    col8.metric("Financial Health Score", f"{health.total_score}/100", score_label(health.total_score))

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Income vs Expenses Trend")
        if not m_summary.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=m_summary["Month"], y=m_summary["Income"], name="Income", marker_color=PRIMARY_COLOR))
            fig.add_trace(go.Bar(x=m_summary["Month"], y=m_summary["Expense"], name="Expense", marker_color=DANGER_COLOR))
            fig.update_layout(barmode="group", height=380, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No transaction data available yet. Upload a CSV or use the sample dataset.")

    with right:
        st.subheader("Financial Health Breakdown")
        comp_df = pd.DataFrame(
            {"Component": list(health.components.keys()), "Score": list(health.components.values())}
        )
        fig2 = px.bar(comp_df, x="Score", y="Component", orientation="h", color="Score",
                      color_continuous_scale="Greens", range_x=[0, 30])
        fig2.update_layout(height=380, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Quick Insights")
    cat_summary = category_summary(df, "Expense")
    top_cat = cat_summary.iloc[0]["Category"] if not cat_summary.empty else None
    top_cat_amt = cat_summary.iloc[0]["Total"] if not cat_summary.empty else None

    insights = generate_insights(profile, health, top_category=top_cat, top_category_amount=top_cat_amt)
    for insight in insights:
        st.markdown(f'<div class="insight-item">• {insight}</div>', unsafe_allow_html=True)


# ==========================================================================
# PAGE: FINANCIAL PROFILE
# ==========================================================================
def page_profile():
    st.title("👤 Financial Profile")
    st.caption("Enter your financial details to power all calculations across the app.")

    profile: FinancialProfile = st.session_state["profile"]

    with st.form("profile_form"):
        st.subheader("Income & Expenses")
        c1, c2 = st.columns(2)
        with c1:
            income = st.number_input("Monthly Income (₹)", min_value=0.0, value=float(profile.monthly_income), step=1000.0)
            fixed_exp = st.number_input("Monthly Fixed Expenses (₹)", min_value=0.0, value=float(profile.monthly_fixed_expenses), step=500.0,
                                         help="Rent, EMIs, insurance premiums, subscriptions, etc.")
            variable_exp = st.number_input("Monthly Variable Expenses (₹)", min_value=0.0, value=float(profile.monthly_variable_expenses), step=500.0,
                                            help="Food, shopping, entertainment, travel, etc.")
        with c2:
            current_savings = st.number_input("Current Savings (₹)", min_value=0.0, value=float(profile.current_savings), step=1000.0)
            investment = st.number_input("Monthly Investment (₹)", min_value=0.0, value=float(profile.monthly_investment), step=500.0)
            debt = st.number_input("Monthly Debt / EMI (₹)", min_value=0.0, value=float(profile.monthly_debt), step=500.0)

        st.subheader("Financial Goal")
        c3, c4 = st.columns(2)
        with c3:
            goal_name = st.text_input("Financial Goal Name", value=profile.goal_name or "Emergency Fund")
            target_amount = st.number_input("Target Amount (₹)", min_value=0.0, value=float(profile.target_amount), step=5000.0)
        with c4:
            risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"],
                                           index=["Low", "Moderate", "High"].index(profile.risk_tolerance)
                                           if profile.risk_tolerance in ["Low", "Moderate", "High"] else 1)
            horizon = st.number_input("Investment Horizon (years)", min_value=0.0, max_value=50.0,
                                       value=float(profile.investment_horizon_years), step=0.5)

        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)

    if submitted:
        errors = []
        if income == 0:
            errors.append("Monthly income is 0 — most calculations (savings rate, health score, ratios) will show as 0 or undefined until you enter a value.")
        if fixed_exp + variable_exp > income * 3 and income > 0:
            errors.append("Your total expenses look unusually high relative to your income. Please double-check the values.")

        st.session_state["profile"] = FinancialProfile(
            monthly_income=income,
            monthly_fixed_expenses=fixed_exp,
            monthly_variable_expenses=variable_exp,
            current_savings=current_savings,
            monthly_investment=investment,
            monthly_debt=debt,
            goal_name=goal_name.strip() or "Financial Goal",
            target_amount=target_amount,
            risk_tolerance=risk_tolerance,
            investment_horizon_years=horizon,
        )

        for w in errors:
            st.warning(w)
        st.success("Profile saved successfully! Explore other tabs to see your updated analysis.")

    st.markdown("---")
    st.subheader("Current Profile Summary")
    p = st.session_state["profile"]
    summary_df = pd.DataFrame({
        "Field": ["Monthly Income", "Fixed Expenses", "Variable Expenses", "Total Expenses",
                  "Current Savings", "Monthly Investment", "Monthly Debt", "Monthly Savings",
                  "Goal", "Target Amount", "Risk Tolerance", "Investment Horizon"],
        "Value": [
            format_inr(p.monthly_income), format_inr(p.monthly_fixed_expenses), format_inr(p.monthly_variable_expenses),
            format_inr(p.total_monthly_expenses), format_inr(p.current_savings), format_inr(p.monthly_investment),
            format_inr(p.monthly_debt), format_inr(p.monthly_savings), p.goal_name, format_inr(p.target_amount),
            p.risk_tolerance, f"{p.investment_horizon_years} years",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE: UPLOAD TRANSACTIONS
# ==========================================================================
def page_upload():
    st.title("📤 Upload Transactions")
    st.caption("Upload your own CSV transaction file, or continue using the bundled sample dataset.")

    st.download_button(
        "⬇️ Download Sample CSV",
        data=get_sample_csv_bytes(),
        file_name="sample_transactions.csv",
        mime="text/csv",
        use_container_width=False,
    )

    with st.expander("📋 Expected CSV Format", expanded=False):
        st.code("Date,Category,Amount,Type\n2026-01-01,Food,500,Expense\n2026-01-02,Transport,200,Expense\n2026-01-03,Salary,60000,Income", language="text")
        st.markdown("- **Date**: any standard date format (e.g. YYYY-MM-DD)\n"
                    "- **Category**: text label (e.g. Food, Rent, Salary)\n"
                    "- **Amount**: positive number\n"
                    "- **Type**: `Income` or `Expense`")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        with st.spinner("Reading and validating file..."):
            raw_df, error = read_uploaded_csv(uploaded)

            if error:
                st.error(f"❌ {error}")
            else:
                cleaned_df, fatal_error, warnings = validate_and_clean(raw_df)
                if fatal_error:
                    st.error(f"❌ {fatal_error}")
                else:
                    st.session_state["transactions_df"] = cleaned_df
                    st.session_state["data_source"] = f"Uploaded File ({uploaded.name})"
                    st.session_state["upload_warnings"] = warnings
                    st.success(f"✅ Successfully loaded {len(cleaned_df)} valid transactions.")
                    if warnings:
                        with st.expander(f"⚠️ {len(warnings)} cleaning notice(s)", expanded=True):
                            for w in warnings:
                                st.warning(w)

    st.markdown("---")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔄 Reset to Sample Data", use_container_width=True):
            st.session_state["transactions_df"] = None
            st.session_state["data_source"] = "Sample Data"
            st.session_state["upload_warnings"] = []
            st.success("Switched back to sample data.")

    st.subheader("Active Dataset Preview")
    df = get_active_transactions()
    st.caption(f"Showing {min(len(df), 100)} of {len(df)} transactions")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)


# ==========================================================================
# PAGE: EXPENSE ANALYSIS
# ==========================================================================
def page_expense_analysis():
    st.title("📊 Expense Analysis")
    df = get_active_transactions()

    if df.empty:
        st.info("No transaction data available. Please upload a CSV or use the sample dataset.")
        return

    expense_df = df[df["Type"] == "Expense"]
    if expense_df.empty:
        st.warning("No expense transactions found in the current dataset.")
        return

    total_expense = expense_df["Amount"].sum()
    m_summary = monthly_summary(df)
    avg_monthly_expense = m_summary["Expense"].mean() if not m_summary.empty else 0
    cat_summary = category_summary(df, "Expense")
    top_category = cat_summary.iloc[0]["Category"] if not cat_summary.empty else "N/A"
    top_amount = cat_summary.iloc[0]["Total"] if not cat_summary.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Expenses", format_inr(total_expense))
    c2.metric("Avg Monthly Expenses", format_inr(avg_monthly_expense))
    c3.metric("Highest Category", top_category)
    c4.metric("Highest Category Amount", format_inr(top_amount))

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Category Breakdown", "Monthly Trend", "Essential vs Discretionary"])

    with tab1:
        fig = px.pie(cat_summary, names="Category", values="Total", hole=0.4,
                     title="Expense Distribution by Category")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            cat_summary.assign(Total=lambda d: d["Total"].apply(format_inr),
                                Average=lambda d: d["Average"].apply(format_inr)),
            use_container_width=True, hide_index=True,
        )

    with tab2:
        if not m_summary.empty:
            fig2 = px.line(m_summary, x="Month", y="Expense", markers=True, title="Monthly Expense Trend")
            fig2.update_traces(line_color=DANGER_COLOR)
            fig2.update_layout(height=420)
            st.plotly_chart(fig2, use_container_width=True)

            matrix = monthly_category_matrix(df, "Expense")
            if not matrix.empty:
                st.caption("Monthly expense by category")
                fig3 = px.bar(matrix, x=matrix.index, y=matrix.columns, title="Monthly Category Breakdown")
                fig3.update_layout(height=420, xaxis_title="Month", yaxis_title="Amount (₹)")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Not enough data to show a monthly trend.")

    with tab3:
        bucket_df = essential_vs_discretionary(df)
        if not bucket_df.empty:
            fig4 = px.pie(bucket_df, names="Bucket", values="Total", hole=0.4,
                          color="Bucket", color_discrete_map={"Essential": PRIMARY_COLOR, "Discretionary": WARN_COLOR},
                          title="Essential vs Discretionary Spending")
            fig4.update_layout(height=420)
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("Categorization is a simplified heuristic for educational purposes based on common category names.")
        else:
            st.info("Not enough data for this breakdown.")


# ==========================================================================
# PAGE: ML PREDICTIONS
# ==========================================================================
def page_ml_predictions():
    st.title("🤖 ML Expense Predictions")
    st.caption("Real machine learning models trained on your transaction history.")

    df = get_active_transactions()
    m_summary = monthly_summary(df)

    if m_summary.empty:
        st.info("No transaction data available for training. Please upload a CSV or use the sample dataset.")
        return

    with st.spinner("Training and evaluating models..."):
        result = predict_next_month_expense(m_summary)

    if result.used_ml:
        st.success(f"✅ {result.message}")
    else:
        st.warning(f"⚠️ {result.message}")

    st.metric("Predicted Next-Month Expense", format_inr(result.predicted_next_month))
    st.caption(f"Model used: **{result.best_model_name}**")

    if result.model_results:
        st.markdown("---")
        st.subheader("Model Comparison")
        rows = []
        for name, mr in result.model_results.items():
            rows.append({
                "Model": name,
                "MAE (₹)": round(mr.mae, 2),
                "RMSE (₹)": round(mr.rmse, 2),
                "R²": round(mr.r2, 3) if mr.r2 is not None else "N/A",
                "Selected": "✅" if name == result.best_model_name else "",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            "MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) are in ₹ — lower is better. "
            "R² indicates how much variance is explained by the model (closer to 1 is better; can be negative "
            "on small test sets)."
        )

    st.markdown("---")
    st.subheader("Historical vs Predicted")
    hist = m_summary[["Month", "Expense"]].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["Month"], y=hist["Expense"], mode="lines+markers", name="Historical Expense",
                              line=dict(color=DANGER_COLOR)))
    fig.add_trace(go.Scatter(x=["Next Month"], y=[result.predicted_next_month], mode="markers",
                              name="Predicted", marker=dict(color=ACCENT_COLOR, size=14, symbol="star")))
    fig.update_layout(height=420, xaxis_title="Month", yaxis_title="Expense (₹)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="disclaimer-box">📌 Predictions are statistical estimates based on historical patterns '
        'and are not guaranteed. More historical data generally improves accuracy.</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# PAGE: SAVINGS FORECAST
# ==========================================================================
def page_savings_forecast():
    st.title("📈 Savings Forecast")
    profile: FinancialProfile = st.session_state["profile"]
    df = get_active_transactions()
    m_summary = monthly_summary(df)

    st.caption("Estimate how your savings may grow over time based on your profile.")

    c1, c2 = st.columns([1, 2])
    with c1:
        period_label = st.selectbox("Forecast Period", ["3 months", "6 months", "12 months", "24 months", "36 months"], index=2)
        months = int(period_label.split()[0])
        apply_trend = st.checkbox("Apply historical expense growth trend", value=True,
                                   help="Uses a modest trend estimated from your transaction history, clamped to ±3%/month.")

    growth_rate = estimate_monthly_expense_growth(m_summary) if apply_trend else 0.0

    points = forecast_savings(
        current_savings=profile.current_savings,
        monthly_income=profile.monthly_income,
        monthly_expenses=profile.total_monthly_expenses,
        monthly_investment=profile.monthly_investment,
        months=months,
        expense_growth_rate=growth_rate,
    )
    forecast_df = forecast_to_dataframe(points)

    with c2:
        st.metric(f"Projected Savings after {months} months", format_inr(forecast_df["ProjectedSavings"].iloc[-1]) if not forecast_df.empty else "₹0")
        st.caption(f"Estimated monthly expense growth trend applied: {round(growth_rate*100, 2)}%")

    st.markdown("---")
    if not forecast_df.empty:
        fig = px.line(forecast_df, x="Month", y="ProjectedSavings", markers=True, title="Projected Savings Over Time")
        fig.update_traces(line_color=PRIMARY_COLOR)
        fig.update_layout(height=420, yaxis_title="Projected Savings (₹)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Month-by-Month Forecast")
        display_df = forecast_df.copy()
        for col in ["ProjectedSavings", "ProjectedIncome", "ProjectedExpense"]:
            display_df[col] = display_df[col].apply(format_inr)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown(
        '<div class="disclaimer-box">📌 This forecast is an estimate based on your current profile and simple '
        'assumptions. Actual results will vary with real-world income, spending, and market conditions.</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# PAGE: GOAL PLANNER
# ==========================================================================
def page_goal_planner():
    st.title("🎯 Financial Goal Planner")
    profile: FinancialProfile = st.session_state["profile"]

    with st.form("goal_form"):
        c1, c2 = st.columns(2)
        with c1:
            goal_name = st.text_input("Goal Name", value=profile.goal_name or "My Goal")
            target_amount = st.number_input("Target Amount (₹)", min_value=0.0, value=float(profile.target_amount), step=5000.0)
        with c2:
            current_amount = st.number_input("Current Amount Saved (₹)", min_value=0.0, value=float(profile.current_savings), step=1000.0)
            monthly_contribution = st.number_input("Monthly Contribution (₹)", min_value=0.0,
                                                     value=float(max(profile.monthly_investment, 1000.0)), step=500.0)
        submitted = st.form_submit_button("📊 Calculate Goal Plan", use_container_width=True)

    if submitted or True:
        result = plan_goal(target_amount, current_amount, monthly_contribution)

        st.markdown("---")
        progress_pct = goal_progress_percent(target_amount, current_amount)
        st.subheader(f"Goal: {goal_name}")
        st.progress(min(1.0, progress_pct / 100))
        st.caption(f"{round(progress_pct, 1)}% complete — {format_inr(current_amount)} of {format_inr(target_amount)}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Remaining Amount", format_inr(result.remaining_amount))
        if result.is_already_achieved:
            c2.metric("Status", "🎉 Achieved")
            c3.metric("Months Required", "0")
        elif result.is_unreachable:
            c2.metric("Status", "⚠️ Needs Contribution")
            c3.metric("Months Required", "N/A")
        else:
            c2.metric("Est. Completion Date", result.estimated_completion_date.strftime("%b %Y"))
            c3.metric("Months Required", str(result.months_required))

        st.info(result.message)

        st.markdown("---")
        st.subheader("Reverse Calculator: Target a Specific Date")
        target_date_input = st.date_input("Target Completion Date", value=date.today().replace(year=date.today().year + 1))
        if target_date_input > date.today():
            required_contrib = required_monthly_contribution(target_amount, current_amount, target_date_input)
            st.success(f"To reach {format_inr(target_amount)} by {target_date_input.strftime('%B %Y')}, "
                       f"you need to contribute approximately **{format_inr(required_contrib)}/month**.")
        else:
            st.warning("Please select a future date.")


# ==========================================================================
# PAGE: RISK PROFILE
# ==========================================================================
def page_risk_profile():
    st.title("⚖️ Risk Profile Questionnaire")
    st.caption("Answer a few questions to understand your general risk tolerance.")
    st.info("This tool does NOT provide specific stock, crypto, or security recommendations, and does not promise investment returns.")

    with st.form("risk_form"):
        answers = {}
        for key, q in RISK_QUESTIONS.items():
            options = list(q["options"].keys())
            default_idx = 1 if len(options) > 1 else 0
            answers[key] = st.radio(q["label"], options, index=default_idx, key=f"risk_{key}")
        submitted = st.form_submit_button("🧮 Calculate Risk Profile", use_container_width=True)

    if submitted:
        result = score_answers(answers)
        st.session_state["risk_answers"] = answers
        st.session_state["risk_result"] = result

    result = st.session_state.get("risk_result")
    if result:
        st.markdown("---")
        color = {"LOW": DANGER_COLOR, "MODERATE": WARN_COLOR, "HIGH": PRIMARY_COLOR}.get(result.category, ACCENT_COLOR)
        st.markdown(f"### Your Risk Profile: <span style='color:{color}'>{result.category}</span>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Risk Score", f"{result.total_score}/{result.max_score}", f"{result.percentage}%")
        with c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result.percentage,
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#FFEBEE"},
                        {"range": [40, 70], "color": "#FFF3E0"},
                        {"range": [70, 100], "color": "#E8F5E9"},
                    ],
                },
                title={"text": "Risk Tolerance"},
            ))
            fig.update_layout(height=280)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(f'<div class="insight-item">{result.explanation}</div>', unsafe_allow_html=True)


# ==========================================================================
# PAGE: AI RECOMMENDATIONS
# ==========================================================================
def page_recommendations():
    st.title("💡 AI Recommendations")
    st.caption("Personalized, rule-based recommendations generated from your actual financial data.")

    profile: FinancialProfile = st.session_state["profile"]
    health = compute_financial_health_score(profile)
    df = get_active_transactions()
    cat_summary = category_summary(df, "Expense")
    top_category = cat_summary.iloc[0]["Category"] if not cat_summary.empty else None

    risk_result = st.session_state.get("risk_result")
    risk_category = risk_result.category if risk_result else None

    goal_progress = goal_progress_percent(profile.target_amount, profile.current_savings) if profile.target_amount else None

    recs = generate_recommendations(
        profile=profile,
        health=health,
        risk_category=risk_category,
        top_expense_category=top_category,
        goal_progress_pct=goal_progress,
    )

    for i, rec in enumerate(recs, 1):
        st.markdown(f"**{i}.** {rec}")
        st.markdown("")

    if not risk_result:
        st.info("💡 Tip: Complete the Risk Profile questionnaire for more tailored guidance.")

    st.markdown(
        '<div class="disclaimer-box">📌 These recommendations are generated by local rule-based logic using your '
        'own data. They are for educational purposes only and do not constitute professional financial advice. '
        'Consult a licensed financial advisor for personalized guidance.</div>',
        unsafe_allow_html=True,
    )


# ==========================================================================
# PAGE: ABOUT
# ==========================================================================
def page_about():
    st.title("ℹ️ About This Project")
    st.markdown("""
### AI Financial Advisor
**End-to-End Data Science & Machine Learning Application**

This is a local, offline personal-finance analysis tool built as a data science
portfolio project. It demonstrates a complete pipeline: data ingestion and
cleaning, exploratory analysis, feature engineering, supervised machine
learning (with model comparison), time-based forecasting, rule-based
recommendation logic, and an interactive dashboard — all running entirely
on your machine.

**What it does:**
- Analyzes income, expenses, and savings from your data
- Trains real regression models to predict next-month expenses
- Forecasts future savings under simple assumptions
- Helps plan and track financial goals
- Profiles general risk tolerance via a questionnaire
- Generates personalized, rule-based recommendations
- Visualizes everything with interactive charts

**What it does NOT do:**
- It does not connect to any bank, brokerage, or live market data
- It does not use any paid or external AI API
- It does not recommend specific stocks, funds, or securities
- It does not guarantee any investment returns

### Disclaimer
This application is provided for **educational and portfolio purposes only**.
It does not constitute professional financial, investment, tax, or legal advice.
All calculations are simplified, transparent heuristics intended to
demonstrate data science and machine learning techniques. Always consult a
qualified, licensed financial advisor before making real financial decisions.
""")


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------
PAGE_FUNCS = {
    "🏠 Home": page_home,
    "👤 Financial Profile": page_profile,
    "📤 Upload Transactions": page_upload,
    "📊 Expense Analysis": page_expense_analysis,
    "🤖 ML Predictions": page_ml_predictions,
    "📈 Savings Forecast": page_savings_forecast,
    "🎯 Goal Planner": page_goal_planner,
    "⚖️ Risk Profile": page_risk_profile,
    "💡 AI Recommendations": page_recommendations,
    "ℹ️ About": page_about,
}

try:
    PAGE_FUNCS[page]()
except Exception as exc:  # noqa: BLE001 - top-level safety net so users never see a raw stack trace
    st.error("⚠️ Something went wrong while rendering this page.")
    st.error(f"Details: {exc}")
    with st.expander("Technical details (for debugging)"):
        st.code(traceback.format_exc())
