import pandas as pd
import streamlit as st

from src import config as C
from src.prediction import load_artifacts, predict_customer



st.set_page_config(
    page_title="Customer Churn Risk", 
    page_icon="📉", 
    layout="wide"
)


REPORT_PATH = C.REPORTS_DIR / "customer_risk_report.csv"


@st.cache_data
def load_report() -> pd.DataFrame:
    return pd.read_csv(REPORT_PATH)


@st.cache_resource
def get_artifacts():
    return load_artifacts()


st.title("Customer Churn Risk")

if not REPORT_PATH.exists() or not (C.MODELS_DIR / "churn_model.pkl").exists():
    st.error("Model or report not found. Run `python run_pipeline.py` first.")
    st.stop()

report = load_report()
art = get_artifacts()
tab_dash, tab_predict = st.tabs(["Dashboard", "Predict a customer"])


# ------------------------------- Dashboard -------------------------------
with tab_dash:
    predicted_churn = int((report["Churn_Probability"] >= C.RISK_LOW_MAX).sum())
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total customers", f"{len(report):,}")
    c2.metric(
        "Predicted churn customers", 
        f"{predicted_churn:,}", help=f"Churn probability >= {C.RISK_LOW_MAX:.0%}"
    )
    c3.metric("High-risk customers", f"{(report['Risk_Level'] == 'High').sum():,}")
    c4.metric("Medium-risk customers", f"{(report['Risk_Level'] == 'Medium').sum():,}")
    c5.metric("Average churn probability", f"{report['Churn_Probability'].mean():.1%}")

    left, right = st.columns(2)
    
    with left:
        st.subheader("Customers by risk level")
        st.bar_chart(report["Risk_Level"].value_counts().reindex(["Low", "Medium", "High"]), color="#2a78d6")
    with right:
        st.subheader("Customers by retention priority")
        pr = report["Retention_Priority"].value_counts().drop("No action", errors="ignore").sort_index()
        st.bar_chart(pr, color="#2a78d6")

    st.subheader("Who should the retention team contact first?")
    f1, f2 = st.columns(2)
    
    risk_sel = f1.multiselect(
        "Risk level", 
        ["High", "Medium", "Low"], 
        default=["High"]
    )

    prio_sel = f2.multiselect(
        "Retention priority", 
        sorted(report["Retention_Priority"].unique()),
        default=["Priority 1", "Priority 2", "Priority 3"]
    )

    view = report[report["Risk_Level"].isin(risk_sel) & report["Retention_Priority"].isin(prio_sel)]
    cols = [
        "Customer_ID", "Churn_Probability", "Risk_Level", "Customer_Value", "Retention_Priority", "Key_Risk_Factor", "Recommended_Action"
    ]

    st.caption(f"{len(view):,} customers match")
    st.dataframe(
        view[cols], 
        width="stretch", 
        hide_index=True,
        column_config={
            "Churn_Probability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0, max_value=1, format="%.2f"
            )
        }
    )
    st.download_button("Download this list (CSV)", view[cols].to_csv(index=False), "retention_list.csv", "text/csv")


# ------------------------------- Prediction -------------------------------
with tab_predict:
    st.subheader("Score a single customer")
    
    with st.form("customer"):
        a, b = st.columns(2)
        tenure = a.number_input("Tenure (months)", 1, 120, 12)
        monthly = b.number_input("Monthly charges (INR)", 100.0, 2000.0, 599.0, step=10.0)
        contract = a.selectbox("Contract", C.ORDINAL_FEATURES["Contract_Type"])
        tickets = b.number_input("Support tickets", 0, 30, 1)
        usage = a.number_input("Usage hours / month", 0.0, 300.0, 20.0)
        satisfaction = b.slider("Customer satisfaction (1-5)", 1, 5, 3)
        last_login = a.number_input("Days since last login", 0, 365, 5)
        complaints = b.number_input("Complaints", 0, 30, 0)
        go = st.form_submit_button("Predict churn risk")
    
    st.caption("Fields not shown here (age, location, payment method...) use typical training-data values.")

    if go:
        res = predict_customer({
            "Tenure_Months": tenure, 
            "Monthly_Charges": monthly, 
            "Contract_Type": contract,
            "Support_Tickets": tickets, 
            "Usage_Hours": usage, 
            "Customer_Satisfaction": satisfaction,
            "Last_Login_Days": last_login, 
            "Complaints": complaints},
            art
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Churn probability", f"{res['Churn_Probability']:.0%}")
        r2.metric("Risk level", res["Risk_Level"])
        r3.metric("Retention priority", res["Retention_Priority"])
        
        st.progress(min(max(res["Churn_Probability"], 0.0), 1.0))
        st.write(f"**Main risk factor:** {res['Key_Risk_Factor']}")
        st.write(f"**Recommended action:** {res['Recommended_Action']}")
        st.write(f"**Customer value tier:** {res['Customer_Value']}")
