import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "random_forest_model.pkl")
feature_names = joblib.load(BASE_DIR / "models" / "feature_names.pkl")

# -----------------------------
# Premium Header
# -----------------------------

st.markdown("""
<style>

/* Remove Top Padding */
.block-container{
    padding-top:2rem;
}

/* Header Card */
.header-container{
    background:linear-gradient(135deg,#2563EB,#1E3A8A);
    padding:35px;
    border-radius:20px;
    box-shadow:0px 8px 25px rgba(0,0,0,0.18);
    text-align:center;
    margin-bottom:20px;
}

/* Title */
.header-title{
    font-size:48px;
    font-weight:800;
    color:white;
    margin-bottom:8px;
}

/* Subtitle */
.header-subtitle{
    font-size:20px;
    color:#E5E7EB;
    margin-bottom:30px;
}

/* Info Cards */
.info-row{
    display:flex;
    justify-content:center;
    gap:20px;
    flex-wrap:wrap;
}

.info-card{
    background:rgba(255,255,255,0.12);
    padding:18px 28px;
    border-radius:15px;
    min-width:170px;
    backdrop-filter:blur(8px);
}

.info-title{
    color:#BFDBFE;
    font-size:14px;
    font-weight:600;
}

.info-value{
    color:white;
    font-size:20px;
    font-weight:700;
    margin-top:6px;
}

</style>

<div class="header-container">

<div class="header-title">
📊 Customer Churn Prediction Dashboard
</div>

<div class="header-subtitle">
AI Powered Telecom Customer Analytics using Machine Learning
</div>

<div class="info-row">

<div class="info-card">
<div class="info-title">🤖 MODEL</div>
<div class="info-value">Random Forest</div>
</div>

<div class="info-card">
<div class="info-title">📈 ACCURACY</div>
<div class="info-value">82.6%</div>
</div>

<div class="info-card">
<div class="info-title">💼 DOMAIN</div>
<div class="info-value">Telecom</div>
</div>

<div class="info-card">
<div class="info-title">⚡ STATUS</div>
<div class="info-value">Production Ready</div>
</div>

</div>

</div>

""", unsafe_allow_html=True)

st.divider()

# ==========================
# Dashboard Statistics
# ==========================

st.markdown("## 📈 Dashboard Overview")

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div style="
        background:#2563EB;
        padding:20px;
        border-radius:15px;
        color:white;
        text-align:center;
        box-shadow:0px 4px 12px rgba(0,0,0,0.15);
    ">
        <h3>👥</h3>
        <h2>7043</h2>
        <p>Total Customers</p>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div style="
        background:#16A34A;
        padding:20px;
        border-radius:15px;
        color:white;
        text-align:center;
        box-shadow:0px 4px 12px rgba(0,0,0,0.15);
    ">
        <h3>🎯</h3>
        <h2>82.6%</h2>
        <p>Model Accuracy</p>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div style="
        background:#F59E0B;
        padding:20px;
        border-radius:15px;
        color:white;
        text-align:center;
        box-shadow:0px 4px 12px rgba(0,0,0,0.15);
    ">
        <h3>📊</h3>
        <h2>26.5%</h2>
        <p>Dataset Churn Rate</p>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown("""
    <div style="
        background:#7C3AED;
        padding:20px;
        border-radius:15px;
        color:white;
        text-align:center;
        box-shadow:0px 4px 12px rgba(0,0,0,0.15);
    ">
        <h3>🤖</h3>
        <h2>Random Forest</h2>
        <p>Algorithm</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
# ==========================
# Sidebar
# ==========================

with st.sidebar:

   BASE_DIR = Path(__file__).resolve().parent.parent

with st.sidebar:

    st.image(
    BASE_DIR / "images" / "churn_logo.png",
    width=140
)

    st.markdown(
        "<h3 style='text-align:center;'>Customer Churn AI</h3>",
        unsafe_allow_html=True
    )

    st.markdown("""
<p style="
text-align:center;
color:#B8C2CC;
font-size:15px;
line-height:1.6;">
Telecom Customer Retention Prediction
</p>
""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 📊 Model")

st.markdown(
    "<h3 style='text-align:center; margin-top:15px;'>Customer Churn AI</h3>",
    unsafe_allow_html=True
)

st.markdown(
    """
        <p style="text-align:center; color:gray;">
        Telecom Customer Retention Prediction
        </p>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

st.markdown("### 📊 Model")
st.info("""
🤖 **Random Forest**

Accuracy: **82.6%**
""")

st.markdown("### 🎯 Purpose")

st.write(
        "Predict whether a telecom customer is likely to churn based on customer information and subscribed services."
    )

st.markdown("---")

st.info("""
### 👩‍💻 Developer

**Sejal Singh**

Machine Learning Project
""")
st.markdown("---")

st.subheader("📂 Batch Prediction")

uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

st.markdown("---")

st.subheader("🎨 Theme")

dark_mode = st.toggle("Dark Mode", value=True)

st.markdown("---")
# ==========================
# User Inputs
# ==========================

col1, col2, col3 = st.columns(3)

# ---------------- Customer ----------------
with col1:
    st.subheader("👤 Customer Information")

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (Months)", 0, 72, 12)

# ---------------- Services ----------------
with col2:
    st.subheader("🌐 Services")

    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple = st.selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    security = st.selectbox(
        "Online Security",
        ["No", "Yes", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["No", "Yes", "No internet service"]
    )

    device = st.selectbox(
        "Device Protection",
        ["No", "Yes", "No internet service"]
    )

    support = st.selectbox(
        "Tech Support",
        ["No", "Yes", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes", "No internet service"]
    )

# ---------------- Billing ----------------
with col3:
    st.subheader("💳 Billing")

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )


# ==========================
# Prediction
# ==========================

c1, c2, c3 = st.columns([1, 2, 1])

with c2:
    predict = st.button(
        "🚀 Predict Customer Churn",
        type="primary",
        use_container_width=True
    )

if predict:

    # -------------------------
    # Prepare Input
    # -------------------------
    input_df = pd.DataFrame({
        "Gender": [gender],
        "Senior_Citizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "Tenure_Months": [tenure],
        "Phone_Service": [phone],
        "Multiple_Lines": [multiple],
        "Internet_Service": [internet],
        "Online_Security": [security],
        "Online_Backup": [backup],
        "Device_Protection": [device],
        "Tech_Support": [support],
        "Streaming_TV": [tv],
        "Streaming_Movies": [movies],
        "Contract": [contract],
        "Paperless_Billing": [paperless],
        "Payment_Method": [payment],
        "Monthly_Charges": [monthly],
        "Total_Charges": [total]
    })

    # Encode input
    input_df = pd.get_dummies(input_df)

    # Match training columns
    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    st.markdown("## 🎯 Prediction Result")

    left, right = st.columns([1, 1])

    # ===================================
    # Result Card
    # ===================================
    with left:

        if prediction == 1:

            st.markdown(f"""
<div style="
background:linear-gradient(135deg,#7F1D1D,#991B1B);
border-left:8px solid #EF4444;
padding:30px;
border-radius:18px;
box-shadow:0 10px 25px rgba(0,0,0,0.30);
">

<h2 style="
color:#FCA5A5;
font-size:42px;
margin-bottom:20px;
">
⚠️ High Churn Risk
</h2>

<h3 style="
color:#F9FAFB;
font-size:30px;
font-weight:500;
">
Customer is likely to <b>CHURN</b>.
</h3>

<hr style="
border:1px solid rgba(255,255,255,0.15);
margin:30px 0;
">

<h2 style="
color:#FECACA;
font-size:38px;
">
Probability: {probability:.2%}
</h2>

</div>
""", unsafe_allow_html=True)

        else:

            st.markdown(f"""
<div style="
background:linear-gradient(135deg,#14532D,#1B4332);
border-left:8px solid #22C55E;
padding:30px;
border-radius:18px;
box-shadow:0 10px 25px rgba(0,0,0,0.30);
">

<h2 style="
color:#4ADE80;
font-size:42px;
margin-bottom:20px;
">
✅ Low Churn Risk
</h2>

<h3 style="
color:#F9FAFB;
font-size:30px;
font-weight:500;
">
Customer is likely to <b>STAY</b>.
</h3>

<hr style="
border:1px solid rgba(255,255,255,0.15);
margin:30px 0;
">

<h2 style="
color:#86EFAC;
font-size:38px;
">
Probability: {probability:.2%}
</h2>

</div>
""", unsafe_allow_html=True)
        st.metric(
            "🎯 Churn Probability",
            f"{probability:.2%}"
        )

        st.progress(probability)

    # ===================================
    # Gauge Meter
    # ===================================
    with right:

        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Churn Risk"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#2563EB"},
                    "steps": [
                        {"range": [0, 40], "color": "#86EFAC"},
                        {"range": [40, 70], "color": "#FACC15"},
                        {"range": [70, 100], "color": "#F87171"}
                    ]
                }
            )
        )

        fig.update_layout(height=350)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ===================================
    # Recommendation
    # ===================================
    st.markdown("---")
    st.subheader("💡 AI Recommendation")

    if probability < 0.30:

        st.success("""
### 🟢 Low Risk

Customer has a very low probability of churn.

**Recommendation**
- Continue current service.
- Send appreciation emails.
- Offer loyalty rewards.
        """)

    elif probability < 0.70:

        st.warning("""
### 🟡 Medium Risk

Customer may churn in the future.

**Recommendation**
- Offer personalized discounts.
- Improve customer engagement.
- Recommend suitable service upgrades.
        """)

    else:

        st.error("""
### 🔴 High Risk

Customer is very likely to churn.

**Recommendation**
- Contact the customer immediately.
- Offer retention discounts.
- Assign a customer support executive.
- Resolve complaints as soon as possible.
        """)


# ===================================
# Feature Importance
# ===================================

st.markdown("---")
st.subheader("📊 Top Feature Importance")

try:
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    ).head(10)

    fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        text_auto=".3f",
        title="Top 10 Most Important Features"
    )

    fig.update_layout(
        height=500,
        yaxis={"categoryorder": "total ascending"}
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except Exception:
    st.info(
        "Feature importance is not available for the loaded model."
    )


# ===================================
# Download Report
# ===================================

if predict:

    report = pd.DataFrame({
        "Prediction": [
            "Churn" if prediction == 1 else "Stay"
        ],
        "Probability": [
            round(probability * 100, 2)
        ],
        "Gender": [gender],
        "Contract": [contract],
        "Monthly Charges": [monthly],
        "Total Charges": [total]
    })

    csv = report.to_csv(index=False)

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        file_name="customer_prediction_report.csv",
        mime="text/csv",
        use_container_width=True
    )


# ===================================
# Customer Summary
# ===================================

st.markdown("---")
st.subheader("📋 Customer Summary")

summary1, summary2, summary3 = st.columns(3)

with summary1:
    st.info(f"""
**👤 Customer Profile**

- Gender : {gender}
- Senior Citizen : {senior}
- Partner : {partner}
- Dependents : {dependents}
- Tenure : {tenure} Months
""")

with summary2:
    st.info(f"""
**🌐 Services**

- Internet : {internet}
- Contract : {contract}
- Phone Service : {phone}
- Tech Support : {support}
- Streaming TV : {tv}
""")

with summary3:
    st.info(f"""
**💳 Billing**

- Monthly Charges : ${monthly:.2f}
- Total Charges : ${total:.2f}
- Paperless Billing : {paperless}
- Payment Method :
{payment}
""")

# ===================================
# Batch Prediction
# ===================================

if uploaded_file is not None:

    st.markdown("---")
    st.subheader("📂 Batch Prediction Results")

    batch_df = pd.read_csv(uploaded_file)

    st.write("### Uploaded Dataset")
    st.dataframe(batch_df)

    batch_encoded = pd.get_dummies(batch_df)

    batch_encoded = batch_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    predictions = model.predict(batch_encoded)

    probabilities = model.predict_proba(batch_encoded)[:, 1]

    batch_df["Prediction"] = [
        "Churn" if x == 1 else "Stay"
        for x in predictions
    ]

    batch_df["Probability"] = probabilities

    st.success("Prediction Completed ✅")

    st.dataframe(batch_df)

    csv = batch_df.to_csv(index=False)

    st.download_button(
        "📥 Download Batch Results",
        csv,
        "batch_predictions.csv",
        "text/csv",
        use_container_width=True
    )

# ===================================
# Analytics Dashboard
# ===================================

st.markdown("---")
st.subheader("📊 Customer Analytics")

analytics1, analytics2 = st.columns(2)

with analytics1:

    contract_df = pd.DataFrame({
        "Contract": [
            "Month-to-month",
            "One Year",
            "Two Year"
        ],
        "Customers": [
            3875,
            1473,
            1695
        ]
    })

    fig1 = px.pie(
        contract_df,
        values="Customers",
        names="Contract",
        title="Contract Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

with analytics2:

    internet_df = pd.DataFrame({
        "Service":[
            "Fiber",
            "DSL",
            "No Internet"
        ],
        "Customers":[
            3096,
            2421,
            1526
        ]
    })

    fig2 = px.bar(
        internet_df,
        x="Service",
        y="Customers",
        title="Internet Service Distribution",
        text_auto=True
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

st.markdown("---")

analytics3, analytics4 = st.columns(2)

with analytics3:

    churn_df = pd.DataFrame({
        "Status":[
            "Stayed",
            "Churned"
        ],
        "Customers":[
            5174,
            1869
        ]
    })

    fig3 = px.pie(
        churn_df,
        values="Customers",
        names="Status",
        hole=0.55,
        title="Overall Churn Rate"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

with analytics4:

    charges_df = pd.DataFrame({
        "Range":[
            "$0-30",
            "$30-60",
            "$60-90",
            "$90+"
        ],
        "Customers":[
            910,
            2210,
            2500,
            1423
        ]
    })

    fig4 = px.bar(
        charges_df,
        x="Range",
        y="Customers",
        title="Monthly Charges Distribution",
        text_auto=True
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )
# ==========================
# Footer
# ==========================

st.markdown("---")

st.markdown("""
<div style="
background:linear-gradient(135deg,#111827,#1F2937);
padding:30px;
border-radius:18px;
text-align:center;
margin-top:30px;
box-shadow:0px 8px 20px rgba(0,0,0,0.15);
">

<h2 style="color:white;margin-bottom:10px;">
📊 Customer Churn Prediction Dashboard
</h2>

<p style="color:#D1D5DB;font-size:17px;">
Built with ❤️ using
<b>Python • Streamlit • Scikit-Learn • Plotly</b>
</p>

<hr style="border:1px solid #374151;">

<p style="color:#9CA3AF;font-size:16px;">
👩‍💻 Developed by
</p>

<h2 style="color:#60A5FA;margin:5px 0;">
Sejal Singh
</h2>


<p style="color:#9CA3AF;font-size:14px;">
© 2026 Customer Churn Prediction Dashboard. All Rights Reserved.
</p>

</div>
""", unsafe_allow_html=True)

