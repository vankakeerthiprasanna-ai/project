import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="ChurnGuard AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 19px;
}

.stApp {
    background-color: #F8FAFC;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
}

section[data-testid="stSidebar"] * {
    color: #F1F5F9 !important;
    font-size: 18px !important;
}

section[data-testid="stSidebar"] .stRadio label {
    font-size: 18px !important;
    font-weight: 500 !important;
    padding: 12px 0 !important;
}

/* Metric Cards */
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 22px;
}

div[data-testid="stMetric"] label {
    font-size: 16px !important;
    color: #64748B !important;
    font-weight: 600 !important;
}

div[data-testid="stMetric"] div {
    font-size: 34px !important;
    color: #0F172A !important;
    font-weight: 700 !important;
}

/* Main Titles */
h1 {
    font-size: 40px !important;
    font-weight: 800 !important;
    color: #0F172A !important;
}

h2, h3 {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #0F172A !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
    color: white !important;
    border: none;
    border-radius: 10px;
    height: 54px;
    font-size: 19px !important;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #B91C1C 0%, #991B1B 100%);
    color: white !important;
}

/* Labels */
.stSlider label, .stNumberInput label, .stSelectbox label {
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Retention Cards */
.retention-card {
    background: white;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 26px 30px;
    margin-bottom: 20px;
    font-size: 18px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ==================== LOAD ====================
@st.cache_resource
def load_model():
    model = joblib.load("models/churn_model.pkl")
    features = joblib.load("models/features.pkl")
    return model, features

@st.cache_data
def load_data():
    return pd.read_csv("data/telecom_churn_data.csv")

model, features = load_model()
df = load_data()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("##")
    st.markdown("### 📡 ChurnGuard AI")
    st.caption("Customer Intelligence Platform")
    st.markdown("---")
    
    page = st.radio(
        "Go to",
        ["Overview", "Predict Risk", "Analytics", "Retention"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("**Model Information**")
    st.markdown("""
    - Algorithm: Gradient Boosting  
    - Features Used: 13  
    - Training Records: 5,000
    """)

# ==================== PAGES ====================
if page == "Overview":
    st.title("Customer Churn Prediction")
    st.markdown("##### AI-powered platform to identify at-risk customers and reduce revenue leakage")
    st.markdown("---")

    total = len(df)
    churned = int(df["Churn"].sum())
    active = total - churned
    churn_rate = (churned / total) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Customers", f"{total:,}")
    m2.metric("Active Customers", f"{active:,}")
    m3.metric("Churned Customers", f"{churned:,}")
    m4.metric("Churn Rate", f"{churn_rate:.1f}%")

    st.markdown("##")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.pie(
            names=["Active", "Churned"],
            values=[active, churned],
            hole=0.55,
            color_discrete_sequence=["#10B981", "#DC2626"]
        )
        fig.update_layout(title="Customer Status", height=380, font=dict(size=16))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        imp = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=True).tail(8)

        fig2 = px.bar(
            imp, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=["#FEE2E2", "#DC2626"]
        )
        fig2.update_layout(title="Top Drivers of Churn", height=380, coloraxis_showscale=False, font=dict(size=15))
        st.plotly_chart(fig2, use_container_width=True)

elif page == "Predict Risk":
    st.title("Predict Customer Risk")
    st.markdown("##### Enter customer details to check churn probability")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (Months)", 1, 60, 12)
        monthly_recharge = st.number_input("Monthly Recharge (₹)", 99, 999, 399)
        recharge_freq = st.slider("Recharge Frequency / Month", 1, 8, 2)
        data_usage = st.number_input("Avg Data Usage (GB)", 5.0, 150.0, 40.0)

    with col2:
        service_quality = st.slider("Service Quality Score", 1.0, 10.0, 7.0)
        call_drop = st.slider("Call Drop Rate (%)", 0.0, 15.0, 3.0)
        complaints = st.slider("Complaint Count", 0, 12, 1)
        resolution_days = st.slider("Avg Resolution Days", 0.5, 10.0, 2.0)

    with col3:
        competitor = st.slider("Competitor Coverage Score", 1.0, 10.0, 5.0)
        plan_fit = st.slider("Plan Fit Score", 1.0, 10.0, 7.0)
        age = st.slider("Age", 18, 65, 32)
        gender = st.selectbox("Gender", ["Male", "Female"])
        location = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"])

    st.markdown("##")

    if st.button("Run Prediction"):
        gender_enc = 1 if gender == "Male" else 0
        loc_map = {"Urban": 2, "Semi-Urban": 1, "Rural": 0}

        input_data = np.array([[
            tenure, monthly_recharge, recharge_freq, data_usage,
            service_quality, call_drop, complaints, resolution_days,
            competitor, plan_fit, age, gender_enc, loc_map[location]
        ]])

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] * 100

        st.markdown("---")

        res1, res2 = st.columns([1, 1.3])

        with res1:
            if pred == 1:
                st.error("High Risk Detected")
                st.markdown(f"### Probability: **{prob:.1f}%**")
                st.caption("Immediate retention action recommended")
            else:
                st.success("Low Risk")
                st.markdown(f"### Probability: **{prob:.1f}%**")
                st.caption("Customer appears stable")

        with res2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob,
                number={"suffix": "%", "font": {"size": 42}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#DC2626" if prob >= 50 else "#10B981"},
                    "steps": [
                        {"range": [0, 30], "color": "#D1FAE5"},
                        {"range": [30, 60], "color": "#FEF3C7"},
                        {"range": [60, 100], "color": "#FEE2E2"}
                    ]
                }
            ))
            fig.update_layout(height=280, margin=dict(t=20, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

elif page == "Analytics":
    st.title("Analytics")
    st.markdown("##### Deep dive into churn patterns")
    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(df, x="Churn", y="Service_Quality_Score", color="Churn",
                     color_discrete_sequence=["#10B981", "#DC2626"])
        fig.update_layout(title="Service Quality vs Churn", height=390, showlegend=False, font=dict(size=15))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(df, x="Churn", y="Complaint_Count", color="Churn",
                     color_discrete_sequence=["#10B981", "#DC2626"])
        fig.update_layout(title="Complaints vs Churn", height=390, showlegend=False, font=dict(size=15))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.scatter(df, x="Plan_Fit_Score", y="Competitor_Coverage_Score",
                         color="Churn", color_discrete_sequence=["#10B981", "#DC2626"], opacity=0.6)
        fig.update_layout(title="Plan Fit vs Competitor Coverage", height=390, font=dict(size=15))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(df, x="Tenure_Months", color="Churn", barmode="overlay",
                           color_discrete_sequence=["#10B981", "#DC2626"])
        fig.update_layout(title="Tenure Distribution", height=390, font=dict(size=15))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Retention":
    st.title("Retention Strategies")
    st.markdown("##### Recommended actions based on risk level")
    st.markdown("---")

    st.markdown("""
    <div class="retention-card">
        <h3 style="color:#DC2626; margin-top:0;">High Risk (Above 60%)</h3>
        <ul>
            <li>Trigger immediate outreach from retention specialist</li>
            <li>Offer personalized plan upgrade with limited-time discount</li>
            <li>Assign priority support queue</li>
            <li>Provide complimentary data or calling benefits</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="retention-card">
        <h3 style="color:#D97706; margin-top:0;">Medium Risk (30% – 60%)</h3>
        <ul>
            <li>Send targeted plan recommendation based on usage</li>
            <li>Activate loyalty bonus points</li>
            <li>Share network improvement updates in their area</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="retention-card">
        <h3 style="color:#059669; margin-top:0;">Low Risk (Below 30%)</h3>
        <ul>
            <li>Continue regular engagement campaigns</li>
            <li>Offer referral incentives</li>
            <li>Collect feedback to maintain satisfaction</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)