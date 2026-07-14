import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page configurations
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1E3A8A; margin-bottom: 5px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔮 Telco Customer Churn Prediction Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">An End-to-End Production ML Pipeline constructed using Scikit-learn & Streamlit for DevelopersHub Corporation Internship Phase 2.</div>',
    unsafe_allow_html=True)

# Smart Relative Path Resolution
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(current_dir, "saved_pipeline", "customer_churn_pipeline.joblib")


@st.cache_resource
def load_ml_pipeline(path):
    if os.path.exists(path):
        return joblib.load(path)
    # Fallback in case path resolver checks root instead
    fallback = "./Task 2 - Customer Churn/saved_pipeline/customer_churn_pipeline.joblib"
    if os.path.exists(fallback):
        return joblib.load(fallback)
    raise FileNotFoundError(f"Pipeline target file not found at: {path}")


try:
    pipeline = load_ml_pipeline(pipeline_path)
    st.success("✅ Scikit-learn Production Pipeline API loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading serialized model weights: {e}")
    st.stop()

# Layout Configuration
st.markdown("### 📋 Customer Demographics & Service Profiles")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen (Age >= 65)", [0, 1])
    partner = st.selectbox("Has a Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months with company)", min_value=0, max_value=120, value=12)

with col2:
    phone_service = st.selectbox("Phone Service Status", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines Configuration", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security Addon", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup Addon", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection Contract", ["No", "Yes", "No internet service"])

with col3:
    tech_support = st.selectbox("Premium Tech Support Service", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV Subscription", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies Subscription", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing Profile", ["Yes", "No"])
    payment_method = st.selectbox("Preferred Payment Mode", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

st.markdown("### 💳 Financial Billing Information")
f_col1, f_col2 = st.columns(2)
with f_col1:
    monthly_charges = st.number_input("Monthly Subscription Charges ($)", min_value=0.0, value=65.0)
with f_col2:
    total_charges = st.number_input("Total Cumulative Charges ($)", min_value=0.0, value=780.0)

input_data = pd.DataFrame([{
    'gender': gender, 'SeniorCitizen': int(senior_citizen), 'Partner': partner, 'Dependents': dependents,
    'tenure': int(tenure), 'PhoneService': phone_service, 'MultipleLines': multiple_lines,
    'InternetService': internet_service, 'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
    'DeviceProtection': device_protection, 'TechSupport': tech_support, 'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies, 'Contract': contract, 'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method, 'MonthlyCharges': float(monthly_charges), 'TotalCharges': float(total_charges)
}])

st.markdown("---")

if st.button("🚀 Analyze Churn Likelihood", type="primary"):
    with st.spinner("Processing transaction inputs..."):
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]

        churn_probability = probabilities[1] * 100
        retention_probability = probabilities[0] * 100

        st.markdown("### 📊 Prediction Output Metrics")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            if prediction == 1:
                st.error(f"⚠️ **High Churn Risk Detected!**")
                st.metric(label="Churn Probability Score", value=f"{churn_probability:.2f}%")
            else:
                st.success(f"💖 **Low Churn Risk / Safe Retention Profile**")
                st.metric(label="Retention Stability Score", value=f"{retention_probability:.2f}%")
        with p_col2:
            st.markdown(f"**Structural Decomposition Metrics:**")
            st.info(
                f"• Active Churn Risk Probability: **{churn_probability:.2f}%**\n\n• Loyal Customer Retention Probability: **{retention_probability:.2f}%**")