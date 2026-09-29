import streamlit as st
import pandas as pd
import joblib

model = joblib.load("churn_random_forest.pkl")
scaler = joblib.load("churn_scaler.pkl")
columns = joblib.load("churn_columns.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict whether the customer is likely to churn.")

st.divider()

st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=1
    )

    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=24.8
    )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=24.8
    )

with col3:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No", "Yes"]
    )
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )

st.subheader("Internet Services")

col1, col2, col3 = st.columns(3)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes"]
    )

    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes"]
    )

st.subheader("Streaming Services")

col1, col2 = st.columns(2)

with col1:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No", "Yes"]
    )

with col2:
    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No", "Yes"]
    )

st.subheader("Contract and Payment")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )

st.divider()

if st.button("🔮 Predict Churn", use_container_width=True):

    input_data = {
        "gender": 1 if gender == "Male" else 0,
        "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
        "Partner": 1 if partner == "Yes" else 0,
        "Dependents": 1 if dependents == "Yes" else 0,

        "tenure": tenure,

        "PhoneService": 1 if phone_service == "Yes" else 0,
        "MultipleLines": 1 if multiple_lines == "Yes" else 0,
        "OnlineSecurity": 1 if online_security == "Yes" else 0,
        "OnlineBackup": 1 if online_backup == "Yes" else 0,
        "DeviceProtection": 1 if device_protection == "Yes" else 0,
        "TechSupport": 1 if tech_support == "Yes" else 0,
        "StreamingTV": 1 if streaming_tv == "Yes" else 0,
        "StreamingMovies": 1 if streaming_movies == "Yes" else 0,

        "PaperlessBilling": 1 if paperless_billing == "Yes" else 0,

        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "InternetService_DSL":
            1 if internet_service == "DSL" else 0,

        "InternetService_Fiber optic":
            1 if internet_service == "Fiber optic" else 0,

        "InternetService_No":
            1 if internet_service == "No" else 0,

        "Contract_Month-to-month":
            1 if contract == "Month-to-month" else 0,

        "Contract_One year":
            1 if contract == "One year" else 0,

        "Contract_Two year":
            1 if contract == "Two year" else 0,

        "PaymentMethod_Bank transfer (automatic)":
            1 if payment_method == "Bank transfer (automatic)" else 0,

        "PaymentMethod_Credit card (automatic)":
            1 if payment_method == "Credit card (automatic)" else 0,

        "PaymentMethod_Electronic check":
            1 if payment_method == "Electronic check" else 0,

        "PaymentMethod_Mailed check":
            1 if payment_method == "Mailed check" else 0
    }

    input_df = pd.DataFrame([input_data])

    input_df = input_df.reindex(
        columns=columns,
        fill_value=0
    )

    scale_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    input_df[scale_columns] = scaler.transform(
        input_df[scale_columns]
    )

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error("⚠️ Customer is likely to CHURN.")
    else:
        st.success("✅ Customer is likely to STAY.")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    st.progress(float(probability))
