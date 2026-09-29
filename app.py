import streamlit as st
import pandas as pd
import joblib
import shap

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

model = joblib.load("churn_random_forest.pkl")
scaler = joblib.load("churn_scaler.pkl")
columns = joblib.load("churn_columns.pkl")

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn based on their profile and service details.")

st.divider()

st.subheader("👤 Customer Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])

with col2:
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

with col3:
    partner = st.selectbox("Partner", ["No", "Yes"])

with col4:
    dependents = st.selectbox("Dependents", ["No", "Yes"])


st.subheader("📞 Services")

col1, col2, col3 = st.columns(3)

with col1:
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])

with col3:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    online_security = st.selectbox(
        "Online Security",
        ["No", "Yes"]
    )

with col2:
    online_backup = st.selectbox(
        "Online Backup",
        ["No", "Yes"]
    )

with col3:
    device_protection = st.selectbox(
        "Device Protection",
        ["No", "Yes"]
    )

with col4:
    tech_support = st.selectbox(
        "Tech Support",
        ["No", "Yes"]
    )

col1, col2, col3 = st.columns(3)

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

with col3:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )


st.subheader("💳 Billing & Contract")

col1, col2, col3, col4 = st.columns(4)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0
    )

with col3:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=600.0
    )

with col4:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

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

predict_button = st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
)

if predict_button:

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
        "InternetService_DSL": 1 if internet_service == "DSL" else 0,
        "InternetService_Fiber optic": 1 if internet_service == "Fiber optic" else 0,
        "InternetService_No": 1 if internet_service == "No" else 0,
        "Contract_Month-to-month": 1 if contract == "Month-to-month" else 0,
        "Contract_One year": 1 if contract == "One year" else 0,
        "Contract_Two year": 1 if contract == "Two year" else 0,
        "PaymentMethod_Bank transfer (automatic)": 1 if payment_method == "Bank transfer (automatic)" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if payment_method == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if payment_method == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if payment_method == "Mailed check" else 0
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
    probability = model.predict_proba(input_df)[0]

    st.divider()
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        if prediction == 1:
            st.error("⚠️ Customer is likely to churn")
        else:
            st.success("✅ Customer is likely to stay")

    with col2:
        st.metric(
            "Churn Probability",
            f"{probability[1] * 100:.2f}%"
        )

    st.progress(float(probability[1]))

    st.divider()
    st.subheader("🔍 Prediction Explanation")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        feature_shap = shap_values[1][0]

    elif hasattr(shap_values, "values"):
        values = shap_values.values

        if values.ndim == 3:
            feature_shap = values[0, :, 1]
        else:
            feature_shap = values[0]

    elif shap_values.ndim == 3:
        feature_shap = shap_values[0, :, 1]

    else:
        feature_shap = shap_values[0]

    explanation_df = pd.DataFrame({
        "Feature": columns,
        "SHAP Value": feature_shap
    })

    feature_names = {
        "gender": "Gender",
        "SeniorCitizen": "Senior Citizen",
        "Partner": "Partner",
        "Dependents": "Dependents",
        "tenure": "Tenure",
        "PhoneService": "Phone Service",
        "MultipleLines": "Multiple Lines",
        "OnlineSecurity": "Online Security",
        "OnlineBackup": "Online Backup",
        "DeviceProtection": "Device Protection",
        "TechSupport": "Tech Support",
        "StreamingTV": "Streaming TV",
        "StreamingMovies": "Streaming Movies",
        "PaperlessBilling": "Paperless Billing",
        "MonthlyCharges": "Monthly Charges",
        "TotalCharges": "Total Charges",
        "InternetService_DSL": "DSL Internet",
        "InternetService_Fiber optic": "Fiber Optic Internet",
        "InternetService_No": "No Internet Service",
        "Contract_Month-to-month": "Month-to-Month Contract",
        "Contract_One year": "One-Year Contract",
        "Contract_Two year": "Two-Year Contract",
        "PaymentMethod_Bank transfer (automatic)": "Bank Transfer",
        "PaymentMethod_Credit card (automatic)": "Credit Card",
        "PaymentMethod_Electronic check": "Electronic Check",
        "PaymentMethod_Mailed check": "Mailed Check"
    }

    explanation_df["Feature"] = explanation_df["Feature"].map(
        feature_names
    )

    positive = explanation_df[
        explanation_df["SHAP Value"] > 0
    ].sort_values(
        "SHAP Value",
        ascending=False
    ).head(5)

    negative = explanation_df[
        explanation_df["SHAP Value"] < 0
    ].sort_values(
        "SHAP Value",
        ascending=True
    ).head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### ⚠️ Factors Increasing Churn")

        if len(positive) == 0:
            st.info("No significant factors increasing churn.")

        else:
            max_value = positive["SHAP Value"].abs().max()

            for _, row in positive.iterrows():
                value = abs(float(row["SHAP Value"]))

                st.write(f"**{row['Feature']}**")
                st.progress(
                    min(value / max_value, 1.0)
                )
                st.caption(f"Impact: {value:.4f}")

    with col2:
        st.markdown("### ✅ Factors Reducing Churn")

        if len(negative) == 0:
            st.info("No significant factors reducing churn.")

        else:
            max_value = negative["SHAP Value"].abs().max()

            for _, row in negative.iterrows():
                value = abs(float(row["SHAP Value"]))

                st.write(f"**{row['Feature']}**")
                st.progress(
                    min(value / max_value, 1.0)
                )
                st.caption(f"Impact: {value:.4f}")

    st.caption(
        "Positive SHAP values push the prediction toward churn, "
        "while negative SHAP values push the prediction toward staying."
    )