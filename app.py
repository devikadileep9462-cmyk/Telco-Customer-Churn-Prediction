
import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "models/best_churn_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# ==========================================
# 3. APPLICATION TITLE
# ==========================================

st.title("📊 Telco Customer Churn Prediction")

st.write(
    "An AI-powered application to predict whether "
    "a telecom customer is likely to leave the company."
)

st.divider()


if model is None:

    st.error(
        "Trained model not found! "
        "Please run train.py first."
    )

    st.stop()


st.success("Trained model loaded successfully!")


# ==========================================
# 4. CUSTOMER INPUT FORM
# ==========================================

st.subheader("Enter Customer Details")

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    # --------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------

    with col1:

        st.markdown("### Customer Information")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

    # --------------------------------------
    # SERVICES INFORMATION
    # --------------------------------------

    with col2:

        st.markdown("### Services Information")

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

    # --------------------------------------
    # ACCOUNT INFORMATION
    # --------------------------------------

    with col3:

        st.markdown("### Account Information")

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0,
            step=1.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "Predict Customer Churn",
        use_container_width=True
    )


# ==========================================
# 5. MAKE PREDICTION
# ==========================================

if submitted:

    input_data = pd.DataFrame({

        "gender": [gender],

        "SeniorCitizen": [senior_citizen],

        "Partner": [partner],

        "Dependents": [dependents],

        "tenure": [tenure],

        "PhoneService": [phone_service],

        "MultipleLines": [multiple_lines],

        "InternetService": [internet_service],

        "OnlineSecurity": [online_security],

        "OnlineBackup": [online_backup],

        "DeviceProtection": [device_protection],

        "TechSupport": [tech_support],

        "StreamingTV": [streaming_tv],

        "StreamingMovies": [streaming_movies],

        "Contract": [contract],

        "PaperlessBilling": [paperless_billing],

        "PaymentMethod": [payment_method],

        "MonthlyCharges": [monthly_charges],

        "TotalCharges": [total_charges]

    })


    # Get prediction

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    churn_percentage = probability * 100


    st.divider()

    st.subheader("Prediction Results")


    # ======================================
    # 6. DISPLAY RESULTS
    # ======================================

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Churn Probability",
            f"{churn_percentage:.2f}%"
        )

    with result_col2:

        st.metric(
            "Customer Retention Probability",
            f"{(1 - probability) * 100:.2f}%"
        )


    if prediction == 1:

        st.error(
            "⚠️ High Churn Risk: Customer is predicted to churn."
        )

        st.markdown("### Recommended Retention Actions")

        st.write(
            "- Offer a personalized retention discount."
        )

        st.write(
            "- Encourage the customer to choose a longer contract."
        )

        st.write(
            "- Review customer service and technical support."
        )

        st.write(
            "- Provide suitable service packages."
        )

    else:

        st.success(
            "✅ Low Churn Risk: Customer is predicted to stay."
        )

        st.markdown("### Recommended Actions")

        st.write(
            "- Continue providing quality customer service."
        )

        st.write(
            "- Maintain customer satisfaction."
        )

        st.write(
            "- Offer suitable loyalty benefits."
        )


    # ======================================
    # 7. PROBABILITY VISUALIZATION
    # ======================================

    st.subheader("Churn Probability Analysis")

    probability_df = pd.DataFrame({
        "Category": ["Churn", "Stay"],
        "Probability": [
            probability,
            1 - probability
        ]
    })

    st.bar_chart(
        probability_df.set_index("Category")
    )


    st.caption(
        "Note: Predictions are based on the trained "
        "machine learning model and should support, "
        "not replace, business decisions."
    )
