
import streamlit as st
import pandas as pd
import joblib
import os


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ChurnIQ | Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}

[data-testid="stSidebar"] {
    background-color: #111C30;
    border-right: 1px solid #263449;
}

[data-testid="stSidebar"] * {
    color: #E2E8F0;
}

.main-title {
    font-size: 36px;
    font-weight: 800;
    color: #F8FAFC;
    margin-bottom: 5px;
}

.subtitle {
    color: #94A3B8;
    font-size: 15px;
    margin-bottom: 25px;
}

.section-title {
    color: #F8FAFC;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(135deg, #17243A, #1B3046);
    border: 1px solid #2B4059;
    border-radius: 14px;
    padding: 22px;
    min-height: 125px;
}

.metric-label {
    color: #94A3B8;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
}

.metric-value {
    color: #5EEAD4;
    font-size: 30px;
    font-weight: 800;
    margin-top: 8px;
}

.info-card {
    background-color: #17243A;
    border: 1px solid #2B4059;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 15px;
}

div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #0D9488;
    color: white;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    padding: 10px 20px;
    width: 100%;
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    background-color: #0F766E;
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background-color: #17243A;
    border: 1px solid #2B4059;
    padding: 16px;
    border-radius: 12px;
}

div[data-testid="stMetricLabel"] {
    color: #94A3B8;
}

div[data-testid="stMetricValue"] {
    color: #5EEAD4;
}

hr {
    border-color: #263449;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = "models/best_churn_model.pkl"


@st.cache_resource
def load_model():

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    return None


model = load_model()


# ==========================================
# 4. SIDEBAR NAVIGATION
# ==========================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 10px 0 25px 0;">
        <h1 style="color:#5EEAD4; margin-bottom:0;">
            ◈ ChurnIQ
        </h1>
        <p style="color:#94A3B8; font-size:13px;">
            CUSTOMER INTELLIGENCE PLATFORM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Home",
            "Predict Churn",
            "Model Insights"
        ]
    )

    st.markdown("---")

    st.markdown("""
    <div style="background:#17243A;
                padding:15px;
                border-radius:12px;">
        <p style="color:#94A3B8; font-size:12px;">
            POWERED BY
        </p>
        <h4 style="color:#5EEAD4; margin:0;">
            Machine Learning
        </h4>
        <p style="color:#94A3B8; font-size:12px;">
            Tuned Stacking Classifier
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 5. MODEL CHECK
# ==========================================

if model is None:

    st.error(
        "Trained model not found. "
        "Please run train.py before starting the application."
    )

    st.stop()


# ==========================================
# 6. HOME DASHBOARD
# ==========================================

if page == "Home":

    st.markdown(
        '<div class="main-title">Customer Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Understand customer behaviour and identify churn risks '
        'using machine learning.'
        '</div>',
        unsafe_allow_html=True
    )

    # Model metrics

    col1, col2, col3, col4 = st.columns(4)

    metrics = [
        ("ROC-AUC", "84.64%"),
        ("Accuracy", "75.23%"),
        ("Churn Recall", "79.95%"),
        ("F1 Score", "63.15%")
    ]

    for column, metric in zip(
        [col1, col2, col3, col4],
        metrics
    ):

        with column:

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{metric[0]}</div>
                <div class="metric-value">{metric[1]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown("""
        <div class="info-card">
            <h3 style="color:#5EEAD4;">About ChurnIQ</h3>
            <p style="color:#CBD5E1; line-height:1.8;">
                ChurnIQ is an AI-powered customer retention
                application designed to predict whether a
                telecom customer is likely to leave the company.
            </p>
            <p style="color:#CBD5E1; line-height:1.8;">
                It uses machine learning and ensemble learning
                techniques to analyze customer information,
                estimate churn probability and suggest
                retention actions.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with right:

        st.markdown("""
        <div class="info-card">
            <h3 style="color:#5EEAD4;">Best Model</h3>
            <h2 style="color:#F8FAFC;">
                Tuned Stacking
            </h2>
            <p style="color:#94A3B8;">
                Ensemble classification model
            </p>
            <hr>
            <p style="color:#CBD5E1;">
                ROC-AUC: <b style="color:#5EEAD4;">0.8464</b>
            </p>
            <p style="color:#CBD5E1;">
                Churn Recall: <b style="color:#5EEAD4;">79.95%</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Machine Learning Workflow</div>',
        unsafe_allow_html=True
    )

    workflow = st.columns(5)

    workflow_items = [
        ("01", "Data Cleaning"),
        ("02", "EDA"),
        ("03", "Model Training"),
        ("04", "Evaluation"),
        ("05", "Prediction")
    ]

    for column, item in zip(workflow, workflow_items):

        with column:

            st.markdown(f"""
            <div class="info-card" style="text-align:center;">
                <h2 style="color:#5EEAD4;">{item[0]}</h2>
                <p style="color:#CBD5E1; font-size:13px;">
                    {item[1]}
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.info(
        "Navigate to 'Predict Churn' to analyze an individual customer."
    )


# ==========================================
# 7. CUSTOMER CHURN PREDICTION
# ==========================================

elif page == "Predict Churn":

    st.markdown(
        '<div class="main-title">Customer Churn Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter customer details to estimate the likelihood of churn.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Customer Information</div>',
        unsafe_allow_html=True
    )

    with st.form("customer_prediction_form"):

        col1, col2, col3 = st.columns(3)

        # CUSTOMER DETAILS

        with col1:

            st.markdown("#### Personal Details")

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
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

        # SERVICES

        with col2:

            st.markdown("#### Service Details")

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

        # ACCOUNT DETAILS

        with col3:

            st.markdown("#### Account Details")

            contract = st.selectbox(
                "Contract",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
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
                "Monthly Charges ($)",
                min_value=0.0,
                value=70.0,
                step=1.0
            )

            total_charges = st.number_input(
                "Total Charges ($)",
                min_value=0.0,
                value=840.0,
                step=10.0
            )

        st.markdown("---")

        submitted = st.form_submit_button(
            "✦  Predict Customer Churn",
            use_container_width=True
        )

    # ======================================
    # 8. PREDICTION RESULTS
    # ======================================

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

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        churn_percentage = probability * 100

        retention_percentage = (1 - probability) * 100

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Prediction Results</div>',
            unsafe_allow_html=True
        )

        result1, result2 = st.columns(2)

        with result1:

            st.metric(
                "Churn Probability",
                f"{churn_percentage:.2f}%"
            )

        with result2:

            st.metric(
                "Retention Probability",
                f"{retention_percentage:.2f}%"
            )

        st.markdown("### Churn Risk Assessment")

        if probability >= 0.75:

            risk = "Critical"

            st.error(
                "CRITICAL RISK — This customer has a high estimated "
                "churn probability."
            )

        elif probability >= 0.50:

            risk = "High"

            st.warning(
                "HIGH RISK — This customer may be likely to churn."
            )

        elif probability >= 0.30:

            risk = "Moderate"

            st.info(
                "MODERATE RISK — Monitor this customer and consider "
                "retention activities."
            )

        else:

            risk = "Low"

            st.success(
                "LOW RISK — This customer has a relatively low "
                "estimated churn probability."
            )

        st.progress(float(probability))

        st.caption(
            f"Estimated churn risk: {risk} | "
            f"Model probability: {churn_percentage:.2f}%"
        )

        if prediction == 1:

            st.markdown("### Recommended Retention Actions")

            recommendations = [
                "Offer a personalized retention discount.",
                "Encourage the customer to choose a longer contract.",
                "Review technical support and service satisfaction.",
                "Provide suitable service packages.",
                "Consider personalized loyalty benefits."
            ]

            for recommendation in recommendations:

                st.markdown(
                    f"- {recommendation}"
                )

        else:

            st.markdown("### Recommended Customer Actions")

            recommendations = [
                "Continue providing quality customer service.",
                "Maintain customer satisfaction.",
                "Offer suitable loyalty benefits.",
                "Monitor customer engagement."
            ]

            for recommendation in recommendations:

                st.markdown(
                    f"- {recommendation}"
                )

        st.caption(
            "Predictions are estimates based on the trained model "
            "and should support, not replace, business decisions."
        )


# ==========================================
# 9. MODEL INSIGHTS
# ==========================================

elif page == "Model Insights":

    st.markdown(
        '<div class="main-title">Model Performance & Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Explore the performance of the machine learning models.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Best Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", "75.23%")

    col2.metric("ROC-AUC", "84.64%")

    col3.metric("Churn Recall", "79.95%")

    # Model comparison data

    st.markdown("### Model Comparison")

    comparison_data = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "AdaBoost Original",
            "XGBoost Original",
            "Soft Voting Original",
            "Stacking Original",
            "AdaBoost Tuned",
            "XGBoost Tuned",
            "Soft Voting Tuned",
            "Stacking Tuned"
        ],

        "ROC-AUC": [
            0.8416,
            0.8208,
            0.8424,
            0.8436,
            0.8455,
            0.8389,
            0.8448,
            0.8459,
            0.8464
        ]

    })

    st.bar_chart(
        comparison_data.set_index("Model"),
        y="ROC-AUC"
    )

    # Feature importance

    st.markdown("### Important Churn Features")

    feature_data = pd.DataFrame({

        "Feature": [
            "Month-to-month Contract",
            "Fiber optic Internet",
            "No Online Security",
            "Two year Contract",
            "One year Contract",
            "DSL Internet",
            "Streaming Movies",
            "No Tech Support",
            "No Internet Service",
            "Customer Tenure"
        ],

        "Importance": [
            0.468664,
            0.081037,
            0.056475,
            0.035192,
            0.026545,
            0.026396,
            0.023744,
            0.022870,
            0.017912,
            0.015003
        ]

    })

    st.bar_chart(
        feature_data.set_index("Feature"),
        y="Importance"
    )

    st.info(
        "Feature importance values are taken from the tuned "
        "XGBoost model. They indicate predictive contribution, "
        "not causation."
    )

    st.markdown("### Project Technologies")

    tech_col1, tech_col2, tech_col3 = st.columns(3)

    tech_col1.markdown(
        "**Machine Learning**\n\n"
        "- Scikit-learn\n"
        "- XGBoost\n"
        "- AdaBoost"
    )

    tech_col2.markdown(
        "**Data Analysis**\n\n"
        "- Pandas\n"
        "- NumPy\n"
        "- Matplotlib\n"
        "- Seaborn"
    )

    tech_col3.markdown(
        "**Deployment**\n\n"
        "- Streamlit\n"
        "- Joblib\n"
        "- Python"
    )


# ==========================================
# 10. FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:10px;">
    <p style="color:#64748B; font-size:12px;">
        ChurnIQ | AI-Powered Customer Retention Dashboard
    </p>
    <p style="color:#64748B; font-size:12px;">
        Built with Python, Machine Learning and Streamlit
    </p>
</div>
""", unsafe_allow_html=True)
