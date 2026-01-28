import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Smart Loan Approval System",
    layout="wide"
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>

/* ===== GLOBAL BACKGROUND ===== */
.stApp {
    background: linear-gradient(
        135deg,
        #f0f4ff,
        #e6f7f1,
        #fff7e6
    );
    background-attachment: fixed;
    padding: 100px;
}

/* ===== TITLE SPACING ===== */
h1 {
    margin-top: 2.5rem !important;
}

/* ===== SIDEBAR FIX ===== */
section[data-testid="stSidebar"] {
    height: 100vh;
    overflow-y: auto !important;
    padding-bottom: 2rem;
}

/* ===== MAIN CONTENT CARD ===== */
.block-container {
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

/* ===== SIDEBAR GRADIENT ===== */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1f3c88,
        #2a5298,
        #1e3c72
    );
    color: white;
}

/* ===== SIDEBAR LABELS ===== */
section[data-testid="stSidebar"] label {
    font-size: 16px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 0.6rem 1.4rem;
    font-size: 16px;
    font-weight: 600;
    border: none;
    transition: 0.3s ease;
}

.stButton>button:hover {
    transform: scale(1.03);
    box-shadow: 0 6px 18px rgba(0,0,0,0.2);
}

/* ===== ALERTS ===== */
div.stAlert-success {
    background: linear-gradient(90deg, #e0f8e9, #c6f6d5);
    border-radius: 10px;
}

div.stAlert-error {
    background: linear-gradient(90deg, #ffe0e0, #ffbdbd);
    border-radius: 10px;
}

/* ===== FOOTER ===== */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE & DESCRIPTION
# =========================================================
st.title("🎯 Smart Loan Approval System – Stacking Model")
st.write(
    "This system uses a **Stacking Ensemble Machine Learning model** to predict "
    "whether a loan will be approved by combining multiple ML models for better decision making."
)

# =========================================================
# LOAD & PREPROCESS DATA (CACHED)
# =========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("loan.csv")

    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include='object').columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

X_train, X_test, y_train, y_test = load_data()

# =========================================================
# TRAIN MODELS (CACHED — CRITICAL FIX)
# =========================================================
@st.cache_resource
def train_models(X_train, y_train):

    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000))
    ])

    dt_model = DecisionTreeClassifier(random_state=42)

    rf_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    stack_model = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("dt", DecisionTreeClassifier(random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=150, random_state=42))
        ],
        final_estimator=LogisticRegression(),
        cv=5
    )

    stack_model.fit(X_train, y_train)

    return lr_model, dt_model, rf_model, stack_model

with st.spinner("Training models... Please wait ⏳"):
    lr_model, dt_model, rf_model, stack_model = train_models(X_train, y_train)

# =========================================================
# SIDEBAR INPUTS
# =========================================================
st.sidebar.header("📝 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0)
co_income = st.sidebar.number_input("Co-Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
loan_term = st.sidebar.number_input("Loan Amount Term", min_value=0)

credit_history = st.sidebar.radio("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Salaried", "Self-Employed"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# =========================================================
# MODEL ARCHITECTURE DISPLAY
# =========================================================
st.subheader("🧠 Stacking Model Architecture")
st.info("""
**Base Models Used**
- Logistic Regression
- Decision Tree
- Random Forest

**Meta Model Used**
- Logistic Regression
""")

# =========================================================
# INPUT PREPARATION
# =========================================================
def prepare_input():
    data = {
        'ApplicantIncome': app_income,
        'CoapplicantIncome': co_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': 1 if credit_history == "Yes" else 0,
        'Self_Employed_Yes': 1 if employment == "Self-Employed" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }

    input_df = pd.DataFrame([data])

    for col in X_train.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[X_train.columns]

# =========================================================
# PREDICTION BUTTON
# =========================================================
if st.button("🔘 Check Loan Eligibility (Stacking Model)"):

    input_data = prepare_input()

    lr_pred = lr_model.predict(input_data)[0]
    dt_pred = dt_model.predict(input_data)[0]
    rf_pred = rf_model.predict(input_data)[0]

    final_pred = stack_model.predict(input_data)[0]
    confidence = stack_model.predict_proba(input_data)[0][final_pred] * 100

    st.subheader("📊 Prediction Result")

    if final_pred == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write("### 📊 Base Model Predictions")
    st.write(f"• Logistic Regression → {'Approved' if lr_pred == 1 else 'Rejected'}")
    st.write(f"• Decision Tree → {'Approved' if dt_pred == 1 else 'Rejected'}")
    st.write(f"• Random Forest → {'Approved' if rf_pred == 1 else 'Rejected'}")

    st.write("### 🧠 Final Stacking Decision")
    st.write(f"**Confidence Score:** {confidence:.2f}%")

    st.subheader("📌 Business Explanation")

    if final_pred == 1:
        st.write(
            "Based on income, credit history, and combined predictions from multiple "
            "machine learning models, the applicant is likely to repay the loan. "
            "Therefore, the stacking model **approves the loan**."
        )
    else:
        st.write(
            "Based on income, credit history, and combined predictions from multiple "
            "models, the applicant is unlikely to repay the loan reliably. "
            "Therefore, the stacking model **rejects the loan**."
        )
