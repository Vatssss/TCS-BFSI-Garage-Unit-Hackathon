import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = joblib.load("voting_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="German Credit Risk Prediction", layout="wide")
st.title("\U0001F4B8 German Credit Risk Prediction App | By Vatsal Saxena")
st.markdown("""
This application predicts whether a loan applicant is a **Good** or **Bad** credit risk using a trained ensemble model.
""")

with st.sidebar:
    st.header("User Input Parameters")

    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 18, 75, 30)
    job = st.selectbox("Job", [0, 1, 2, 3])
    housing = st.selectbox("Housing", ["own", "free", "rent"])
    saving_acct = st.selectbox("Saving Account", ["little", "moderate", "rich", "quite rich", "no_info"])
    checking_acct = st.selectbox("Checking Account", ["little", "moderate", "rich", "no_info"])
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
    duration = st.slider("Duration (months)", 4, 72, 24)
    purpose = st.selectbox("Purpose", ['radio/TV', 'education', 'furniture/equipment', 'new car', 'used car', 'business', 'domestic appliance', 'repairs', 'other'])

# Encoding inputs
input_dict = {
    'Sex': 0 if sex == 'male' else 1,
    'Age': age,
    'Job': job,
    'Credit amount': credit_amount,
    'Duration': duration
}

# One-hot encoding
one_hot_cols = {
    f'Housing_{housing}': 1,
    f'Saving accounts_{saving_acct}': 1,
    f'Checking account_{checking_acct}': 1,
    f'Purpose_{purpose}': 1
}

# Fill missing one-hot columns with 0
for col in [
    'Housing_free', 'Housing_own',
    'Saving accounts_little', 'Saving accounts_moderate', 'Saving accounts_quite rich', 'Saving accounts_rich',
    'Checking account_little', 'Checking account_moderate', 'Checking account_rich',
    'Purpose_domestic appliance', 'Purpose_education', 'Purpose_furniture/equipment', 'Purpose_new car',
    'Purpose_other', 'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_used car']:
    if col not in one_hot_cols:
        one_hot_cols[col] = 0

# Age binning
if age <= 25:
    age_group = 'Young'
elif age <= 35:
    age_group = 'Adult'
elif age <= 50:
    age_group = 'Senior'
else:
    age_group = 'Elder'

input_dict.update({
    'Age_group_Adult': 1 if age_group == 'Adult' else 0,
    'Age_group_Senior': 1 if age_group == 'Senior' else 0,
    'Age_group_Elder': 1 if age_group == 'Elder' else 0
})

# Credit binning
if credit_amount <= 1365:
    credit_bin = 'Low'
elif credit_amount <= 2319:
    credit_bin = 'Medium'
elif credit_amount <= 3972:
    credit_bin = 'High'
else:
    credit_bin = 'Very_High'

input_dict.update({
    'Credit_bin_Medium': 1 if credit_bin == 'Medium' else 0,
    'Credit_bin_High': 1 if credit_bin == 'High' else 0,
    'Credit_bin_Very_High': 1 if credit_bin == 'Very_High' else 0
})

# Final feature vector
full_input = {**input_dict, **one_hot_cols}
input_df = pd.DataFrame([full_input])
input_df = input_df.reindex(columns=feature_names, fill_value=0)
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Risk Score")
        st.progress(int(probability * 100))
        st.metric("Probability of Bad Credit Risk", f"{probability:.2%}")
    with col2:
        if prediction == 1:
            st.error("Prediction: Bad Credit Risk")
        else:
            st.success("Prediction: Good Credit Risk")

    # Explanation (if available)
    st.markdown("---")
    st.subheader("Model Insights")
    try:
        feat_df = pd.read_csv("feature_importance.csv").head(10)
        st.bar_chart(data=feat_df, x="Feature", y="Importance")
    except FileNotFoundError:
        st.info("Feature importance file not found.")
