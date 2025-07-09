import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------- Theme Toggle --------------------
mode = st.sidebar.radio("🎨 Choose Theme", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("""<style>body { background-color: #0e1117; color: white; }</style>""", unsafe_allow_html=True)

# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown("""
This interactive tool uses **Logistic Regression** and **XGBoost** models to predict whether a telecom customer is likely to churn.
""")

# -------------------- Load Models --------------------
@st.cache_resource
def load_models():
    logreg = joblib.load("models/logreg_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    return logreg, xgb_model, scaler, label_encoders

logreg, xgb_model, scaler, label_encoders = load_models()

input_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                  'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                  'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                  'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                  'MonthlyCharges', 'TotalCharges']

# -------------------- Input Option --------------------
input_mode = st.sidebar.radio("📥 Input Mode", ["Single Customer", "CSV Upload"])

# -------------------- Collect Inputs --------------------
def preprocess_input(df):
    df_copy = df.copy()
    for col in label_encoders:
        df_copy[col] = label_encoders[col].transform(df_copy[col])
    scaled = scaler.transform(df_copy)
    return scaled

if input_mode == "Single Customer":
    st.sidebar.header("🧾 Customer Profile")
    user_data = {}
    for feature in input_features:
        if feature in label_encoders:
            options = list(label_encoders[feature].classes_)
            user_data[feature] = st.sidebar.selectbox(f"{feature}", options)
        else:
            user_data[feature] = st.sidebar.number_input(f"{feature}", min_value=0.0, step=0.5)
    input_df = pd.DataFrame([user_data])
    input_scaled = preprocess_input(input_df)

else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV with customer data", type=["csv"])
    if uploaded_file:
        input_df = pd.read_csv(uploaded_file)
        st.write("📄 Uploaded Data Preview:")
        st.dataframe(input_df.head())
        input_scaled = preprocess_input(input_df)
    else:
        st.warning("📎 Please upload a CSV file.")
        st.stop()

# -------------------- Prediction --------------------
logreg_preds = logreg.predict_proba(input_scaled)[:, 1]
xgb_preds = xgb_model.predict_proba(input_scaled)[:, 1]
avg_preds = (logreg_preds + xgb_preds)/2

input_df["LogReg_Churn_Prob"] = logreg_preds
input_df["XGBoost_Churn_Prob"] = xgb_preds

# -------------------- Show Prediction --------------------
st.subheader("🔍 Prediction Results")

if input_mode == "Single Customer":
    st.metric("Logistic Regression", f"{logreg_preds[0]:.2%}")
    st.metric("XGBoost", f"{xgb_preds[0]:.2%}")
    st.metric("Average",f"{avg_preds[0]:.2%}")
else:
    st.dataframe(input_df)

# -------------------- SHAP Explainability --------------------
st.subheader("🧠 Model Explanation (XGBoost)")
explainer = shap.Explainer(xgb_model)
shap_values = explainer(input_scaled)

if input_mode == "Single Customer":
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
else:
    with st.expander("📈 SHAP Summary Plot (All CSV Rows)"):
        fig2 = plt.figure()
        shap.summary_plot(shap_values, features=input_df[input_features], show=False)
        st.pyplot(fig2)

# -------------------- Export Predictions --------------------
st.subheader("📤 Download Results")

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(input_df)
st.download_button("⬇️ Download Predictions as CSV", data=csv_data, file_name="churn_predictions.csv", mime="text/csv")

# -------------------- Footer --------------------
st.markdown("---")

