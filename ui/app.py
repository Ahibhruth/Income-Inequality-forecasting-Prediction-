
import os
import streamlit as st
import pandas as pd
import joblib
# from src.evaluate_placeholder import None  # placeholder to avoid import errors in snippet

st.set_page_config(page_title="Income Inequality Dashboard", layout="wide")

st.title("💰 Income Inequality Forecasting")
st.write("Upload a CSV with the same processed columns as your training data to get predictions.")

MODEL_PATH = os.path.join("models", "latest_model.pkl")
PROCESSED_X_PATH = os.path.join("data", "processed", "X_train.csv")
RAW_PATH = os.path.join("data", "raw", "data1.csv")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Dataset preview")
    if os.path.exists(RAW_PATH):
        df_raw = pd.read_csv(RAW_PATH)
        st.dataframe(df_raw.head(100))
    else:
        st.info("Place data/raw/data1.csv to preview the dataset here.")

with col2:
    st.header("Model status")
    if os.path.exists(MODEL_PATH):
        st.success("Model found: models/latest_model.pkl")
        try:
            model = joblib.load(MODEL_PATH)
            st.write("Model type:", type(model).__name__)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
    else:
        st.warning("Model not found. Run: python src/train.py")

st.markdown("---")
st.header("Batch prediction (CSV)")

uploaded = st.file_uploader("Upload a processed CSV (same columns as training) for batch predictions", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        model = joblib.load(MODEL_PATH)
        preds = model.predict(df)
        df["Predicted_Financial_Stress"] = preds
        st.dataframe(df.head(50))
        csv = df.to_csv(index=False)
        st.download_button("Download predictions", csv, "predictions.csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.info("Make sure uploaded CSV has exactly the same features (columns) used in training.")
