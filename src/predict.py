"""
src/predict.py
- Programmatic helper for loading model and predicting one record (dict) or DataFrame
"""
import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "latest_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found. Train a model first (python src/train.py).")
    return joblib.load(MODEL_PATH)

def predict_single(record: dict):
    model = load_model()
    df = pd.DataFrame([record])
    pred = model.predict(df)[0]
    return float(pred)

def predict_df(df: pd.DataFrame):
    model = load_model()
    preds = model.predict(df)
    return preds
