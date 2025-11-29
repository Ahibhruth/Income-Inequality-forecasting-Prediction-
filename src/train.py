"""
src/train.py
- Loads processed train/test CSVs
- Trains a RandomForestRegressor
- Logs basic metrics to MLflow
- Saves model to models/latest_model.pkl
"""
import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

PROCESSED_DIR = os.path.join("data", "processed")
MODEL_DIR = "models"
EXPERIMENT_NAME = "income_inequality_forecasting"

def load_data():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv"))
    return X_train, y_train.values.ravel(), X_test, y_test.values.ravel()

def train(n_estimators: int = 300, random_state: int = 42):
    X_train, y_train, X_test, y_test = load_data()

    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))
        mlflow.sklearn.log_model(model, "model")

        model_path = os.path.join(MODEL_DIR, "latest_model.pkl")
        joblib.dump(model, model_path)

        print(f"Training complete. Model saved to {model_path}")
        print(f"MAE: {mae:.6f}, R2: {r2:.6f}")

if __name__ == "__main__":
    train()
