"""
src/preprocess.py
- Reads data/raw/data1.csv (already processed in Colab)
- Splits into train/test and writes to data/processed/
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = os.path.join("data", "raw", "data1.csv")
PROCESSED_DIR = os.path.join("data", "processed")

def preprocess(test_size: float = 0.2, random_state: int = 42):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"{RAW_PATH} not found. Put your data1.csv in data/raw/")
    df = pd.read_csv(RAW_PATH)

    if "Financial_Stress" not in df.columns:
        raise ValueError("Target column 'Financial_Stress' not found in data1.csv")

    X = df.drop("Financial_Stress", axis=1)
    y = df["Financial_Stress"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    X_train.to_csv(os.path.join(PROCESSED_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

    print("Preprocessing complete. Files saved to data/processed/")

if __name__ == "__main__":
    preprocess()
