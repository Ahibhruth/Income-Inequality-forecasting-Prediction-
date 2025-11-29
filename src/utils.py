
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = (mean_squared_error(y_true, y_pred)) ** 0.5
    return {"mae": float(mae), "rmse": float(rmse)}
