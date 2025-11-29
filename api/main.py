"""
api/main.py - FastAPI application
POST /predict  => accepts {"features": { ... }} and returns predicted financial stress
GET  /health   => health check
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict_single

app = FastAPI(title="Income Inequality Forecasting API", version="1.0")

class InputData(BaseModel):
    features: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(payload: InputData):
    try:
        pred = predict_single(payload.features)
        return {"predicted_financial_stress": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
