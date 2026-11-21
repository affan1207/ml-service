import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="STREAM ML Inference Service (PUE / WUE)", version="1.0.0")

pue_wue_bundle = joblib.load("artifacts/pue_wue_quantile_bundle.joblib")


class FacilityReadingRequest(BaseModel):
    cooling_system: str = Field(..., example="Chiller + Cooling Tower")
    climate_zone: str = Field(..., example="4B")
    facility_size: str = Field(..., example="Midsize")
    submitted_pue: Optional[float] = Field(None, example=1.28)
    submitted_wue: Optional[float] = Field(None, example=3.55)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "pue_wue_metrics": pue_wue_bundle["metrics"],
    }


@app.get("/categories")
def get_valid_categories():
    """Valid dropdown options for the frontend form."""
    return pue_wue_bundle.get("valid_categories", {})


@app.post("/predict/pue-wue")
def predict_pue_wue(req: FacilityReadingRequest):
    if req.submitted_pue is not None and not (1.0 <= req.submitted_pue <= 3.5):
        raise HTTPException(
            status_code=422,
            detail="Physical rule violation: PUE must be between 1.0 and 3.5.",
        )

    X = pd.DataFrame([{
        "cooling_system": req.cooling_system,
        "climate_zone": req.climate_zone,
        "facility_size": req.facility_size,
    }])

    pue_pred = {
        k: round(float(m.predict(X)[0]), 4)
        for k, m in pue_wue_bundle["pue_models"].items()
    }
    wue_pred = {
        k: round(float(m.predict(X)[0]), 4)
        for k, m in pue_wue_bundle["wue_models"].items()
    }

    pue_anomaly = False
    wue_anomaly = False
    if req.submitted_pue is not None:
        pue_anomaly = not (pue_pred["p10"] <= req.submitted_pue <= pue_pred["p90"])
    if req.submitted_wue is not None:
        wue_anomaly = not (wue_pred["p10"] <= req.submitted_wue <= wue_pred["p90"])

    return {
        "predicted_pue": pue_pred,
        "predicted_wue": wue_pred,
        "anomaly_detected": pue_anomaly or wue_anomaly,
        "flags": {
            "pue_out_of_range": pue_anomaly,
            "wue_out_of_range": wue_anomaly,
        },
        "requires_evidence": pue_anomaly or wue_anomaly,
    }
