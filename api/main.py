from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import joblib
import json
import os


app = FastAPI(title="Fertilizer Recommendation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    N: float = Field(...)
    P: float = Field(...)
    K: float = Field(...)
    temperature: float = Field(...)
    humidity: float = Field(...)
    pH: float = Field(...)
    rainfall: float = Field(...)
    moisture: float = Field(...)
    crop: str = Field(..., description="Crop name or numeric code as string")


_MODEL = None
_FEATURE_ORDER = None
_CROP_MAPPING = None
_LABEL_MAPPING = None
_SCALER = None
_CROP_ENCODER = None
_FERTILIZER_ENCODER = None
_ARTIFACTS_DIR = None


def _load_artifacts():
    global _MODEL, _FEATURE_ORDER, _CROP_MAPPING, _LABEL_MAPPING, _ARTIFACTS_DIR

    # Search in: project root (..), api dir (.), and current working directory
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        os.path.abspath(os.path.dirname(__file__)),
        os.path.abspath(os.getcwd()),
    ]

    artifacts_dir = None
    for cand in candidates:
        if os.path.exists(os.path.join(cand, "xgb_model.joblib")):
            artifacts_dir = cand
            break

    if artifacts_dir is None:
        searched = " | ".join(candidates)
        raise RuntimeError(
            "Model artifact not found: xgb_model.joblib. "
            "Please run the 'Save trained XGBoost model and metadata' cell in the notebook "
            "to generate artifacts in the project root, or place them next to api/main.py. "
            f"Searched: {searched}"
        )

    _ARTIFACTS_DIR = artifacts_dir

    model_path = os.path.join(artifacts_dir, "xgb_model.joblib")
    feature_order_path = os.path.join(artifacts_dir, "feature_order.json")
    crop_mapping_path = os.path.join(artifacts_dir, "crop_mapping.json")
    label_mapping_path = os.path.join(artifacts_dir, "label_mapping.json")
    scaler_path = os.path.join(artifacts_dir, "scaler.joblib")
    crop_encoder_path = os.path.join(artifacts_dir, "crop_encoder.joblib")
    fertilizer_encoder_path = os.path.join(artifacts_dir, "fertilizer_encoder.joblib")

    _MODEL = joblib.load(model_path)

    with open(feature_order_path, "r", encoding="utf-8") as f:
        _FEATURE_ORDER = json.load(f)

    with open(crop_mapping_path, "r", encoding="utf-8") as f:
        _CROP_MAPPING = json.load(f)

    with open(label_mapping_path, "r", encoding="utf-8") as f:
        raw_label_mapping = json.load(f)
        # Coerce keys to int in case JSON stored them as strings
        _LABEL_MAPPING = {int(k): v for k, v in raw_label_mapping.items()}

    # Optional: load scaler and encoders if present
    global _SCALER, _CROP_ENCODER, _FERTILIZER_ENCODER
    if os.path.exists(scaler_path):
        _SCALER = joblib.load(scaler_path)
    if os.path.exists(crop_encoder_path):
        _CROP_ENCODER = joblib.load(crop_encoder_path)
    if os.path.exists(fertilizer_encoder_path):
        _FERTILIZER_ENCODER = joblib.load(fertilizer_encoder_path)


@app.on_event("startup")
def startup_event():
    _load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata():
    return {
        "feature_order": _FEATURE_ORDER,
        "crops": list(_CROP_MAPPING.keys()),
        "label_mapping": _LABEL_MAPPING,
        "artifacts_dir": _ARTIFACTS_DIR,
        "uses_preprocessor": bool(_SCALER and _CROP_ENCODER),
    }


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        crop_key = str(req.crop)
        if crop_key in _CROP_MAPPING:
            crop_code = _CROP_MAPPING[crop_key]
        else:
            # Attempt numeric conversion fallback
            try:
                crop_code = int(float(crop_key))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Unknown crop: {req.crop}") from exc

        feature_values = {
            "N": req.N,
            "P": req.P,
            "K": req.K,
            "temperature": req.temperature,
            "humidity": req.humidity,
            "pH": req.pH,
            "rainfall": req.rainfall,
            "moisture": req.moisture,
            "crop": crop_code,
        }

        ordered = [feature_values[name] for name in _FEATURE_ORDER]
        X = np.array([ordered], dtype=float)

        # If scaler/encoders available, apply the exact training-time preprocessing
        if _SCALER is not None and _CROP_ENCODER is not None:
            # numeric first, then crop encoded to match training pipeline
            X_num = X[:, :-1]
            X_num_scaled = _SCALER.transform(X_num)
            # Crop already encoded as integer above
            X = np.hstack([X_num_scaled, X[:, -1].reshape(-1, 1)])

        pred_class = int(_MODEL.predict(X)[0])

        # Prefer fertilizer encoder for decoding if available
        if _FERTILIZER_ENCODER is not None:
            try:
                fertilizer = _FERTILIZER_ENCODER.inverse_transform([pred_class])[0]
            except Exception:
                fertilizer = _LABEL_MAPPING.get(pred_class, str(pred_class))
        else:
            fertilizer = _LABEL_MAPPING.get(pred_class, str(pred_class))
        return {"fertilizer": fertilizer, "predicted_class": pred_class}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000

