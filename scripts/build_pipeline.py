import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


NUM_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall', 'moisture']


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw_csv = os.path.join(project_root, "dataset.csv")
    if not os.path.exists(raw_csv):
        raise SystemExit(f"dataset.csv not found at {raw_csv}")

    df = pd.read_csv(raw_csv)

    # Split first to avoid leakage; fit encoders and scaler on train only
    y_raw = df['fertilizer_name']
    X_raw = df[['crop'] + NUM_COLS].copy()

    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_raw, y_raw, test_size=0.3, random_state=42, stratify=y_raw
    )
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp_raw, y_temp_raw, test_size=0.5, random_state=42, stratify=y_temp_raw
    )

    # Fit encoders on train
    crop_encoder = LabelEncoder()
    crop_encoder.fit(X_train_raw['crop'])

    fertilizer_encoder = LabelEncoder()
    fertilizer_encoder.fit(y_train_raw)

    # Fit scaler on train numeric features
    scaler = StandardScaler()
    scaler.fit(X_train_raw[NUM_COLS])

    # Transform splits
    def transform_block(X_block, y_block):
        crop_enc = crop_encoder.transform(X_block['crop'])
        X_num_scaled = scaler.transform(X_block[NUM_COLS])
        X_final = np.hstack([X_num_scaled, crop_enc.reshape(-1, 1)])
        y_enc = fertilizer_encoder.transform(y_block)
        return X_final, y_enc

    X_train, y_train = transform_block(X_train_raw, y_train_raw)
    X_val, y_val = transform_block(X_val_raw, y_val_raw)
    X_test, y_test = transform_block(X_test_raw, y_test_raw)

    # Train XGBoost on preprocessed features
    model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    # Persist artifacts
    joblib.dump(model, os.path.join(project_root, "xgb_model.joblib"))
    joblib.dump(scaler, os.path.join(project_root, "scaler.joblib"))
    joblib.dump(crop_encoder, os.path.join(project_root, "crop_encoder.joblib"))
    joblib.dump(fertilizer_encoder, os.path.join(project_root, "fertilizer_encoder.joblib"))

    # Also persist feature order for reference (scaled num cols + crop_enc)
    with open(os.path.join(project_root, "feature_order.json"), "w", encoding="utf-8") as f:
        json.dump(NUM_COLS + ["crop"], f)

    # Expose readable label mapping for convenience
    label_mapping = {int(i): cls for i, cls in enumerate(fertilizer_encoder.classes_)}
    with open(os.path.join(project_root, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_mapping, f)

    # Expose crop classes for frontend dropdown (by name)
    with open(os.path.join(project_root, "crop_mapping.json"), "w", encoding="utf-8") as f:
        json.dump({name: int(idx) for idx, name in enumerate(crop_encoder.classes_)}, f)

    print("Artifacts saved to project root:")
    for fn in [
        "xgb_model.joblib", "scaler.joblib", "crop_encoder.joblib", "fertilizer_encoder.joblib",
        "feature_order.json", "label_mapping.json", "crop_mapping.json"
    ]:
        print(os.path.join(project_root, fn))


if __name__ == "__main__":
    main()


