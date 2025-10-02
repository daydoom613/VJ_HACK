import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(project_root, "preprocessed_fertilizer_dataset.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Expect these columns to exist
    feature_order = [
        "N", "P", "K", "temperature", "humidity", "pH", "rainfall", "moisture", "crop"
    ]
    target_col = "fertilizer_name"

    missing = [c for c in feature_order + [target_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in dataset: {missing}")

    X = df[feature_order].copy()
    y = df[target_col].copy()

    # Stratify by target for balanced classes
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    # Save splits for reference (optional but useful)
    X_train.to_csv(os.path.join(project_root, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(project_root, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(project_root, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(project_root, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(project_root, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(project_root, "y_test.csv"), index=False)

    # Train XGBoost (default reasonable params)
    model = XGBClassifier(eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train.values.ravel())

    # Build mappings
    # Crop mapping: expect numeric codes; expose as string keys for frontend dropdown
    unique_crops = sorted(int(v) for v in pd.concat([X_train["crop"], X_val["crop"], X_test["crop"]]).unique())
    crop_mapping = {str(c): int(c) for c in unique_crops}

    # Label mapping: class id -> label string (here just the numeric class as string)
    if hasattr(model, "classes_"):
        unique_labels = [int(c) for c in model.classes_]
    else:
        unique_labels = sorted(int(v) for v in pd.concat([y_train, y_val, y_test]).unique())
    label_mapping = {int(c): str(int(c)) for c in unique_labels}

    # Persist artifacts in project root
    joblib.dump(model, os.path.join(project_root, "xgb_model.joblib"))
    with open(os.path.join(project_root, "feature_order.json"), "w", encoding="utf-8") as f:
        json.dump(feature_order, f)
    with open(os.path.join(project_root, "crop_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(crop_mapping, f)
    with open(os.path.join(project_root, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_mapping, f)

    print("Saved artifacts to:")
    print(os.path.join(project_root, "xgb_model.joblib"))
    print(os.path.join(project_root, "feature_order.json"))
    print(os.path.join(project_root, "crop_mapping.json"))
    print(os.path.join(project_root, "label_mapping.json"))


if __name__ == "__main__":
    main()


