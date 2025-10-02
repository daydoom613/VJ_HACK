import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ================================
# Step 1: Load merged dataset
# ================================
df = pd.read_csv("merged_crop_fertilizer.csv")

# ================================
# Step 2: Prepare training data
# ================================
# Use only rows with fertilizer_name for training
fertilizer_only = df.dropna(subset=["fertilizer_name"]).copy()

# Features: N, P, K only
X_train = fertilizer_only[["N", "P", "K"]].copy()
y_train = fertilizer_only["fertilizer_name"]

# Encode target
fertilizer_encoder = LabelEncoder()
y_train_encoded = fertilizer_encoder.fit_transform(y_train.astype(str))

# ================================
# Step 3: Train model
# ================================
rf_model = RandomForestClassifier(n_estimators=300, random_state=42)
rf_model.fit(X_train, y_train_encoded)

# ================================
# Step 4: Predict missing fertilizer_name
# ================================
missing_mask = df["fertilizer_name"].isna()
X_unlabeled = df.loc[missing_mask, ["N", "P", "K"]]

# Predict and update in the same DataFrame
preds = rf_model.predict(X_unlabeled)
df.loc[missing_mask, "fertilizer_name"] = fertilizer_encoder.inverse_transform(preds)

# ================================
# Step 5: Save back to same file
# ================================
df.to_csv("merged_crop_fertilizer.csv", index=False)

print("✅ merged_crop_fertilizer.csv updated with filled fertilizer_name")
