#Feature relevance analysis for Moisture prediction using Random Forest Regression
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ================================
# Step 1: Load Fertilizer dataset
# ================================
fertilizer_df = pd.read_csv("Fertilizer Prediction.csv")

# Standardize column names (like before)
fertilizer_df = fertilizer_df.rename(columns={
    "Temparature": "temperature",
    "Humidity ": "humidity",
    "Moisture": "moisture",
    "Soil Type": "soil_type",
    "Crop Type": "crop",
    "Nitrogen": "N",
    "Phosphorous": "P",
    "Potassium": "K",
    "Fertilizer Name": "fertilizer_name"
})

# ================================
# Step 2: Prepare features
# ================================
X = fertilizer_df[["N", "P", "K", "temperature", "humidity", "fertilizer_name"]].copy()
y = fertilizer_df["moisture"]

# Encode categorical target column 'fertilizer_name'
X["fertilizer_name"] = LabelEncoder().fit_transform(X["fertilizer_name"].astype(str))

# ================================
# Step 3: Train regression model
# ================================
rf_reg = RandomForestRegressor(n_estimators=300, random_state=42)
rf_reg.fit(X, y)

# ================================
# Step 4: Feature importance
# ================================
importances = rf_reg.feature_importances_
features = X.columns

# Plot feature importance
plt.figure(figsize=(8,5))
plt.barh(features, importances, color="lightgreen")
plt.xlabel("Importance Score")
plt.title("Feature Importance for Moisture Prediction")
plt.show()

# Print ranked list
print("Feature Relevance for Moisture:")
for feat, score in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{feat}: {score:.4f}")




import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ================================
# Step 1: Load datasets
# ================================
fertilizer_df = pd.read_csv("Fertilizer Prediction.csv")
merged_df = pd.read_csv("merged_crop_fertilizer.csv")

# Standardize fertilizer dataset columns
fertilizer_df = fertilizer_df.rename(columns={
    "Temparature": "temperature",
    "Humidity ": "humidity",
    "Moisture": "moisture",
    "Soil Type": "soil_type",
    "Crop Type": "crop",
    "Nitrogen": "N",
    "Phosphorous": "P",
    "Potassium": "K",
    "Fertilizer Name": "fertilizer_name"
})

# ================================
# Step 2: Train regression model
# ================================
X_train = fertilizer_df[["N", "P", "humidity", "temperature"]].copy()
y_train = fertilizer_df["moisture"]

rf_reg = RandomForestRegressor(n_estimators=300, random_state=42)
rf_reg.fit(X_train, y_train)

# ================================
# Step 3: Predict missing moisture in merged dataset
# ================================
missing_mask = merged_df["moisture"].isna()

# Select rows with missing moisture
X_missing = merged_df.loc[missing_mask, ["N", "P", "humidity", "temperature"]]

# Predict values
predicted_moisture = rf_reg.predict(X_missing)

# Fill the missing moisture values
merged_df.loc[missing_mask, "moisture"] = predicted_moisture

# ================================
# Step 4: Save updated merged dataset
# ================================
merged_df.to_csv("merged_crop_fertilizer.csv", index=False)

print("✅ Missing moisture values have been predicted and filled in merged_crop_fertilizer.csv")
