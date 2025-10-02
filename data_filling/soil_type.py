#Feature relevance analysis for Soil Type prediction using Random Forest Classifier
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

fertilizer_df = pd.read_csv("Fertilizer Prediction.csv")

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

# Encode target
le_target = LabelEncoder()
fertilizer_df["soil_type_encoded"] = le_target.fit_transform(fertilizer_df["soil_type"])

# Encode categorical features
categorical_cols = ["crop", "fertilizer_name"]  # add any other non-numeric columns
for col in categorical_cols:
    fertilizer_df[col] = LabelEncoder().fit_transform(fertilizer_df[col])

# Features and target
X = fertilizer_df.drop(columns=["soil_type", "soil_type_encoded"])
y = fertilizer_df["soil_type_encoded"]

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X, y)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("Feature relevance for predicting soil_type:\n")
print(importance)


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

fertilizer_df = pd.read_csv("Fertilizer Prediction.csv")
merged_df = pd.read_csv("merged_crop_fertilizer.csv")

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

# Encode target
le_soil = LabelEncoder()
fertilizer_df["soil_type_encoded"] = le_soil.fit_transform(fertilizer_df["soil_type"])

# Encode categorical features
le_crop = LabelEncoder()
fertilizer_df["crop"] = le_crop.fit_transform(fertilizer_df["crop"])
merged_df["crop"] = le_crop.transform(merged_df["crop"])  # encode same as training

# Features and target
features = ["crop", "moisture", "N", "P"]
X_train = fertilizer_df[features]
y_train = fertilizer_df["soil_type_encoded"]

# Train classifier
clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)

# Predict missing soil_type
missing_mask = merged_df["soil_type"].isna()
X_missing = merged_df.loc[missing_mask, features]
predicted_soil = clf.predict(X_missing)

# Fill missing values and decode back to original soil_type
merged_df.loc[missing_mask, "soil_type"] = le_soil.inverse_transform(predicted_soil)

# Save updated dataset
merged_df.to_csv("merged_crop_fertilizer.csv", index=False)

print("✅ Missing soil_type values have been predicted and filled in merged_crop_fertilizer.csv")
