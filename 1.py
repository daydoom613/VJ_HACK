# ✅ Step 1: Preprocessing Dataset

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1️⃣ Load dataset
df = pd.read_csv("dataset.csv")

# 2️⃣ Encode 'crop' using LabelEncoder
crop_encoder = LabelEncoder()
df['crop'] = crop_encoder.fit_transform(df['crop'])

# 3️⃣ Encode 'fertilizer_name' using LabelEncoder
fertilizer_encoder = LabelEncoder()
df['fertilizer_name'] = fertilizer_encoder.fit_transform(df['fertilizer_name'])

# 4️⃣ Scale numerical features
num_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall', 'moisture']
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# ✅ Final preprocessed dataset
print("✅ Preprocessing complete!")
print(df.head())

# (Optional) Save preprocessed dataset
df.to_csv("preprocessed_fertilizer_dataset.csv", index=False)

