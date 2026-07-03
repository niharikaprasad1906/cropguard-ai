import pandas as pd
import numpy as np
import json
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Generating synthetic dataset...")
np.random.seed(42)

with open("models/crop_list.json", "r") as f:
    crop_list = json.load(f)
with open("models/country_list.json", "r") as f:
    country_list = json.load(f)

# Fit LabelEncoders
le_crop = LabelEncoder()
le_crop.fit(crop_list)

le_country = LabelEncoder()
le_country.fit(country_list)

# Generate synthetic data
n_samples = 20000

# Random choices
crops = np.random.choice(crop_list, n_samples)
countries = np.random.choice(country_list, n_samples)

crop_enc = le_crop.transform(crops)
country_enc = le_country.transform(countries)
year = np.random.randint(1990, 2025, n_samples)
rainfall = np.random.uniform(200, 3000, n_samples)
avg_temp = np.random.uniform(10, 35, n_samples)
pesticides = np.random.uniform(0, 100, n_samples)
area = np.random.uniform(1, 100, n_samples)

# Synthetic target: Yield (tonnes/hectare)
yield_base = 2.0 + (crop_enc * 0.1) + (country_enc * 0.05)
yield_rain = np.where((rainfall > 500) & (rainfall < 1500), 1.0, -0.5)
yield_temp = np.where((avg_temp > 18) & (avg_temp < 28), 0.5, -0.5)
yield_pest = np.log1p(pesticides) * 0.2
yield_trend = (year - 1990) * 0.02
noise = np.random.normal(0, 0.5, n_samples)

target_yield = yield_base + yield_rain + yield_temp + yield_pest + yield_trend + noise
target_yield = np.clip(target_yield, 0.1, 15.0)

df = pd.DataFrame({
    "crop_enc": crop_enc,
    "country_enc": country_enc,
    "year": year,
    "rainfall": rainfall,
    "avg_temp": avg_temp,
    "pesticides": pesticides,
    "area": area,
    "target": target_yield
})

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest model (200 trees)...")
model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, preds) ** 0.5)
print("R2:", r2_score(y_test, preds))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/yield_model.pkl")
joblib.dump(X.columns.tolist(), "models/yield_columns.pkl")
joblib.dump(le_crop, "models/yield_crop_encoder.pkl")
joblib.dump(le_country, "models/yield_country_encoder.pkl")
print("Saved yield model and encoders to models/")