import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/raw/yield/crop_production.csv"
)

print("Columns:", df.columns)

# ==========================================
# CLEAN DATA
# ==========================================

# Remove missing target values
df = df.dropna(subset=["Production"])

# Fill missing feature values
df = df.ffill()

# ==========================================
# FEATURES + TARGET
# ==========================================

X = df.drop("Production", axis=1)

# Remove heavy categorical column
if "District_Name" in X.columns:
    X = X.drop("District_Name", axis=1)

y = df["Production"]

# ==========================================
# ENCODE CATEGORICALS
# ==========================================

X = pd.get_dummies(X)

# ==========================================
# SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# MODEL
# ==========================================

print("\nTraining yield model...\n")

model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# PREDICT
# ==========================================

predictions = model.predict(X_test)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)

print("\n===== RESULTS =====")

print(f"RMSE: {rmse:.2f}")

print(f"R2 Score: {r2:.4f}")

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/yield_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "models/yield_columns.pkl"
)

print("\nYield model saved successfully!")