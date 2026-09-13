import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# -----------------------------------------
# 1. Create folders
# -----------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
os.makedirs("data", exist_ok=True)

# -----------------------------------------
# 2. Load advanced semiconductor dataset
# -----------------------------------------

df = pd.read_csv(
    "data/mosfet_advanced_dataset.csv"
)

print("----------------------------------------")
print("ADVANCED MOSFET DATASET")
print("----------------------------------------")

print(df.head())

print("\nTotal samples:", len(df))

# -----------------------------------------
# 3. Define AI inputs and output
# -----------------------------------------

features = [
    "VGS_V",
    "VDS_V",
    "Temperature_C",
    "Width_um",
    "Length_um"
]

X = df[features]

y = df["ID_mA"]

print("\nAI INPUT FEATURES:")

for feature in features:
    print("-", feature)

print("\nAI OUTPUT:")
print("- ID_mA")

# -----------------------------------------
# 4. Split data
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# -----------------------------------------
# 5. Create Random Forest model
# -----------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

# -----------------------------------------
# 6. Train model
# -----------------------------------------

print("\nTraining AI model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")

# -----------------------------------------
# 7. Predict test data
# -----------------------------------------

predictions = model.predict(
    X_test
)

# -----------------------------------------
# 8. Calculate metrics
# -----------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n----------------------------------------")
print("MODEL PERFORMANCE")
print("----------------------------------------")

print(f"MAE  = {mae:.6f} mA")
print(f"RMSE = {rmse:.6f} mA")
print(f"R2   = {r2:.6f}")

# -----------------------------------------
# 9. Save trained model
# -----------------------------------------

model_path = "models/mosfet_advanced_rf.pkl"

joblib.dump(
    model,
    model_path
)

print("\nModel saved:")
print(model_path)

# -----------------------------------------
# 10. Save metrics for dashboard
# -----------------------------------------

metrics = {
    "MAE_mA": float(mae),
    "RMSE_mA": float(rmse),
    "R2": float(r2)
}

with open(
    "models/model_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

# -----------------------------------------
# 11. Save test predictions
# -----------------------------------------

results = X_test.copy()

results["Actual_ID_mA"] = y_test.values
results["Predicted_ID_mA"] = predictions

results.to_csv(
    "data/advanced_test_predictions.csv",
    index=False
)

# -----------------------------------------
# 12. Actual vs Predicted graph
# -----------------------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.7
)

minimum = min(
    y_test.min(),
    predictions.min()
)

maximum = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel(
    "Actual ID from ngspice (mA)"
)

plt.ylabel(
    "AI Predicted ID (mA)"
)

plt.title(
    "Advanced MOSFET Model\n"
    "Actual vs AI Prediction"
)

plt.grid(True)

plt.savefig(
    "graphs/advanced_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------
# 13. Feature importance
# -----------------------------------------

importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\n----------------------------------------")
print("FEATURE IMPORTANCE")
print("----------------------------------------")

print(importance_df)

importance_df.to_csv(
    "data/feature_importance.csv",
    index=False
)

plt.figure(figsize=(8, 5))

plt.bar(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Input Parameter")
plt.ylabel("Importance")

plt.title(
    "MOSFET AI Model Feature Importance"
)

plt.xticks(rotation=30)

plt.grid(
    axis="y"
)

plt.savefig(
    "graphs/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# -----------------------------------------
# 14. Example prediction
# -----------------------------------------

example = pd.DataFrame({
    "VGS_V": [1.4],
    "VDS_V": [1.2],
    "Temperature_C": [75],
    "Width_um": [5.0],
    "Length_um": [0.5]
})

predicted_id = model.predict(
    example
)[0]

print("\n----------------------------------------")
print("EXAMPLE AI PREDICTION")
print("----------------------------------------")

print("VGS         = 1.4 V")
print("VDS         = 1.2 V")
print("Temperature = 75 C")
print("Width       = 5.0 um")
print("Length      = 0.5 um")

print(
    f"\nPredicted ID = "
    f"{predicted_id:.6f} mA"
)

print("\n----------------------------------------")
print("FILES CREATED")
print("----------------------------------------")

print(
    "models/mosfet_advanced_rf.pkl"
)

print(
    "models/model_metrics.json"
)

print(
    "data/advanced_test_predictions.csv"
)

print(
    "data/feature_importance.csv"
)

print(
    "graphs/advanced_actual_vs_predicted.png"
)

print(
    "graphs/feature_importance.png"
)
