import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import joblib


# -----------------------------------
# 1. Load semiconductor dataset
# -----------------------------------

df = pd.read_csv("data/mosfet_dataset.csv")

print("--------------------------------")
print("DATASET LOADED")
print("--------------------------------")

print(df.head())

print("\nTotal samples:", len(df))


# -----------------------------------
# 2. Define INPUT and OUTPUT
# -----------------------------------

# Inputs to AI
X = df[["VGS_V", "VDS_V"]]

# Output AI should predict
y = df["ID_mA"]


# -----------------------------------
# 3. Split training and testing data
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# -----------------------------------
# 4. Create ML model
# -----------------------------------

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)


# -----------------------------------
# 5. Train AI
# -----------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


# -----------------------------------
# 6. Predict test data
# -----------------------------------

predictions = model.predict(X_test)


# -----------------------------------
# 7. Calculate accuracy metrics
# -----------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


print("\n--------------------------------")
print("MODEL PERFORMANCE")
print("--------------------------------")

print(f"MAE  = {mae:.6f} mA")
print(f"RMSE = {rmse:.6f} mA")
print(f"R2   = {r2:.4f}")


# -----------------------------------
# 8. Save trained model
# -----------------------------------

joblib.dump(
    model,
    "data/mosfet_random_forest.pkl"
)

print("\nModel saved as:")
print("data/mosfet_random_forest.pkl")


# -----------------------------------
# 9. Actual vs predicted graph
# -----------------------------------

plt.figure(figsize=(7, 6))

plt.scatter(
    y_test,
    predictions
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

plt.xlabel("Actual ID from ngspice (mA)")
plt.ylabel("AI Predicted ID (mA)")

plt.title(
    "MOSFET Drain Current\n"
    "Actual vs AI Prediction"
)

plt.grid(True)

plt.savefig(
    "graphs/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

print("Graph saved as:")
print("graphs/actual_vs_predicted.png")


# -----------------------------------
# 10. Example prediction
# -----------------------------------

example = pd.DataFrame({
    "VGS_V": [1.4],
    "VDS_V": [1.2]
})

predicted_id = model.predict(example)[0]

print("\n--------------------------------")
print("EXAMPLE AI PREDICTION")
print("--------------------------------")

print("VGS = 1.4 V")
print("VDS = 1.2 V")

print(
    f"Predicted Drain Current ID = "
    f"{predicted_id:.6f} mA"
)
