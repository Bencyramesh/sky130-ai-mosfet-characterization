import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------------------
# LOAD SKY130 DATASET
# -----------------------------------------

df = pd.read_csv(
    "data/sky130_pvt_dataset.csv"
)

print("--------------------------------")
print("SKY130 PVT DATASET")
print("--------------------------------")

print(df.head())
print("\nTotal samples:", len(df))


# -----------------------------------------
# INPUTS AND OUTPUT
# -----------------------------------------

X = df[
    [
        "Corner",
        "Temperature_C",
        "VDS_V"
    ]
]

y = df["Vth_V"]


# -----------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# -----------------------------------------
# PROCESS CORNER ENCODING
# -----------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "corner",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            ["Corner"]
        )
    ],
    remainder="passthrough"
)


# -----------------------------------------
# RANDOM FOREST MODEL
# -----------------------------------------

rf_model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)


# -----------------------------------------
# COMPLETE ML PIPELINE
# -----------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", rf_model)
    ]
)


# -----------------------------------------
# TRAIN
# -----------------------------------------

print("\nTraining SKY130 AI model...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# -----------------------------------------
# TEST
# -----------------------------------------

predictions = model.predict(
    X_test
)


# -----------------------------------------
# METRICS
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


print("\n--------------------------------")
print("MODEL PERFORMANCE")
print("--------------------------------")

print(f"MAE  = {mae:.6f} V")
print(f"RMSE = {rmse:.6f} V")
print(f"R2   = {r2:.6f}")


# -----------------------------------------
# SAVE MODEL
# -----------------------------------------

joblib.dump(
    model,
    "models/sky130_vth_model.pkl"
)


# -----------------------------------------
# SAVE METRICS
# -----------------------------------------

metrics = {
    "MAE_V": float(mae),
    "RMSE_V": float(rmse),
    "R2": float(r2)
}

with open(
    "models/sky130_vth_metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# -----------------------------------------
# ACTUAL VS PREDICTED GRAPH
# -----------------------------------------

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

plt.xlabel(
    "Actual SKY130 VTH (V)"
)

plt.ylabel(
    "AI Predicted VTH (V)"
)

plt.title(
    "SKY130 VTH: ngspice vs AI"
)

plt.grid(True)

plt.savefig(
    "graphs/sky130_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -----------------------------------------
# EXAMPLE PREDICTION
# -----------------------------------------

example = pd.DataFrame({
    "Corner": ["tt"],
    "Temperature_C": [75],
    "VDS_V": [1.2]
})

example_prediction = model.predict(
    example
)[0]

print("\nExample:")
print("Corner      = TT")
print("Temperature = 75 C")
print("VDS         = 1.2 V")
print(
    f"Predicted VTH = "
    f"{example_prediction:.6f} V"
)

print("\nFiles created:")
print("models/sky130_vth_model.pkl")
print("models/sky130_vth_metrics.json")
print("graphs/sky130_actual_vs_predicted.png")
