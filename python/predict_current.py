import joblib
import pandas as pd

# Load trained AI model
model = joblib.load("data/mosfet_random_forest.pkl")

print("----------------------------------")
print("AI MOSFET DRAIN CURRENT PREDICTOR")
print("----------------------------------")

# Get values from user
vgs = float(input("Enter VGS (V): "))
vds = float(input("Enter VDS (V): "))

# Create input for model
input_data = pd.DataFrame({
    "VGS_V": [vgs],
    "VDS_V": [vds]
})

# AI prediction
predicted_id = model.predict(input_data)[0]

# Determine MOSFET region
VTH = 0.7

if vgs <= VTH:
    region = "OFF"

elif vds < (vgs - VTH):
    region = "TRIODE / LINEAR"

else:
    region = "SATURATION"

print("\n----------------------------------")
print("PREDICTION RESULT")
print("----------------------------------")

print(f"VGS = {vgs:.2f} V")
print(f"VDS = {vds:.2f} V")
print(f"MOSFET Region = {region}")

print(
    f"AI Predicted Drain Current ID = "
    f"{predicted_id:.6f} mA"
)
