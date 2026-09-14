import pandas as pd
import matplotlib.pyplot as plt

# Load SKY130 PVT dataset
df = pd.read_csv("data/sky130_pvt_dataset.csv")

print("Dataset loaded successfully")
print("Total samples:", len(df))
print(df.head())

# --------------------------------------------------
# GRAPH 1: VTH vs Temperature
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for corner in df["Corner"].unique():

    corner_data = df[
        (df["Corner"] == corner) &
        (df["VDS_V"] == 1.2)
    ]

    plt.plot(
        corner_data["Temperature_C"],
        corner_data["Vth_V"],
        marker="o",
        label=corner.upper()
    )

plt.xlabel("Temperature (°C)")
plt.ylabel("Threshold Voltage VTH (V)")
plt.title("SKY130 NMOS VTH vs Temperature")
plt.grid(True)
plt.legend()

plt.savefig(
    "graphs/sky130_vth_vs_temperature.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# --------------------------------------------------
# GRAPH 2: VTH vs VDS
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for corner in df["Corner"].unique():

    corner_data = df[
        (df["Corner"] == corner) &
        (df["Temperature_C"] == 25)
    ]

    plt.plot(
        corner_data["VDS_V"],
        corner_data["Vth_V"],
        marker="o",
        label=corner.upper()
    )

plt.xlabel("Drain Source Voltage VDS (V)")
plt.ylabel("Threshold Voltage VTH (V)")
plt.title("SKY130 NMOS VTH vs VDS")
plt.grid(True)
plt.legend()

plt.savefig(
    "graphs/sky130_vth_vs_vds.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print()
print("Graphs created:")
print("graphs/sky130_vth_vs_temperature.png")
print("graphs/sky130_vth_vs_vds.png")
