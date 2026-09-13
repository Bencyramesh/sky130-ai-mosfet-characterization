import numpy as np
import pandas as pd

# Load ngspice output
data = np.loadtxt(
    "data/raw_dataset.dat",
    skiprows=1
)

# ngspice may include its sweep scale as the first column.
# The final three columns contain:
# VGS, VDS and I(Vd)

vgs = data[:, -3]
vds = data[:, -2]

# Reverse sign because of ngspice current convention
id_current = -data[:, -1]

# Convert A to mA
id_mA = id_current * 1000

# Threshold voltage from our MOSFET model
VTH = 0.7

regions = []

for vg, vd in zip(vgs, vds):

    if vg <= VTH:
        region = "OFF"

    elif vd < (vg - VTH):
        region = "TRIODE"

    else:
        region = "SATURATION"

    regions.append(region)


# Create dataset
df = pd.DataFrame({

    "VGS_V": vgs,
    "VDS_V": vds,
    "ID_mA": id_mA,
    "Region": regions

})

# Save CSV
df.to_csv(
    "data/mosfet_dataset.csv",
    index=False
)

print("--------------------------------")
print("MOSFET DATASET CREATED")
print("--------------------------------")

print(df.head(10))

print()
print("Total samples:", len(df))

print()
print("Region distribution:")
print(df["Region"].value_counts())

print()
print("Saved as:")
print("data/mosfet_dataset.csv")
