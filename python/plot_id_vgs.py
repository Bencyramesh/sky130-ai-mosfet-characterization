import numpy as np
import matplotlib.pyplot as plt

# Read ngspice data
data = np.loadtxt("data/id_vgs.dat", skiprows=1)

# Column 1 = VGS
vgs = data[:, 0]

# Column 3 = drain current
# Negative sign is because of ngspice current direction
id_current = -data[:, 2]

# Convert A → mA
id_mA = id_current * 1000

# Plot
plt.figure(figsize=(8, 5))

plt.plot(vgs, id_mA, marker="o", markersize=3)

plt.xlabel("Gate-Source Voltage VGS (V)")
plt.ylabel("Drain Current ID (mA)")
plt.title("NMOS ID-VGS Characteristics")

plt.grid(True)

# Save graph
plt.savefig("graphs/id_vgs.png", dpi=300, bbox_inches="tight")

print("Graph successfully saved as graphs/id_vgs.png")
