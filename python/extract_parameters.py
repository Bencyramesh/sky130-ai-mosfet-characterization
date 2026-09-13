import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# 1. Read ID-VGS simulation data
# ------------------------------------------------

data = np.loadtxt("data/id_vgs.dat", skiprows=1)

# First column = VGS
vgs = data[:, 0]

# Last column = drain current
id_current = -data[:, -1]

# ------------------------------------------------
# 2. Calculate transconductance gm
# gm = dID / dVGS
# ------------------------------------------------

gm = np.gradient(id_current, vgs)

# Maximum transconductance
gm_max_index = np.argmax(gm)

gm_max = gm[gm_max_index]
vgs_at_gm_max = vgs[gm_max_index]

# ------------------------------------------------
# 3. Estimate threshold voltage VTH
# Using sqrt(ID) vs VGS method
# ------------------------------------------------

# Only use region where drain current is clearly above zero
mask = id_current > 1e-6

vgs_fit = vgs[mask]
id_fit = id_current[mask]

sqrt_id = np.sqrt(id_fit)

# Linear fit:
# sqrt(ID) = m*VGS + c
m, c = np.polyfit(vgs_fit, sqrt_id, 1)

# At sqrt(ID)=0:
# VTH = -c/m
vth_estimated = -c / m

# ------------------------------------------------
# 4. Print results
# ------------------------------------------------

print("----------------------------------")
print("MOSFET CHARACTERIZATION RESULTS")
print("----------------------------------")

print(f"Estimated Threshold Voltage VTH = {vth_estimated:.3f} V")

print(
    f"Maximum Transconductance gm = "
    f"{gm_max * 1000:.3f} mS"
)

print(
    f"VGS at maximum gm = "
    f"{vgs_at_gm_max:.3f} V"
)

# ------------------------------------------------
# 5. Save results
# ------------------------------------------------

with open("data/mosfet_parameters.txt", "w") as f:

    f.write("MOSFET CHARACTERIZATION RESULTS\n")
    f.write("--------------------------------\n")

    f.write(
        f"Threshold Voltage VTH = "
        f"{vth_estimated:.3f} V\n"
    )

    f.write(
        f"Maximum gm = "
        f"{gm_max * 1000:.3f} mS\n"
    )

    f.write(
        f"VGS at maximum gm = "
        f"{vgs_at_gm_max:.3f} V\n"
    )

# ------------------------------------------------
# 6. Plot gm
# ------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(vgs, gm * 1000)

plt.xlabel("Gate-Source Voltage VGS (V)")
plt.ylabel("Transconductance gm (mS)")
plt.title("NMOS Transconductance vs VGS")

plt.grid(True)

plt.savefig(
    "graphs/gm_vgs.png",
    dpi=300,
    bbox_inches="tight"
)

print("----------------------------------")
print("Results saved:")
print("data/mosfet_parameters.txt")
print("graphs/gm_vgs.png")

