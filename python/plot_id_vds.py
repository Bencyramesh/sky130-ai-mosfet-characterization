import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os

# Different gate voltages we want to test
vgs_values = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8]

os.makedirs("data/id_vds", exist_ok=True)

plt.figure(figsize=(8, 5))

for vgs in vgs_values:

    spice_file = f"""
* NMOS ID-VDS simulation

Vd drain 0 0
Vg gate 0 {vgs}

M1 drain gate 0 0 NMOS_MODEL W=10u L=1u

.model NMOS_MODEL NMOS (
+ LEVEL=1
+ VTO=0.7
+ KP=200u
+ LAMBDA=0.02
+ )

.dc Vd 0 1.8 0.05

.control
run
set wr_singlescale
set wr_vecnames
wrdata data/id_vds/vgs_{vgs}.dat v(drain) i(Vd)
.endc

.end
"""

    # Create temporary SPICE file
    with open("spice/temp_id_vds.cir", "w") as f:
        f.write(spice_file)

    # Run ngspice automatically
    subprocess.run(
        ["ngspice", "-b", "spice/temp_id_vds.cir"],
        stdout=subprocess.DEVNULL
    )

    # Read simulation result
    data = np.loadtxt(
        f"data/id_vds/vgs_{vgs}.dat",
        skiprows=1
    )

    # First column = VDS
    vds = data[:, 0]

    # Last column = drain current
    id_current = -data[:, -1]

    # Convert A -> mA
    id_mA = id_current * 1000

    plt.plot(vds, id_mA, label=f"VGS = {vgs} V")


plt.xlabel("Drain-Source Voltage VDS (V)")
plt.ylabel("Drain Current ID (mA)")
plt.title("NMOS ID-VDS Characteristics")

plt.grid(True)
plt.legend()

plt.savefig(
    "graphs/id_vds.png",
    dpi=300,
    bbox_inches="tight"
)

print("ID-VDS simulation completed.")
print("Graph saved as graphs/id_vds.png")
