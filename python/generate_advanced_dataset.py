import subprocess
import numpy as np
import pandas as pd
import os

# ---------------------------------------------------
# Semiconductor parameters we want to vary
# ---------------------------------------------------

temperatures = [0, 27, 75, 125]

widths = [
    (2.0, "2u"),
    (5.0, "5u"),
    (10.0, "10u")
]

lengths = [
    (0.18, "0.18u"),
    (0.5, "0.5u"),
    (1.0, "1u")
]

all_data = []

os.makedirs("data/advanced", exist_ok=True)

simulation_number = 0


# ---------------------------------------------------
# Run different W, L and Temperature combinations
# ---------------------------------------------------

for temp in temperatures:

    for width_um, width_spice in widths:

        for length_um, length_spice in lengths:

            simulation_number += 1

            print(
                f"Running simulation {simulation_number}: "
                f"T={temp}C, "
                f"W={width_um}um, "
                f"L={length_um}um"
            )

            output_file = (
                f"data/advanced/"
                f"sim_{simulation_number}.dat"
            )

            # -------------------------------------------
            # Create SPICE circuit automatically
            # -------------------------------------------

            spice_code = f"""
* Advanced MOSFET Dataset

.temp {temp}

Vd drain 0 0
Vg gate 0 0

M1 drain gate 0 0 NMOS_MODEL
+ W={width_spice}
+ L={length_spice}

.model NMOS_MODEL NMOS (
+ LEVEL=1
+ VTO=0.7
+ KP=200u
+ LAMBDA=0.02
+ )

* Sweep VGS and VDS
.dc Vg 0 1.8 0.2 Vd 0 1.8 0.2

.control

run

set wr_singlescale
set wr_vecnames

wrdata {output_file} v(gate) v(drain) i(Vd)

.endc

.end
"""

            temp_spice_file = "spice/temp_advanced.cir"

            with open(temp_spice_file, "w") as f:
                f.write(spice_code)

            # -------------------------------------------
            # Run ngspice
            # -------------------------------------------

            result = subprocess.run(
                [
                    "ngspice",
                    "-b",
                    temp_spice_file
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:

                print("ngspice error:")
                print(result.stderr)

                continue

            # -------------------------------------------
            # Read ngspice output
            # -------------------------------------------

            data = np.loadtxt(
                output_file,
                skiprows=1
            )

            # Last three columns:
            # VGS, VDS, current
            vgs = data[:, -3]
            vds = data[:, -2]

            # ngspice current direction is negative
            id_current = -data[:, -1]

            # Convert A to mA
            id_mA = id_current * 1000

            # -------------------------------------------
            # Store every operating point
            # -------------------------------------------

            for vg, vd, current in zip(
                vgs,
                vds,
                id_mA
            ):

                all_data.append({

                    "VGS_V": vg,

                    "VDS_V": vd,

                    "Temperature_C": temp,

                    "Width_um": width_um,

                    "Length_um": length_um,

                    "ID_mA": current
                })


# ---------------------------------------------------
# Create final dataset
# ---------------------------------------------------

df = pd.DataFrame(all_data)

df.to_csv(
    "data/mosfet_advanced_dataset.csv",
    index=False
)

print()
print("-------------------------------------")
print("ADVANCED MOSFET DATASET CREATED")
print("-------------------------------------")

print(df.head())

print()
print("Total samples:", len(df))

print()
print("Dataset saved as:")
print("data/mosfet_advanced_dataset.csv")
