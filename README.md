# AI-Driven MOSFET PVT Characterization using SKY130

AI-assisted semiconductor device characterization using the SKY130 open-source PDK, ngspice, Python, and machine learning.

## Project Overview

This project automates MOSFET characterization across different process, voltage, and temperature conditions using SKY130 transistor models.

ngspice is used as the reference semiconductor simulator. Python automatically runs simulations, extracts device parameters, generates datasets, and trains machine-learning models for rapid MOSFET parameter prediction.

## Current Characterization

The project evaluates the SKY130 NMOS device across:

- Process corners: TT, FF, SS, FS, SF
- Multiple temperatures
- Multiple VDS operating points
- Automatic threshold-voltage extraction

The generated semiconductor data is used to train an AI surrogate model.

## Project Flow

```text
SKY130 PDK
    ↓
MOSFET Device Model
    ↓
ngspice Simulation
    ↓
PVT Sweep Automation
    ↓
Threshold Voltage Extraction
    ↓
Semiconductor Dataset
    ↓
Machine Learning
    ↓
AI Prediction
    ↓
SPICE Validation
    ↓
Interactive Dashboard
