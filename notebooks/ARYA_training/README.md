# ARYA Training Notebook

This folder contains a simplified training notebook demonstrating the
Agriculture Remotely-sensed Yield Algorithm (ARYA) workflow for maize yield
forecasting in Brazil.

## Contents

- ARYA_Brazil.ipynb
- Data/: input yield and administrative data
- Inputs/: figures used in the notebook

## Requirements

- openEO Python client
- pandas
- geopandas
- numpy
- matplotlib
- rasterio
- openeo
- scipy
- pathlib

## Purpose

Educational notebook demonstrating:
1. Data preparation
2. MODIS DVI signal extraction
3. Crop signal unmixing with WorldCereal maps
3. Gaussian fitting
4. Yield forecasting
