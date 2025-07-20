#!/bin/bash

# Exit immediately if a command exits with a non-zero status:
set -e

# Snellius cluster already has many of the required libraries as modules.
# It is the easiest to install to the user the things that are not already available.

# Must load the modules before running this script:
echo "--- Loading necessary modules ---"
module purge
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
echo "--- Modules loaded successfully ---"


echo "--- Installing Python (3.11.3) packages ---"
echo "Version: $(python --version)"

# Python version: 3.11.3
pip install --user setuptools==65.5.0
pip install --user wheel==0.45.1

# PyTorch Geometric
pip install --user torch_geometric -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

# Triton (compatible with PyTorch 2.1.2)
pip install --user triton==2.1.0

# RDF and graph processing
pip install --user rdflib==7.1.4

# Data handling and visualization
pip install --user pandas==2.2.3
pip install --user seaborn==0.13.2
pip install --user plotly==6.1.0
pip install --user kaleido==1.0.0
pip install --user pyarrow==20.0.0

# Data specific utilities
pip install --user holidays==0.72 # timeseries-related
pip install --user shapely==2.1.0 # working with polygons

# Tabular ML
pip install --user lightgbm==4.6.0

# Hyperparameter tuning
pip install --user optuna==4.3.0

echo "--- Python packages installed successfully ---"

echo "Environment setup complete."