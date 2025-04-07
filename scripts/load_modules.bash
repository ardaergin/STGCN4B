#!/bin/bash

##### Script to load modules in Snellius #####

# Clear any previously loaded modules
module purge

# Load base environment
module load 2023

# Load Python and related modules
module load Python/3.11.3-GCCcore-12.3.0

# Load GPU modules
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Load PyTorch and related modules
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Load Python bundle
module load Python-bundle-PyPI/2023.06-GCCcore-12.3.0

# Load visualization modules
module load matplotlib/3.7.2-gfbf-2023a

# Load machine learning modules
module load scikit-learn/1.3.1-gfbf-2023a

# Display currently loaded modules
module list

echo "All modules loaded successfully!"
