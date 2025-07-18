#!/bin/bash

# =================================================================
# Wrapper script to submit multiple LGBM array jobs to SLURM.
# Each job will have a different configuration based on the
# parameters defined in the arrays below.
# =================================================================

echo "Starting submission of multiple LGBM jobs..."

# --- 1. DEFINE EXPERIMENT PARAMETERS ---
# Define the set of parameters for each run.
# Make sure the arrays have the same number of elements!

# Suffix for the output folder, e.g., M1d, M2d
FOLDER_SUFFIXES=("T1d" "T2d" "T3d" "T4d" "T5d" "T6d" "T7d" "T8d")

# Forecast horizons for each experiment, e.g., 1, 2
FORECAST_HORIZONS=(1 2 3 4 5 6 7 8)

# Other parameters can be added here in the same way.
# For this example, we'll keep the other arguments constant.
PREDICTION_TYPE="delta"
MEASUREMENT_VARIABLE="Temperature"

# --- 2. LOOP THROUGH PARAMETERS AND SUBMIT JOBS ---
# Get the total number of experiments to run.
num_experiments=${#FOLDER_SUFFIXES[@]}

# Loop from 0 to num_experiments - 1
for (( i=0; i<${num_experiments}; i++ )); do
    # Get the parameters for the current iteration
    suffix=${FOLDER_SUFFIXES[i]}
    horizon=${FORECAST_HORIZONS[i]}

    echo "----------------------------------------------------"
    echo "Submitting job $((i+1)) of ${num_experiments}"
    echo "Folder Suffix: ${suffix}"
    echo "Forecast Horizon: ${horizon}"
    echo "----------------------------------------------------"

    # Construct and execute the sbatch command
    sbatch scripts/run_experiment_LGBM.job \
        --folder_suffix "${suffix}" \
        --forecast_horizons "${horizon}" \
        --prediction_type "${PREDICTION_TYPE}" \
        --measurement_variable "${MEASUREMENT_VARIABLE}"
    
    # Add a small delay to avoid overwhelming the SLURM controller
    sleep 1
done

echo "All jobs submitted successfully."