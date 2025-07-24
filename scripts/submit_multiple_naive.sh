#!/bin/bash

# =================================================================
# Wrapper script to submit multiple Naive-Persistence array jobs to SLURM.
# Each job will have a different configuration based on the
# parameters defined in the arrays below.
# =================================================================

echo "Starting submission of multiple Naive-Persistence array jobs..."

# --- 1. DEFINE EXPERIMENT PARAMETERS ---
MEASUREMENT_VARIABLE="Temperature"
FORECAST_HORIZONS=(1 2 3 4 5 6 7 8)
FOLDER_SUFFIXES=("T1" "T2" "T3" "T4" "T5" "T6" "T7" "T8")

# --- 2. LOOP THROUGH PARAMETERS AND SUBMIT JOBS ---
num_experiments=${#FOLDER_SUFFIXES[@]}

for (( i=0; i<${num_experiments}; i++ )); do
    suffix=${FOLDER_SUFFIXES[i]}
    horizon=${FORECAST_HORIZONS[i]}
    
    echo "----------------------------------------------------"
    echo "Submitting job $((i+1)) of ${num_experiments}"
    echo "Folder Suffix: ${suffix}"
    echo "Forecast Horizon: ${horizon}"
    echo "----------------------------------------------------"
    
    sbatch scripts/run_experiment_naive.job \
        --folder_suffix "${suffix}" \
        --forecast_horizons "${horizon}" \
        --measurement_variable "${MEASUREMENT_VARIABLE}"
    
    # Add a small delay to avoid overwhelming the SLURM controller
    sleep 1
done

echo "All jobs submitted successfully."