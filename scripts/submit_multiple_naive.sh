#!/bin/bash

# =================================================================
# Wrapper script to submit multiple Naive-Persistence array jobs to SLURM.
# It loops through different measurement variables and forecast horizons.
# =================================================================

echo "Starting submission of multiple Naive-Persistence array jobs..."

# --- 1. DEFINE EXPERIMENT PARAMETERS ---
# Define the full variable names and their corresponding short prefixes for the suffix
MEASUREMENT_VARS=("Temperature" "CO2Level" "Humidity")
MEASUREMENT_PREFIXES=("T" "C" "H")

# Define the forecast horizons to test
FORECAST_HORIZONS=(1 2 3 4 5 6 7 8)

# --- 2. LOOP THROUGH PARAMETERS AND SUBMIT JOBS ---
# Outer loop for measurement variables
for j in "${!MEASUREMENT_VARS[@]}"; do
    variable=${MEASUREMENT_VARS[j]}
    prefix=${MEASUREMENT_PREFIXES[j]}

    # Inner loop for forecast horizons
    for horizon in "${FORECAST_HORIZONS[@]}"; do
        # Construct the folder suffix dynamically (e.g., T1, C1, H1, etc.)
        suffix="${prefix}${horizon}"

        echo "----------------------------------------------------"
        echo "Submitting Job:"
        echo "  Measurement Variable: ${variable}"
        echo "  Forecast Horizon:     ${horizon}"
        echo "  Folder Suffix:        ${suffix}"
        echo "----------------------------------------------------"

        sbatch scripts/run_experiment_naive.job \
            --folder_suffix "${suffix}" \
            --forecast_horizons "${horizon}" \
            --measurement_variable "${variable}"

        # Add a small delay to avoid overwhelming the SLURM controller
        sleep 1
    done
done

echo "All jobs submitted successfully."