import os
from ..config.args import parse_args 

def get_data_filename(args):
    """
    Generates a data filename based on the provided experiment arguments.

    Args:
        args: A configuration object (e.g., from argparse) containing all necessary parameters.
    """
    # 1. Determine file extension based on model family
    if args.model_family == "graph":
        extension = "pt"
    elif args.model_family == "tabular":
        extension = "joblib"
    else:
        raise ValueError(f"Invalid model_family for filename: {args.model_family}")

    # 2. Build filename components
    # Prefix
    if args.model_family == "graph":
        model_prefix = f"{args.graph_type}_{args.model_family}_input"
        model_inputs = f"{args.adjacency_type}_{args.gso_mode}"
    else: # tabular
        model_prefix = f"{args.model_family}_input"
        model_inputs = None # No extra input types for tabular

    # Suffix for weather
    weather_suffix = "WeatherOut" if args.skip_incorporating_weather else "WeatherIn"

    # Task-specific part
    if args.task_type == "workhour_classification":
        task_part = "floor7"
    elif args.task_type == "consumption_forecast":
        task_part = "consumption"
    elif args.task_type == "measurement_forecast":
        task_part = args.measurement_variable
    else:
        raise ValueError(f"Invalid task_type: {args.task_type}")

    # 3. Assemble the final filename
    # Using a list and filter to handle optional parts cleanly
    parts = [model_prefix, task_part, model_inputs, args.interval, weather_suffix]
    # Filter out any 'None' values (like the 'inputs' for tabular)
    valid_parts = [part for part in parts if part is not None]
    
    filename = f"{'_'.join(valid_parts)}.{extension}"
    
    return filename

if __name__ == "__main__":
    """
    Entry point for SLURM job scripts — parses args and prints the data filename.
    """
    args = parse_args()
    fname = get_data_filename(args)
    print(fname)