import os
from ..config.args import parse_args 

def get_data_filename():
    """
    Parses command-line arguments and returns the corresponding
    data file name that needs to be copied to scratch space.
    """
    args = parse_args()
    
    # Prefix: Graph/Tabular
    if args.data_to_build == "graph":
        fname_prefix = f"{args.graph_type}_{args.data_to_build}_input"
    elif args.data_to_build == "tabular":
        fname_prefix = f"{args.data_to_build}_input"
    else:
        raise ValueError(f"Invalid data_to_build: {args.data_to_build}")
    
    # Weather
    if args.skip_incorporating_weather:
        weather_suffix = "WeatherOut"
    else: 
        weather_suffix = "WeatherIn"

    if args.task_type == "workhour_classification":
        fname = f"{fname_prefix}_floor7_{args.adjacency_type}_{args.gso_mode}_{args.interval}_{weather_suffix}.pt"
    elif args.task_type == "consumption_forecast":
        fname = f"{fname_prefix}_consumption_{args.adjacency_type}_{args.gso_mode}_{args.interval}_{weather_suffix}.pt"
    elif args.task_type == "measurement_forecast":
        fname = f"{fname_prefix}_{args.measurement_variable}_{args.adjacency_type}_{args.gso_mode}_{args.interval}_{weather_suffix}.pt"
    else:
        raise ValueError(f"Invalid task_type: {args.task_type}")
    
    return fname

if __name__ == "__main__":
    """
    Entry point for SLURM job scripts — prints the data filename to stdout.
    """
    fname = get_data_filename()
    print(fname)