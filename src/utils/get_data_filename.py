import os
from ..config.args import parse_args 

def main():
    """
    Parses command-line arguments and prints the corresponding
    data file name that needs to be copied to scratch space.
    """
    args = parse_args()
    
    if args.task_type == "measurement_forecast":
        fname = f"torch_input_{args.adjacency_type}_{args.interval}_{args.graph_type}_{args.measurement_type}.pt"
    else:
        fname = f"torch_input_{args.adjacency_type}_{args.interval}_{args.graph_type}.pt"
        
    print(fname)

if __name__ == "__main__":
    main()