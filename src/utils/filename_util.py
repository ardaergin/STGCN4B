from typing import Optional, Literal

from ..config.args import parse_args


def get_data_filename(
        file_type:      Literal['dataframe', 'metadata', 'hetero_input'],
        interval:       str,
        weather_mode:   Optional[Literal['node', 'feature']]                                  = None,
        model_family:   Optional[Literal['graph', 'tabular']]                                 = None,
        task_type:      Optional[Literal['consumption_forecast', 'measurement_forecast']]     = None,
        hetero_mode:    Optional[Literal['full', 'minimal']]                                  = None,
    ) -> str:
    """
    Generates a data filename based on the pipeline's output logic.

    Args:
        file_type (str): The type of file ('dataframe', 'metadata', 'hetero_data').
        interval (str): The time interval used (e.g., '1h', '30min').
        weather_mode (str): for homogenous stgcn, (options: 'node' or 'feature').
        model_family (str): The family of the model ('graph', 'tabular').
        task_type (str): The overall task ('consumption_forecast', or 'measurement_forecast').
        hetero_mode (str): The mode for heterogeneous data (options: 'full' or 'minimal').
    
    Returns:
        str: The base filename without an extension.
    """
    if file_type == 'metadata' or model_family == 'graph':
        if weather_mode is None: 
            raise ValueError("weather_mode must be provided for metadata and graph model files.")
        elif weather_mode == "feature": weather_suffix = "Wfeat"
        elif weather_mode == "node":    weather_suffix = "Wnode"
        else: raise ValueError(f"Invalid weather_mode: {weather_mode}")
    
    if file_type == 'metadata':
        return f"metadata_{interval}_{weather_suffix}"
    
    elif file_type == 'dataframe':
        
        if model_family == 'tabular':
            if task_type is None: 
                raise ValueError("task_type must be provided for tabular model files.")
            if   task_type == 'consumption_forecast': task_name = 'consumption'
            elif task_type == 'measurement_forecast': task_name = 'measurement'
            else: raise ValueError(f"Invalid task_type '{task_type}' for a tabular file.")
            return f"tabular_{task_name}_{interval}"
        
        elif model_family == 'graph':
            return f"stgcn_{interval}_{weather_suffix}"
        
        else:
            raise ValueError(f"Invalid model_family provided: '{model_family}'")
    
    elif file_type == 'hetero_input':
        return f"hetero_input_{interval}_{hetero_mode}"
    
    else:
        raise ValueError(f"Invalid file_type provided: '{file_type}'")

if __name__ == "__main__":
    """
    Entry point for SLURM job scripts. 
    It parses arguments and prints the corresponding base filename.
    """
    args = parse_args() 
    base_filename = get_data_filename(
        model_family=args.model_family,
        file_type=args.file_type,
        task_type=args.task_type,
        interval=args.interval,
        weather_mode=args.weather_mode
    )
    print(base_filename)