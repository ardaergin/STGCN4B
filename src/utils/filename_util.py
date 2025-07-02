from typing import Optional
from ..config.args import parse_args

def get_data_filename(
        file_type: str,
        task_type: Optional[str] = None,
        model_family: Optional[str] = None,
        interval: Optional[str] = None,
        incorporate_weather: Optional[bool] = None,
    ) -> str:
    """
    Generates a data filename based on the pipeline's output logic.

    Args:
        file_type (str): The type of file ('dataframe', 'metadata').
        model_family (str): The family of the model ('graph', 'tabular').
        task_type (str): The overall task ('workhour_classification', 
                         'consumption_forecast', etc.).
        interval (str): The time interval used (e.g., '1h', '30min').
        incorporate_weather (bool): Flag indicating if weather data was used.

    Returns:
        str: The base filename without an extension.
    """
    # Case 1: Special handling for 'workhour_classification' task
    if task_type == "workhour_classification":
        if file_type == 'metadata':
            # Metadata for this task is independent of the model family.
            return "workhour_metadata"
        elif file_type == 'dataframe':
            # Dataframe names depend on the model family.
            if model_family == 'tabular':
                return "workhour_tabular"
            elif model_family == 'graph':
                return "workhour_stgcn"
            else:
                raise ValueError(f"Invalid model_family '{model_family}' for workhour_classification.")
        else:
            raise ValueError(f"Invalid file_type '{file_type}' for workhour_classification.")

    # Case 2: For all other tasks, interval and weather status are required.
    if interval is None or incorporate_weather is None:
        raise ValueError(
            f"If task_type is not 'workhour_classification', 'interval' and "
            f"'incorporate_weather' must be provided."
        )

    weather_suffix = "WeatherIn" if incorporate_weather else "WeatherOut"

    # For other tasks, metadata is shared, but dataframes are specific.
    if file_type == 'metadata':
        # Metadata filenames are shared across model families for these tasks.
        return f"metadata_{interval}_{weather_suffix}"
    
    elif file_type == 'dataframe':
        if model_family == 'tabular':
            # Tabular dataframe names are task-specific.
            if task_type == 'consumption_forecast':
                task_name = 'consumption'
            elif task_type == 'measurement_forecast':
                task_name = 'measurement'
            else:
                raise ValueError(f"Invalid task_type '{task_type}' for a tabular file.")
            
            return f"tabular_{task_name}_{interval}_{weather_suffix}"

        elif model_family == 'graph':
            # Graph dataframe names are shared between consumption/measurement tasks.
            return f"stgcn_{interval}_{weather_suffix}"
        
        else:
            raise ValueError(f"Invalid model_family provided: '{model_family}'")
    else:
        raise ValueError(f"Invalid file_type provided: '{file_type}'")

if __name__ == "__main__":
    """
    Entry point for SLURM job scripts. It parses arguments and prints the 
    corresponding base filename.
    """
    # Note: You will need to update parse_args() to include 'model_family'.
    args = parse_args() 
    base_filename = get_data_filename(
        model_family=args.model_family, # New argument
        file_type=args.file_type,
        task_type=args.task_type,
        interval=args.interval,
        incorporate_weather=args.incorporate_weather
    )
    print(base_filename)