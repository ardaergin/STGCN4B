import os

from ...utils.missingness_plot import plot_missing_values

# Logging setup
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Importing mixin classes
from .temporal import TemporalBuilderMixin
from .temporal_viz import TemporalVisualizerMixin
from .spatial import SpatialBuilderMixin
from .spatial_viz import SpatialVisualizerMixin
from .homo_graph import HomogGraphBuilderMixin
# from .hetero_graph import HeteroGraphBuilderMixin
from .tabular import TabularBuilderMixin
from ...utils.filename_util import get_data_filename


class OfficeGraphBuilder(
    SpatialBuilderMixin,
    SpatialVisualizerMixin,
    TemporalBuilderMixin,
    TemporalVisualizerMixin,
    HomogGraphBuilderMixin,
    # HeteroGraphBuilderMixin,
    TabularBuilderMixin
    ):
    """
    Consolidated class to build and manipulate graphs from OfficeGraph data,
    including spatial relationships and temporal features.
    """
    
    def __init__(self, office_graph):
        self.office_graph = office_graph
        
        self.room_uris = list(self.office_graph.rooms.keys())
        self.room_names = {}
        for room_uri in self.room_uris:
            self.room_names[room_uri] = self.office_graph.rooms[room_uri].room_number
        
        # Property configuration
        self.ignored_property_types = {
            "DeviceStatus", "BatteryLevel", # unnecessary
            "Contact", "Motion", "thermostatHeatingSetpoint" # too few measurements
            }
        self.used_property_types = ["Temperature", "CO2Level", "Humidity"]
        
        # Static Room class attributes to use for modeling, default 'standard' preset:
        self.static_room_attributes = ['hasWindows', 'has_multiple_windows', 
                                       'window_direction_sin', 'window_direction_cos', 
                                       'isProperRoom', 'norm_areas_minmax']
        
    def set_build_mode(self, mode: str, measurement_variable=None, measurement_variable_stat=None):
        """
        Options for 'mode':
            - "workhour_classification"
            - "consumption_forecast"
            - "measurement_forecast"
        """
        valid_options = ["workhour_classification", "consumption_forecast", "measurement_forecast"]
        if mode not in valid_options:
            raise ValueError(f"Invalid build mode '{mode}'. Valid options are: {valid_options}")
        else:
            self.build_mode = mode
        
        # Measurement forecast task checks & setup
        if mode == "measurement_forecast" and measurement_variable is None:
            raise ValueError("measurement_variable must be specified for 'measurement_forecast' mode.")
        if mode == "measurement_forecast" and measurement_variable_stat is None:
            raise ValueError("measurement_stat must be specified for 'measurement_forecast' mode.")
        
        valid_measurement_variables = ["Temperature", "CO2Level", "Humidity"]
        if mode == "measurement_forecast" and measurement_variable not in valid_measurement_variables:
            raise ValueError(f"Invalid measurement variable '{measurement_variable}'. Valid options are: {valid_measurement_variables}")
        else:
            self.measurement_variable = measurement_variable
            self.measurement_variable_stat = measurement_variable_stat
        
        return None

def main():
    # Argument parser
    from ...config.args import parse_args
    args = parse_args()
    
    # Loading OfficeGraph data
    from ..officegraph import OfficeGraph
    office_graph = OfficeGraph.from_pickles(floors_to_load = args.floors, data_dir=args.data_dir)
    
    # Setting up the builder
    builder = OfficeGraphBuilder(office_graph)
    
    # Setting build mode based on task type
    builder.set_build_mode(
        mode=args.task_type, 
        measurement_variable=args.measurement_variable, 
        measurement_variable_stat=args.measurement_variable_stat)

    logger.info(f"Builder initialized with build mode '{builder.build_mode}'.")

    os.makedirs(args.processed_data_dir, exist_ok=True)
    if args.make_and_save_plots:
        os.makedirs(args.builder_plots_dir, exist_ok=True) # For plots

        # Missingness plots path
        missing_plots_dir = os.path.join(args.builder_plots_dir, "missingness_plots")
        os.makedirs(missing_plots_dir, exist_ok=True)

        # plot_suffix for missingness plots
        if args.floors == [7]:
            plot_suffix = "(floor 7)"
        else:
            plot_suffix = "(all floors)"

    # ============================
    # SPATIAL SETUP
    # ============================
    logger.info("Setting up spatial components...")

    # Presets for static room attributes
    static_attr_presets = {
        'minimal': ['isProperRoom', 'norm_areas_minmax'],
        'standard': ['hasWindows', 'has_multiple_windows', 'window_direction_sin', 'window_direction_cos', 
                    'isProperRoom', 'norm_areas_minmax'],
        'all': ['hasWindows', 'has_multiple_windows', 
                'window_direction_sin', 'window_direction_cos', 
                'hasBackWindows', 'hasFrontWindows', 'hasRightWindows', 'hasLeftWindows', 
                'isProperRoom', 
                'norm_areas_minmax', 'norm_areas_prop', 
                'polygons_doc.centroid',
                'polygons_doc.width', 'polygons_doc.height',
                'polygons_doc.compactness', 'polygons_doc.rect_fit', 'polygons_doc.aspect_ratio', 'polygons_doc.perimeter']
    }
    # Setting static room attributes based on preset
    builder.static_room_attributes = static_attr_presets[args.static_attr_preset]

    # Initialize room polygons
    builder.initialize_room_polygons(
        polygon_type=args.polygon_type,
        simplify_polygons=args.simplify_polygons,
        simplify_epsilon=args.simplify_epsilon
    )
    
    # Normalize room areas
    builder.normalize_room_areas()

    if args.model_family == "graph":
        # Build horizontal adjacency
        builder.build_horizontal_adjacency(
            mode=args.adjacency_type,
            distance_threshold=args.distance_threshold
        )

        # Build vertical adjacency
        builder.build_vertical_adjacency(
            mode=args.adjacency_type,
            min_overlap_area=0.05,
            min_weight=0
        )
        
        # Combined horizontal + vertical adjacency
        builder.build_combined_room_to_room_adjacency()

        # Calculate information propagation masks and apply them
        logger.info("Building masked adjacency matrices for information propagation...")
        builder.build_masked_adjacencies()
        
        # Build outside adjacency
        if not args.skip_incorporating_weather:
            logger.info("Calculating outside adjacency...")
            builder.build_outside_adjacency(mode=args.adjacency_type)
    
    # ============================
    # TEMPORAL SETUP
    # ============================
    logger.info("Setting up temporal parameters...")

    # Initialize time parameters
    builder.initialize_time_parameters(
        start_time=args.start_time,
        end_time=args.end_time,
        interval=args.interval,
        use_sundays=args.use_sundays
    )
    # Split time buckets
    builder.build_weekly_blocks()

    # Get weather data
    if not args.skip_incorporating_weather:
        logger.info("Loading and processing weather data...")
        builder.get_weather_data(weather_csv_path=args.weather_csv_path)
    
    # Targets
    if builder.build_mode == "workhour_classification":
        # Get classification labels
        logger.info("Generating work hour classification labels...")
        builder.get_workhour_labels(country_code=args.country_code)
    elif builder.build_mode == "consumption_forecast":
        # Get forecasting values
        logger.info("Loading and processing consumption data...")
        builder.get_consumption_values(consumption_dir=args.consumption_dir)
    
    # Building the device-level feature DataFrame
    builder.build_device_level_df()

    # Building the room-level feature DataFrame
    builder.build_room_level_df()
    
    # Building the floor-level feature DataFrame    
    builder.build_floor_level_df()

    # Building the building-level feature DataFrame
    builder.build_building_level_df()

    # Missingness plots for all 4 of these DFs
    if args.make_and_save_plots:
        plot_missing_values(builder.device_level_df, df_name=f"Device-level DF {plot_suffix}",
                            save=True, output_dir=missing_plots_dir)
        plot_missing_values(builder.room_level_df, df_name=f"Room-level DF {plot_suffix}",
                            save=True, output_dir=missing_plots_dir)
        plot_missing_values(builder.floor_level_df, df_name=f"Floor-level DF {plot_suffix}",
                            save=True, output_dir=missing_plots_dir)
        plot_missing_values(builder.building_level_df, df_name=f"Building-level DF {plot_suffix}",
                            save=True, output_dir=missing_plots_dir)
        
    # ============================
    # DATA BUILDING
    # ============================
    
    if args.model_family == "tabular":
        if args.task_type == "measurement_forecast":
            builder.add_static_room_features()

        builder.build_tabular_df(
            forecast_horizon=args.forecast_horizon,
            lags=args.lags,
            windows=args.windows,
            shift_amount=args.shift_amount,
            integrate_weather=not args.skip_incorporating_weather
        )
        if args.make_and_save_plots:
            plot_missing_values(builder.tabular_feature_df, df_name=f"Tabular Feature DF {plot_suffix}",
                                save=True, output_dir=missing_plots_dir)
            plot_missing_values(builder.tabular_df, df_name=f"Tabular DF {plot_suffix}",
                                save=True, output_dir=missing_plots_dir)

        # Get file name via helper
        fname = get_data_filename(args)
        full_output_path = os.path.join(args.processed_data_dir, fname)

        # Save
        builder.save_tabular_df(output_path=full_output_path)

    elif args.model_family == "graph":

        # Heterogeneous Graph Builder
        if args.graph_type == "heterogeneous":
            raise NotImplementedError("Not implemented yet.")
            # builder.build_base_hetero_graph()
            # builder.build_hetero_temporal_graphs()

            # # Prepare torch input
            # logger.info("Preparing heterogeneous STGCN input...")
            # heterogeneous_stgcn_input = builder.prepare_hetero_stgcn_input()
            # torch_tensors = builder.convert_hetero_to_torch_tensors(heterogeneous_stgcn_input)

            # fname = get_data_filename(args)
            # full_output_path = os.path.join("data/processed", fname)
            # torch.save(torch_tensors, full_output_path)
            # logger.info(f"Saved tensors to {full_output_path} with default parameters.")
            # 
            # # Save graph schema
            # schema_path = os.path.join(args.output_dir, f"hetero_graph_schema_{args.interval}.txt")
            # schema = builder.visualize_hetero_graph_schema(save_path=schema_path)
            # logger.info("Saved heterogeneous graph schema")

        # Homogeneous Graph Builder
        elif args.graph_type == "homogeneous":
            # Expanding the room_level_df by adding also the empty room-bucket combinations
            builder.expand_room_level_df()

            # Adding static attributes
            builder.add_static_room_features()
            if args.make_and_save_plots:
                plot_missing_values(builder.room_level_df, df_name=f"Room-level DF, Extended {plot_suffix}",
                                    save=True, output_dir=missing_plots_dir)
                
            ######################## DEBUG ########################
            logger.info("######################## DEBUG ########################")
            # 1. First, let's verify the column exists to avoid errors
            if 'isProperRoom' in builder.room_level_df.columns:
                
                # 2. Create a boolean mask to find rows where 'isProperRoom' is NaN
                nan_mask = builder.room_level_df['isProperRoom'].isnull()
                
                # 3. Select the 'room_URIRef' column from the filtered rows
                problematic_rooms = builder.room_level_df.loc[nan_mask, 'room_URIRef']
                
                # 4. Get the unique room URIs (since a room appears in many time buckets)
                unique_problematic_uris = problematic_rooms.unique()
                
                # 5. Display the result
                if len(unique_problematic_uris) > 0:
                    logger.info(f"Found {len(unique_problematic_uris)} room(s) where 'isProperRoom' is NaN:")
                    for uri in unique_problematic_uris:
                        logger.info(f"- {uri}")
                else:
                    logger.info("No rooms found where 'isProperRoom' is NaN.")
                    
            else:
                logger.info("Column 'isProperRoom' not found in the DataFrame.")
            logger.info("######################## DEBUG ########################")
            ######################## DEBUG ########################

            # Incorporate weather data as outside space if specified
            if not args.skip_incorporating_weather:
                logger.info("Integrating outside node with weather data into homogeneous graph...")
                builder.incorporate_weather_as_an_outside_room()
                if args.make_and_save_plots:
                    plot_missing_values(builder.room_level_df, df_name=f"Room-level DF, Extended with Weather {plot_suffix}",
                        save=True, output_dir=missing_plots_dir)

            if builder.build_mode == "measurement_forecast":
                builder.get_targets_and_mask_for_a_variable()
            
            # Feature matrices
            logger.info("Generating feature arrays for homogeneous graph...")
            builder.build_feature_array()
            
            # Get file name via helper
            fname = get_data_filename(args)
            full_output_path = os.path.join(args.processed_data_dir, fname)

            # Prepare numpy input
            logger.info("Preparing homogeneous numpy input...")
            builder.prepare_and_save_numpy_input(output_path=full_output_path)
        
if __name__ == "__main__":
    main()