import os
import torch

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
from .hetero_graph import HeteroGraphBuilderMixin
# from .tabular import TabularBuilderMixin


class OfficeGraphBuilder(
    SpatialBuilderMixin,
    SpatialVisualizerMixin,
    TemporalBuilderMixin,
    TemporalVisualizerMixin,
    HomogGraphBuilderMixin,
    HeteroGraphBuilderMixin,
    # TabularBuilderMixin
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
            "Contact", "thermostatHeatingSetpoint" # too few measurements
            }
        self.used_property_types = ["Temperature", "CO2Level", "Humidity"]
        
        # (Static) Room Attributes to use for modeling
        self.static_room_attributes = [
            'hasWindows', 'has_multiple_windows', 
            'window_direction_sin', 'window_direction_cos', 
            # 'hasBackWindows', 'hasFrontWindows', 'hasRightWindows', 'hasLeftWindows', 
            'isProperRoom', 
            'norm_area_minmax', # 'norm_area_prop', 
            # 'polygons_doc.width', 'polygons_doc.height', 'polygons_doc.centroid',
            # 'polygons_doc.compactness', 'polygons_doc.rect_fit', 'polygons_doc.aspect_ratio', 'polygons_doc.perimeter'
            ]
        
    def set_build_mode(self, mode: str):
        """
        Options:
            - "workhour_classification"
            - "consumption_forecast"
            - "measurement_forecast"
        """
        self.build_mode = mode
        return None

def main():
    # Argument parser
    from ...config.args import parse_args
    args = parse_args()

    # Loading OfficeGraph data
    from ..officegraph import OfficeGraph
    office_graph = OfficeGraph.from_pickles(floors_to_load = args.floors)
    builder = OfficeGraphBuilder(office_graph)
    builder.set_build_mode(mode=args.task_type)
    logger.info(f"Builder initialized with build mode '{builder.build_mode}'.")

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
    logger.info("Loading and processing weather data...")
    builder.get_weather_data(weather_csv_path=args.weather_csv_path)
    
    if builder.build_mode in ("workhour_classification", "consumption_forecast"):
        # Get classification labels
        logger.info("Generating work hour classification labels...")
        builder.get_classification_labels(country_code=args.country_code)
        
        # Get forecasting values
        logger.info("Loading and processing consumption data...")
        builder.get_forecasting_values(consumption_dir=args.consumption_dir)

    # Processing measurements
    logger.info("Processing measurements...")
    
    # Bucket measurements by device and property
    builder.bucket_measurements_by_device_property()
        
    # Build full feature DataFrame
    builder.build_full_feature_df()

    # ============================
    # SPATIAL SETUP
    # ============================
    # Presets for static room attributes
    static_attr_presets = {
        'minimal': ['isProperRoom', 'norm_area_minmax'],
        'standard': ['hasWindows', 'has_multiple_windows', 'window_direction_sin', 'window_direction_cos', 
                    'isProperRoom', 'norm_area_minmax'],
        'all': ['hasWindows', 'has_multiple_windows', 
                'window_direction_sin', 'window_direction_cos', 
                'hasBackWindows', 'hasFrontWindows', 'hasRightWindows', 'hasLeftWindows', 
                'isProperRoom', 
                'norm_area_minmax', 'norm_area_prop', 
                'polygons_doc.centroid',
                'polygons_doc.width', 'polygons_doc.height',
                'polygons_doc.compactness', 'polygons_doc.rect_fit', 'polygons_doc.aspect_ratio', 'polygons_doc.perimeter']
    }
    # Setting static room attributes based on preset
    builder.static_room_attributes = static_attr_presets[args.static_attr_preset]

    logger.info("Setting up spatial components...")
    
    # Initialize room polygons
    builder.initialize_room_polygons(
        polygon_type=args.polygon_type,
        simplify_polygons=args.simplify_polygons,
        simplify_epsilon=args.simplify_epsilon
    )
    
    # Normalize room areas
    builder.normalize_room_areas()
    
    # Build horizontal adjacency
    builder.build_horizontal_adjacency(
        mode=args.adjacency_type,
        distance_threshold=args.distance_threshold
    )

    # Build vertical adjacency
    builder.build_vertical_adjacency(
        min_overlap_area=0.05,
        min_weight=0
    )
    
    # Combined horizontal + vertical adjacency
    builder.build_combined_room_to_room_adjacency()

    # Build outside adjacency
    if not args.skip_incorporating_weather:
        logger.info("Calculating outside adjacency...")
        builder.build_outside_adjacency(mode=args.adjacency_type)

    # Calculate information propagation masks and apply them
    logger.info("Building masked adjacency matrices for information propagation...")
    builder.build_masked_adjacencies()

    # ============================
    # DATA BUILDING
    # ============================
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.data_to_build == "tabular":
        raise NotImplementedError("In progress.")

    elif args.data_to_build == "graph":

        # Heterogeneous Graph Builder
        if args.graph_type == "heterogeneous":
            builder.build_base_hetero_graph()
            builder.build_hetero_temporal_graphs()

            # Prepare torch input
            logger.info("Preparing heterogeneous STGCN input...")
            heterogeneous_stgcn_input = builder.prepare_hetero_stgcn_input()
            torch_tensors = builder.convert_hetero_to_torch_tensors(heterogeneous_stgcn_input)

            if args.task_type == "measurement_forecast":
                file_name = f"stgcn_input_{args.adjacency_type}_{args.interval}_{args.graph_type}_{args.measurement_type}.pt"
            else:
                file_name = f"stgcn_input_{args.adjacency_type}_{args.interval}_{args.graph_type}.pt"

            full_output_path = os.path.join("data/processed", file_name)
            torch.save(torch_tensors, full_output_path)
            logger.info(f"Saved tensors to {full_output_path} with default parameters.")

        # Homogeneous Graph Builder
        elif args.graph_type == "homogeneous":
            builder.build_room_feature_df()

            if args.task_type == "measurement_forecast":
                builder.get_targets_and_mask_for_a_variable(args.measurement_type)

            # Incorporate weather data as outside space if specified
            if not args.skip_incorporating_weather:
                logger.info("Integrating outside node with weather data into homogeneous graph...")
                builder.incorporate_weather_as_an_outside_room()

            # Feature matrices
            logger.info("Generating feature arrays for homogeneous graph...")
            builder.build_feature_array()

            if args.task_type == "measurement_forecast":
                file_name = f"stgcn_input_{args.adjacency_type}_{args.interval}_{args.graph_type}_{args.measurement_type}.pt"
            else:
                file_name = f"stgcn_input_{args.adjacency_type}_{args.interval}_{args.graph_type}.pt"
            full_output_path = os.path.join("data/processed", file_name)

            # Prepare numpy input
            logger.info("Preparing homogeneous numpy input...")
            builder.prepare_and_save_numpy_input(output_path=full_output_path)
        
        if args.graph_type == "heterogeneous":
            # Save graph schema
            schema_path = os.path.join(args.output_dir, f"hetero_graph_schema_{args.interval}.txt")
            schema = builder.visualize_hetero_graph_schema(save_path=schema_path)
            logger.info("Saved heterogeneous graph schema")
        
if __name__ == "__main__":
    main()