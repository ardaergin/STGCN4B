import os
import sys
import torch
import logging

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from .spatial import SpatialBuilderMixin
from .temporal import TemporalBuilderMixin
from .homo_graph import HomogGraphBuilderMixin
from .hetero_graph import HeteroGraphBuilderMixin

class OfficeGraphBuilder(
    SpatialBuilderMixin, 
    TemporalBuilderMixin,
    HomogGraphBuilderMixin,
    HeteroGraphBuilderMixin):
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

        self.static_room_attributes = [
            # Windows:
            'hasWindows',
            'has_multiple_windows',
            'window_direction_sin',
            'window_direction_cos',
            'hasBackWindows',
            'hasFrontWindows',
            'hasRightWindows',
            'hasLeftWindows',
            # Whether it is a room or not:
            'isProperRoom',
            # Area:
            'norm_area_minmax',
            'norm_area_prop',

            # Other polygon-based parameters:
            # 'polygons_doc.centroid',
            # 'polygons_doc.perimeter',
            # 'polygons_doc.width',
            # 'polygons_doc.height',
            # 'polygons_doc.compactness',
            # 'polygons_doc.rect_fit',
            # 'polygons_doc.aspect_ratio'
        ]

    def quick_build_with_default_parameters(self, graph_type = "heterogenous", save=True):
        # Temporal Builder    
        self.initialize_time_parameters()
        self.split_time_buckets()
        self.get_weather_data()
        self.get_classification_labels()
        self.get_forecasting_values()
        self.bucket_measurements_by_device_property()
        self.normalize_bucketed_measurements()
        self.build_full_feature_df()

        # Spatial Builder
        self.initialize_room_polygons()
        self.normalize_room_areas()

        self.build_horizontal_adjacency()
        self.combine_horizontal_adjacencies()

        self.build_vertical_adjacency()

        self.apply_masks_to_adjacency()

        self.build_outside_adjacency()
        self.combine_outside_adjacencies()
        
        # Heterogenous Graph Builder
        if graph_type == "heterogenous":
            self.build_base_hetero_graph()
            self.build_hetero_temporal_graphs()
            heterogenous_stgcn_input = self.prepare_hetero_stgcn_input()
            torch_tensors = self.convert_hetero_to_torch_tensors(heterogenous_stgcn_input)
        # Homogenous Graph Builder
        elif graph_type == "homogenous":
            self.build_homogeneous_graph()
            self.build_temporal_graph_snapshots()
            homogenous_stgcn_input = self.prepare_stgcn_input()
            torch_tensors = self.convert_to_torch_tensors(homogenous_stgcn_input)
        
        if save:
            file_name = f"torch_input_weighted_1h_{graph_type}.pt"
            full_output_path = os.path.join("data/processed", file_name)
            torch.save(torch_tensors, full_output_path)
            logger.info(f"Saved tensors to {full_output_path} with default parameters.")
        
        return None

def main():
    from ..officegraph import OfficeGraph
    office_graph = OfficeGraph.from_pickles(floors_to_load = [4,5,6,7])
    builder = OfficeGraphBuilder(office_graph)
    builder.quick_build_with_default_parameters()

if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     import argparse
#     import os
#     parser = argparse.ArgumentParser(description='Prepare OfficeGraph data for analysis and visualization')
    
#     # Input/output arguments
#     parser.add_argument('--officegraph_path', type=str, 
#                         default='data/processed/officegraph.pkl',
#                         help='Path to the pickled OfficeGraph')
    
#     parser.add_argument('--output_dir', type=str, 
#                         default='data/processed',
#                         help='Directory to save the PyTorch tensors')
    
    
#     # Static Room attributes
#     parser.add_argument('--static_attr_preset', type=str,
#                         choices=['minimal', 'standard', 'all'],
#                         default='standard',
#                         help='Preset for static room attributes: minimal, standard, or all')

#     # Time-related arguments
#     parser.add_argument('--start_time', type=str, 
#                         default="2022-03-07 00:00:00",
#                         help='Start time for analysis (YYYY-MM-DD HH:MM:SS)')

#     parser.add_argument('--end_time', type=str, 
#                         default="2023-01-30 00:00:00",
#                         help='End time for analysis (YYYY-MM-DD HH:MM:SS)')
    
#     parser.add_argument('--interval', type=str, 
#                         default="1h",
#                         help='Frequency of time buckets as a pandas offset string e.g., ("15min", "30min", "1h", "2h")')
    
#     parser.add_argument('--use_sundays', action='store_true',
#                         help='Include Sundays in the analysis')
    
#     parser.add_argument(
#         "--split",
#         type=int,
#         nargs=3,
#         metavar=("TRAIN", "VAL", "TEST"),
#         default=[3, 1, 1],
#         help="train/val/test split in number of blocks (default: 3 1 1)"
#     )
    
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=2658918,
#         help="random seed for reproducibility"
#     )

#     # Weather data arguments
#     parser.add_argument('--weather_csv_path', type=str,
#                         default="data/weather/hourly_weather_2022_2023.csv",
#                         help='Path to weather data CSV file')
    
#     parser.add_argument('--weather_scaler', type=str,
#                         choices=['standard', 'robust', 'minmax'],
#                         default='robust',
#                         help='Scaler type for weather data normalization')

#     # Polygon-related arguments
#     parser.add_argument('--polygon_type', type=str,
#                         choices=['geo', 'doc'],
#                         default='doc',
#                         help='Type of polygon data to use: geo or doc')
    
#     parser.add_argument('--simplify_polygons', action='store_true',
#                         dest='simplify_polygons',
#                         help='Polygon simplification (off by default)')
    
#     parser.add_argument('--simplify_epsilon', type=float,
#                         default=0.1,
#                         help='Epsilon value for polygon simplification (higher = more simplification)')
    
#     # Adjacency-related arguments
#     parser.add_argument('--adjacency_type', type=str,
#                         choices=['binary', 'weighted'],
#                         default='weighted',
#                         help='Type of adjacency to use: binary or weighted')
    
#     parser.add_argument('--distance_threshold', type=float,
#                         default=5.0,
#                         help='Distance threshold (in meters) for room adjacency')
    
#     parser.add_argument('--outside_adjacency_mode', type=str,
#                         choices=['binary', 'weighted'],
#                         default='weighted',
#                         help='Mode for calculating outside adjacency')

#     # Measurement processing arguments
#     parser.add_argument('--measurement_scaler', type=str,
#                         choices=['standard', 'robust', 'minmax'],
#                         default='robust',
#                         help='Scaler type for measurement normalization')
    
#     parser.add_argument('--drop_sum', action='store_true',
#                         help='Drop sum feature from measurements')

#     # Consumption forecasting arguments
#     parser.add_argument('--consumption_dir', type=str, 
#                         default='data/consumption',
#                         help='Directory containing consumption data (for forecasting)')
    
#     parser.add_argument('--consumption_scaler', type=str,
#                         choices=['standard', 'robust', 'minmax'],
#                         default='robust',
#                         help='Scaler type for consumption data normalization')

#     # Classification arguments
#     parser.add_argument('--country_code', type=str,
#                         default='NL',
#                         help='Country code for work hour classification')

#     # Graph visualization arguments
#     parser.add_argument('--plot_floor_plan', action='store_true',
#                         help='Generate floor plan visualization')
    
#     parser.add_argument('--plot_adjacency', action='store_true',
#                         help='Generate adjacency matrix visualization')
    
#     parser.add_argument('--plot_network', action='store_true',
#                         help='Generate network graph visualization')
    
#     parser.add_argument('--plot_outside_adjacency', action='store_true',
#                         help='Generate outside adjacency visualization')
    
#     parser.add_argument('--plot_propagation', action='store_true',
#                         help='Generate interactive information propagation visualization')
    
#     parser.add_argument('--network_layout', type=str,
#                         choices=['spring', 'kamada_kawai', 'planar', 'spatial'],
#                         default='spring',
#                         help='Layout for network graph visualization')
    
#     parser.add_argument('--save_plots', action='store_false',
#                         help='Save plot images instead of displaying them')
    
#     parser.add_argument('--plots_dir', type=str,
#                         default='output/builder',
#                         help='Directory to save plot images')

#     # Tabular baseline arguments
#     parser.add_argument('--build_tabular', action='store_false',
#                         help='Build tabular baseline inputs (homogeneous graph method)')
    
#     parser.add_argument('--build_advanced_tabular', action='store_false',
#                         help='Build advanced tabular dataset with feature engineering')
    
#     parser.add_argument('--include_datetime_features', action='store_false',
#                         help='Include cyclical datetime features in advanced tabular')
    
#     parser.add_argument('--include_sensor_aggregates', action='store_false',
#                         help='Include sensor aggregate features in advanced tabular')
    
#     parser.add_argument('--lag_steps', type=int, nargs='+',
#                         default=[1, 2, 3],
#                         help='Lag steps for time series features')
    
#     parser.add_argument('--rolling_windows', type=int, nargs='+',
#                         default=[3, 6, 12],
#                         help='Rolling window sizes for time series features')
    
#     # Device specification
#     parser.add_argument('--device', type=str,
#                         default='cpu',
#                         help='PyTorch device for tensor operations (cpu, cuda, etc.)')
    
#     args = parser.parse_args()
    
#     # Configure logging
#     logging.basicConfig(level=logging.INFO, 
#                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
#     # Load OfficeGraph
#     from ..officegraph import OfficeGraph
#     from ..extraction import OfficeGraphExtractor
#     logger.info(f"Loading OfficeGraph from {args.officegraph_path}")
#     with open(args.officegraph_path, 'rb') as f:
#         office_graph = pickle.load(f)
    
#     # Initialize builder
#     builder = OfficeGraphBuilder(office_graph)
    
#     # Define presets for static room attributes
#     static_attr_presets = {
#         'minimal': ['isProperRoom', 'norm_area_minmax'],
#         'standard': ['hasWindows', 'has_multiple_windows', 'window_direction_sin', 'window_direction_cos', 
#                      'isProperRoom', 'norm_area_minmax'],
#         'all': ['hasWindows', 'has_multiple_windows', 
#                 'window_direction_sin', 'window_direction_cos', 
#                 'hasBackWindows', 'hasFrontWindows', 'hasRightWindows', 'hasLeftWindows', 
#                 'isProperRoom', 
#                 'norm_area_minmax', 'norm_area_prop', 
#                 'polygons_doc.centroid',
#                 'polygons_doc.width', 'polygons_doc.height',
#                 'polygons_doc.compactness', 'polygons_doc.rect_fit', 'polygons_doc.aspect_ratio', 'polygons_doc.perimeter']
#     }
#     builder.static_room_attributes = static_attr_presets[args.static_attr_preset]

#     # ============================
#     # TEMPORAL SETUP
#     # ============================
#     logger.info("Setting up temporal parameters...")
    
#     # Initialize time parameters
#     builder.initialize_time_parameters(
#         start_time=args.start_time,
#         end_time=args.end_time,
#         interval=args.interval,
#         use_sundays=args.use_sundays
#     )
    
#     # Split time buckets
#     train_blocks, val_blocks, test_blocks = args.split
#     builder.split_time_buckets(
#         train_blocks=train_blocks,
#         val_blocks=val_blocks,
#         test_blocks=test_blocks,
#         seed=args.seed
#     )

#     # Get weather data
#     logger.info("Loading and processing weather data...")
#     builder.get_weather_data(
#         weather_csv_path=args.weather_csv_path,
#         normalize=True,
#         scaler=args.weather_scaler
#     )
    
#     # Get classification labels
#     logger.info("Generating work hour classification labels...")
#     builder.get_classification_labels(country_code=args.country_code)
    
#     # Get forecasting values
#     logger.info("Loading and processing consumption data...")
#     builder.get_forecasting_values(
#         consumption_dir=args.consumption_dir,
#         normalize=True,
#         scaler=args.consumption_scaler
#     )

#     # ============================
#     # SPATIAL SETUP
#     # ============================
#     logger.info("Setting up spatial components...")
    
#     # Initialize room polygons
#     builder.initialize_room_polygons(
#         polygon_type=args.polygon_type,
#         simplify_polygons=args.simplify_polygons,
#         simplify_epsilon=args.simplify_epsilon
#     )
    
#     # Normalize room areas
#     builder.normalize_room_areas()
    
#     # Build room-to-room adjacency
#     builder.build_room_to_room_adjacency(
#         matrix_type=args.adjacency_type,
#         distance_threshold=args.distance_threshold
#     )
    
#     # Calculate outside adjacency
#     logger.info("Calculating outside adjacency...")
#     builder.calculate_outside_adjacency(mode=args.outside_adjacency_mode)
    
#     # Calculate information propagation masks and apply them
#     logger.info("Calculating information propagation...")
#     builder.calculate_information_propagation_masks()
#     builder.apply_masks_to_adjacency()

#     # ============================
#     # MEASUREMENT PROCESSING
#     # ============================
#     logger.info("Processing measurements...")
    
#     # Bucket measurements by device and property
#     builder.bucket_measurements_by_device_property(drop_sum=args.drop_sum)
    
#     # Normalize bucketed measurements
#     builder.normalize_bucketed_measurements(
#         drop_sum=args.drop_sum,
#         scaler=args.measurement_scaler
#     )
    
#     # Build full feature DataFrame
#     builder.build_full_feature_df()

#     # ============================
#     # VISUALIZATION
#     # ============================
#     if args.save_plots:
#         os.makedirs(args.plots_dir, exist_ok=True)
    
#     # Plot floor plan if requested
#     if args.plot_floor_plan:
#         logger.info("Plotting floor plan")
#         fig = builder.plot_floor_plan(
#             normalization='min_max',
#             show_room_ids=True
#         )
#         if args.save_plots:
#             fig.savefig(f"{args.plots_dir}/floor_plan.png", dpi=300, bbox_inches='tight')
#             plt.close(fig)
#         else:
#             plt.show()
    
#     # Plot adjacency matrix if requested
#     if args.plot_adjacency:
#         logger.info("Plotting adjacency matrix")
#         fig = builder.plot_adjacency_matrix(show_room_ids=True)
#         if args.save_plots:
#             fig.savefig(f"{args.plots_dir}/adjacency_matrix.png", dpi=300, bbox_inches='tight')
#             plt.close(fig)
#         else:
#             plt.show()
    
#     # Plot outside adjacency if requested
#     if args.plot_outside_adjacency:
#         logger.info("Plotting outside adjacency")
#         fig = builder.plot_outside_adjacency(show_room_ids=True)
#         if args.save_plots:
#             fig.savefig(f"{args.plots_dir}/outside_adjacency.png", dpi=300, bbox_inches='tight')
#             plt.close(fig)
#         else:
#             plt.show()

#     # ============================
#     # GRAPH BUILDING
#     # ============================
    
#     # Ensure output directory exists
#     os.makedirs(args.output_dir, exist_ok=True)
    
#     # Build graphs based on specified type
#     if args.graph_type in ['homogeneous', 'both']:
#         logger.info("Building homogeneous graph and temporal snapshots...")
        
#         # Build homogeneous graph
#         builder.build_homogeneous_graph()
        
#         # Plot network graph if requested
#         if args.plot_network:
#             logger.info("Plotting network graph")
#             fig = builder.plot_network_graph(
#                 layout=args.network_layout,
#                 node_size_based_on='area',
#                 show_room_ids=True
#             )
#             if args.save_plots:
#                 fig.savefig(f"{args.plots_dir}/network_graph.png", dpi=300, bbox_inches='tight')
#                 plt.close(fig)
#             else:
#                 plt.show()
        
#         # Build temporal graph snapshots
#         builder.build_temporal_graph_snapshots()
        
#         # Generate feature matrices
#         builder.generate_feature_matrices()
        
#         # Prepare and save homogeneous STGCN input
#         logger.info("Preparing homogeneous STGCN input...")
#         homog_stgcn_input = builder.prepare_stgcn_input()
#         homog_torch_tensors = builder.convert_to_torch_tensors(homog_stgcn_input, device=args.device)
        
#         homog_output_path = os.path.join(args.output_dir, f"torch_input_homog_{args.adjacency_type}_{args.interval}.pt")
#         torch.save(homog_torch_tensors, homog_output_path)
#         logger.info(f"Saved homogeneous tensors to {homog_output_path}")
        
#         # Prepare tabular baseline if requested (homogeneous graph method)
#         if args.build_tabular:
#             logger.info("Preparing tabular baseline input (homogeneous method)...")
#             tabular_input = builder.prepare_tabular_input()
#             tabular_path = os.path.join(args.output_dir, f"tabular_input_{args.interval}.npz")
#             np.savez_compressed(
#                 tabular_path,
#                 X=tabular_input["X"],
#                 y_workhour=tabular_input["y"]["workhour"],
#                 y_consumption=tabular_input["y"]["consumption"],
#                 train_idx=tabular_input["train_idx"],
#                 val_idx=tabular_input["val_idx"],
#                 test_idx=tabular_input["test_idx"],
#                 feature_names=np.array(tabular_input["feature_names"], dtype=object),
#                 device_room_uris=np.array(tabular_input["device_room_uris"], dtype=object)
#             )
#             logger.info(f"Saved tabular baseline inputs to {tabular_path}")
        
#         # Prepare advanced tabular dataset if requested (TabularBuilderMixin method)
#         if args.build_advanced_tabular:
#             logger.info("Preparing advanced tabular dataset with feature engineering...")
#             advanced_tabular_input = builder.to_tabular(
#                 drop_sum=args.drop_sum,
#                 include_datetime=args.include_datetime_features,
#                 include_sensor_aggs=args.include_sensor_aggregates,
#                 lag_steps=args.lag_steps,
#                 roll_windows=args.rolling_windows
#             )
#             advanced_tabular_path = os.path.join(args.output_dir, f"advanced_tabular_input_{args.interval}.npz")
#             np.savez_compressed(
#                 advanced_tabular_path,
#                 X=advanced_tabular_input["X"],
#                 y_workhour=advanced_tabular_input["y"]["workhour"],
#                 y_consumption=advanced_tabular_input["y"]["consumption"],
#                 train_idx=advanced_tabular_input["train_idx"],
#                 val_idx=advanced_tabular_input["val_idx"],
#                 test_idx=advanced_tabular_input["test_idx"],
#                 feature_names=np.array(advanced_tabular_input["feature_names"], dtype=object)
#             )
#             logger.info(f"Saved advanced tabular inputs to {advanced_tabular_path}")
#             logger.info(f"Advanced tabular features: {len(advanced_tabular_input['feature_names'])} total features")
    
#     if args.graph_type in ['heterogeneous', 'both']:
#         logger.info("Building heterogeneous graph and temporal snapshots...")
        
#         # Build base heterogeneous graph
#         builder.build_base_hetero_graph()
        
#         # Build temporal heterogeneous graphs
#         builder.build_hetero_temporal_graphs()
        
#         # Prepare and save heterogeneous STGCN input
#         logger.info("Preparing heterogeneous STGCN input...")
#         hetero_stgcn_input = builder.prepare_hetero_stgcn_input()
#         hetero_torch_tensors = builder.convert_hetero_to_torch_tensors(hetero_stgcn_input, device=args.device)
        
#         hetero_output_path = os.path.join(args.output_dir, f"torch_input_hetero_{args.adjacency_type}_{args.interval}.pt")
#         torch.save(hetero_torch_tensors, hetero_output_path)
#         logger.info(f"Saved heterogeneous tensors to {hetero_output_path}")
        
#         # Save graph schema
#         schema_path = os.path.join(args.output_dir, f"hetero_graph_schema_{args.interval}.txt")
#         schema = builder.visualize_hetero_graph_schema(save_path=schema_path)
#         logger.info("Saved heterogeneous graph schema")

#     # Generate interactive propagation visualization if requested
#     if args.plot_propagation:
#         logger.info("Generating interactive propagation visualization...")
#         try:
#             propagation_html_path = os.path.join(args.plots_dir, 'propagation_visualization.html')
#             builder.create_interactive_plotly_visualization(output_file=propagation_html_path)
#             logger.info(f"Saved interactive propagation visualization to {propagation_html_path}")
#         except ImportError:
#             logger.warning("Plotly not available. Skipping interactive propagation visualization.")
#         except Exception as e:
#             logger.error(f"Error creating propagation visualization: {e}")

#     logger.info("Processing complete!")
    
#     # Print summary
#     logger.info("=" * 60)
#     logger.info("PROCESSING SUMMARY")
#     logger.info("=" * 60)
#     logger.info(f"Time buckets: {len(builder.time_buckets)} at {args.interval} intervals")
#     logger.info(f"Rooms: {len(builder.room_uris) if hasattr(builder, 'room_uris') else len(builder.adj_matrix_room_uris)}")
#     if hasattr(builder, 'node_mappings'):
#         logger.info(f"Devices: {len(builder.node_mappings['device'])}")
#         logger.info(f"Device-property combinations: {len(builder.node_mappings['property'])}")
#     logger.info(f"Train/Val/Test split: {len(builder.train_indices)}/{len(builder.val_indices)}/{len(builder.test_indices)} buckets")
#     if args.graph_type in ['homogeneous', 'both']:
#         logger.info(f"Homogeneous graph: {len(builder.homogeneous_graph.nodes)} nodes, {len(builder.homogeneous_graph.edges)} edges")
#         if args.build_tabular:
#             logger.info("Basic tabular dataset created")
#         if args.build_advanced_tabular:
#             logger.info("Advanced tabular dataset with feature engineering created")
#     if args.graph_type in ['heterogeneous', 'both']:
#         logger.info("Heterogeneous graph built with node types: room, device, property, outside, general")
#     logger.info("=" * 60)
