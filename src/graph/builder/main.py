import os
import joblib
import numpy as np
import torch
import pandas as pd
from typing import Dict, Any

from ..officegraph import OfficeGraph

from ...utils.missingness_plot import plot_missing_values
from ...utils.filename_util import get_data_filename

from .temporal import TemporalBuilderMixin
from .temporal_viz import TemporalVisualizerMixin
from .spatial import SpatialBuilderMixin
from .spatial_viz import SpatialVisualizerMixin
from .homo_graph import HomogGraphBuilderMixin
from .hetero_graph import HeteroGraphBuilderMixin
from .tabular import TabularBuilderMixin

# Main logging setup
import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


class OfficeGraphBuilder(
    SpatialBuilderMixin,
    SpatialVisualizerMixin,
    TemporalBuilderMixin,
    TemporalVisualizerMixin,
    HomogGraphBuilderMixin,
    HeteroGraphBuilderMixin,
    TabularBuilderMixin
    ):
    """
    Consolidated class to build and manipulate graphs from OfficeGraph data,
    including spatial relationships and temporal features.
    """
    
    def __init__(
            self, 
            office_graph: OfficeGraph,
            processed_data_dir: str = "data/processed",
            plots_dir: str = "output/visualizations"
    ):
        super().__init__()
        self.office_graph = office_graph
        
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
        
        # Save directories
        self.processed_data_dir = processed_data_dir
        os.makedirs(self.processed_data_dir, exist_ok=True)
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        self.missing_plots_dir = os.path.join(self.plots_dir, "missingness_plots")
        os.makedirs(self.missing_plots_dir, exist_ok=True)
        
        # Plot_suffix for missingness plots
        if self.office_graph.floors_to_load == [7]:
            self.plot_suffix = "(floor 7)"
        else:
            self.plot_suffix = "(all floors)"
        
        logger.info(f"Builder initialized.")
    
    
    #########################
    # Pipelines
    #########################
    
    def run_spatial_pipeline(self, args) -> None:
        logger.info("========== Running the spatial pipeline... ==========")
        
        # Setting static attributes
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
        self.static_room_attributes = static_attr_presets[args.static_attr_preset]
        
        # Initialize canonical room ordering
        self.initialize_canonical_ordering()

        # Initialize room polygons
        self.initialize_room_polygons(polygon_type=args.polygon_type,
                                      simplify_polygons=args.simplify_polygons,
                                      simplify_epsilon=args.simplify_epsilon)
        # Normalize room areas
        self.normalize_room_areas()
        
        logger.info("========== Finished the spatial pipeline. ==========")
        return None
    
    def run_adjacency_pipeline(self, args, adjacency_type: str) -> Dict[str, Any]:
        
        # Horizontal Adjacency
        horizontal_adj_dict = self.build_horizontal_adjacency_dict(
            mode                    = adjacency_type, 
            distance_threshold      = args.distance_threshold
        )
        horizontal_adj_matrix = self.combine_horizontal_adjacencies(
            horizontal_adj_dict     = horizontal_adj_dict
        )

        # Vertical Adjacency
        vertical_adj_matrix = self.build_vertical_adjacency(
            mode                    = adjacency_type,
            min_overlap_area        = 0.05, 
            min_weight              = 0
        )
        
        # Combined horizontal & vertical adjacency matrices
        full_adj_matrix = self.build_combined_room_to_room_adjacency(
            horizontal_adj_matrix   = horizontal_adj_matrix,
            vertical_adj_matrix     = vertical_adj_matrix
        )
        
        # Masked adjacency matrices
        full_masked_adj_matrices = self.create_masked_adjacency_matrices(
            adj_matrix              = full_adj_matrix,
            uri_str_list            = self.room_URIs_str
        )
        horizontal_masked_adj_matrices, vertical_masked_adj_matrices = (
            self.separate_masked_adjacencies(
                masked_adjs             = full_masked_adj_matrices,
                horizontal_adj_matrix   = horizontal_adj_matrix,
                vertical_adj_matrix     = vertical_adj_matrix
            )
        )
        
        # Outside adjacency
        outside_adj_dict = self.build_outside_adjacency(
            horizontal_adj_dict     = horizontal_adj_dict,
            mode                    = adjacency_type
        )
        outside_adj_vector = self.combine_outside_adjacencies(
            outside_adj_dict        = outside_adj_dict
        )
        
        adjacency_dict = {
            "room_URIs_str":                    self.room_URIs_str,
            "n_nodes":                          len(self.room_URIs_str),
            
            "horizontal_adj_matrix":            horizontal_adj_matrix,
            "vertical_adj_matrix":              vertical_adj_matrix,
            "full_adj_matrix":                  full_adj_matrix,

            "horizontal_masked_adj_matrices":   horizontal_masked_adj_matrices,
            "vertical_masked_adj_matrices":     vertical_masked_adj_matrices,
            "full_masked_adj_matrices":         full_masked_adj_matrices,

            "outside_adj_vector":               outside_adj_vector,
        }
        return adjacency_dict
    
    
    def run_temporal_pipeline(self, args):
        logger.info("========== Running the temporal pipeline... ==========")
        
        # Time buckets & Weekly blocks setup
        self.initialize_time_parameters(start_time=args.start_time, end_time=args.end_time,
                                        interval=args.interval, use_sundays=args.use_sundays)
        self.build_weekly_blocks()
        
        # Get weather data
        self.get_weather_data(weather_csv_path=args.weather_csv_path)
        
        # Get workhour labels
        self.get_workhour_labels(
            country_code=args.country_code, 
            workhour_start=args.workhour_start, 
            workhour_end=args.workhour_end,
            save=True
        )
        
        # Get (possible) target data
        self.get_consumption_values(
            consumption_dir=args.consumption_dir, 
            save=True
        )
        
        # Building different level feature DataFrames
        self.build_device_level_df()
        self.build_room_level_df()
        self.build_floor_level_df()
        self.build_building_level_df()
        
        # Building supporting DataFrames
        self.build_expanded_room_level_df()
        self.build_static_room_features_df()
        self.build_time_features_df()
        
        # Missingness plots for all 4 of these DFs
        if args.make_and_save_plots:
            plot_missing_values(self.device_level_df, df_name=f"Device-level DF {self.plot_suffix}",
                                save=True, output_dir=self.missing_plots_dir)
            plot_missing_values(self.room_level_df, df_name=f"Room-level DF {self.plot_suffix}",
                                save=True, output_dir=self.missing_plots_dir)
            plot_missing_values(self.floor_level_df, df_name=f"Floor-level DF {self.plot_suffix}",
                                save=True, output_dir=self.missing_plots_dir)
            plot_missing_values(self.building_level_df, df_name=f"Building-level DF {self.plot_suffix}",
                                save=True, output_dir=self.missing_plots_dir)
        
        logger.info("========== Finished the temporal pipeline. ==========")
        return None
    
    
    def run_tabular_pipeline(self, args) -> None:
        logger.info("========== Running the tabular pipeline... ==========")

        for build_mode in ["consumption_forecast", "measurement_forecast"]:

            self.build_base_tabular_df(build_mode=build_mode)
            self.engineer_tabular_features(build_mode=build_mode,
                lags=args.lags, windows=args.windows, shift_amount=args.shift_amount,
                integrate_weather=True)
            
            if args.make_and_save_plots:
                plot_missing_values(self.tabular_feature_df, df_name=f"Tabular Feature DF {self.plot_suffix}",
                                    save=True, output_dir=self.missing_plots_dir)

            fname_base = get_data_filename(file_type="dataframe", 
                                           interval=args.interval, weather_mode="feature",
                                           model_family="tabular", task_type=build_mode)
            self.save_df_as_parquet(self.tabular_feature_df, file_name=f"{fname_base}.parquet")

        
        logger.info("========== Finished the tabular pipeline. ==========")
        return None
    
    
    def run_homograph_pipeline(self, args) -> None:
        logger.info("========== Running the homogenous graph pipeline... ==========")
        
        # Adding static attributes and time features
        interim_df = (
            self.room_level_df_expanded
            .pipe(self.add_static_room_features_to_df)
            .pipe(self.add_time_features_to_df)
            .pipe(self.add_workhour_labels_to_df)
        )
        if args.make_and_save_plots:
            plot_missing_values(interim_df, df_name=f"Room-level DF, Expanded {self.plot_suffix}",
                                save=True, output_dir=self.missing_plots_dir)

        for mode in ("feature", "node"):
            if mode == "feature":
                df_fn           = self.add_weather_features_to_df
                adj_update_fn   = lambda a: a
            else: # mode == "node"
                df_fn           = self.add_weather_as_node_to_df
                adj_update_fn   = lambda a: self.update_adjacencies_for_weather_as_node(
                                            adjacency_dict=a, outside_URI_str="outside")

            # Filenames
            fnames = {
                file_type: get_data_filename(
                    file_type       = file_type,
                    interval        = args.interval,
                    weather_mode    = mode,
                    model_family    = "graph")
                for file_type in ("dataframe", "metadata")
            }
            
            # Dataframe
            df_out = df_fn(interim_df)
            self.save_df_as_parquet(df_out, f"{fnames['dataframe']}.parquet")

            # MetaData
            graph_data = {
                adj_type: adj_update_fn(self.run_adjacency_pipeline(
                    args            = args, 
                    adjacency_type  = adj_type))
                for adj_type in ("binary", "weighted")
            }
            self.save_metadata(graph_data, f"{fnames['metadata']}.joblib")

        logger.info("========== Finished the homogenous graph pipeline. ==========")
        return None

    def run_heterograph_pipeline(self, args):
        logger.info("========== Running the heterogeneous graph pipeline... ==========")

        # NOTE: we do want the option to have binary or weighted,
        #       however, there is no reason to create two different
        #       data files of temporal graphs. In the preparation phase,
        #       we can just switch them. Since we have the metadata from 
        #       homograph output.
        #       So, weighted here, all weights can be replaced with 1 later.
        adj = self.run_adjacency_pipeline(args, "weighted") 
        self.build_base_hetero_graph(
            adj["horizontal_adj_matrix"],
            adj["vertical_adj_matrix"],
            adj["outside_adj_vector"]
        )
        
        self.build_hetero_temporal_graphs()
        hetero_input = self.prepare_hetero_stgcn_input()
        
        fname_base = get_data_filename(file_type="hetero_input", interval=args.interval)
        self.save_hetero_input(hetero_input, file_name=f"{fname_base}.joblib")
        
        logger.info("========== Finished the heterogeneous graph pipeline. ==========")

    #########################
    # Save Helpers
    #########################
    
    def save_df_as_parquet(self, df: pd.DataFrame, file_name: str) -> None:
        """Convenience helper to save DataFrames as parquet files."""        
        
        # # Ensuring features are correctly categorical
        for col in ['room_uri_str', 'hasWindows','has_multiple_windows','isProperRoom', 'has_measurement']:
            if col in df.columns:
                df[col] = df[col].astype("category")
                # OR:   
                # codes = self.df[col].cat.codes.replace({-1: np.nan})
                # self.df[col] = codes.astype(float)
        
        parquet_file_path = os.path.join(self.processed_data_dir, file_name)
        df.to_parquet(parquet_file_path, engine='pyarrow')
        logger.info(f"Successfully saved DataFrame to {parquet_file_path}")
        return None
    
    def save_metadata(self, graph_data: Dict[str, Any], file_name: str) -> None:        
        """Convenience helper to save current metadata in the Builder."""        
        metadata_file_path = os.path.join(self.processed_data_dir, file_name)
        metadata = {
            "blocks": self.blocks,
            "block_size": self.block_size,
            "time_buckets": self.time_buckets,
        }
        metadata.update(graph_data)
        joblib.dump(metadata, metadata_file_path, compress=0)
        logger.info(f"Saved all metadata to {metadata_file_path}")
        return None

    def save_hetero_input(self, hetero_input: Dict[str, Any], file_name: str) -> None:
        path = os.path.join(self.processed_data_dir, file_name)
        torch.save(hetero_input, path)
        logger.info(f"Saved heterogeneous STGCN input to {path}")
        return None


def main():
    """Main function to build both tabular and graph data in one go."""
    from ...config.args import parse_args
    args = parse_args()
        
    from ..officegraph import OfficeGraph
    office_graph = OfficeGraph.from_pickles(floors_to_load = args.floors, data_dir=args.data_dir)
    
    # Setting up the builder
    builder = OfficeGraphBuilder(
        office_graph=office_graph, 
        processed_data_dir=args.processed_data_dir, 
        plots_dir=args.builder_plots_dir)     
    
    # Running the pipelines
    builder.run_spatial_pipeline(args=args)
    builder.run_temporal_pipeline(args=args)
    builder.run_tabular_pipeline(args=args)
    builder.run_homograph_pipeline(args=args)
    builder.run_heterograph_pipeline(args)


if __name__ == "__main__":
    main()