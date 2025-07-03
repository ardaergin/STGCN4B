import os
import joblib
import pandas as pd
from typing import Dict, Tuple, Any

from ..officegraph import OfficeGraph
from ...utils.missingness_plot import plot_missing_values
# Importing mixin classes:
from .temporal import TemporalBuilderMixin
from .temporal_viz import TemporalVisualizerMixin
from .spatial import SpatialBuilderMixin
from .spatial_viz import SpatialVisualizerMixin
from .homo_graph import HomogGraphBuilderMixin
# from .hetero_graph import HeteroGraphBuilderMixin
from .tabular import TabularBuilderMixin
from ...utils.filename_util import get_data_filename

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
    # HeteroGraphBuilderMixin,
    TabularBuilderMixin
    ):
    """
    Consolidated class to build and manipulate graphs from OfficeGraph data,
    including spatial relationships and temporal features.
    """
    
    def __init__(self, 
                 office_graph: OfficeGraph,
                 processed_data_dir: str = "data/processed",
                 plots_dir: str = "output/visualizations"):
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

    def run_common_spatial_pipeline(self, args) -> None:
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

    
    def run_temporal_pipeline(self, args):
        logger.info("========== Running the temporal pipeline... ==========")

        # Time buckets & Weekly blocks setup
        self.initialize_time_parameters(start_time=args.start_time, end_time=args.end_time,
                                        interval=args.interval, use_sundays=args.use_sundays)
        self.build_weekly_blocks()

        # Get weather data
        if args.incorporate_weather:
            self.get_weather_data(weather_csv_path=args.weather_csv_path)
        
        # Targets based on task
        if not args.task_type == "workhour_classification":
            self.get_workhour_labels(country_code=args.country_code, save=True)
            self.get_consumption_values(consumption_dir=args.consumption_dir, save=True)
        
        # Building different level feature DataFrames
        self.build_device_level_df()
        self.build_room_level_df()
        self.build_expanded_room_level_df()
        self.build_floor_level_df()
        self.build_building_level_df()

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


    def run_tabular_pipeline(self, args, build_mode: str) -> pd.DataFrame:
        logger.info("========== Running the tabular pipeline... ==========")
        
        self.build_base_tabular_df(build_mode=build_mode)
        self.engineer_tabular_features(build_mode=build_mode,
            lags=args.lags, windows=args.windows, shift_amount=args.shift_amount,
            integrate_weather=args.incorporate_weather)
        
        if args.make_and_save_plots:
            plot_missing_values(self.tabular_feature_df, df_name=f"Tabular Feature DF {self.plot_suffix}",
                                save=True, output_dir=self.missing_plots_dir)
        
        logger.info("========== Finished the tabular pipeline. ==========")
        return self.tabular_feature_df


    def run_homograph_pipeline(self, args) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        logger.info("========== Running the homogenous graph pipeline... ==========")
        
        for adjacency_type in ("binary", "weighted"):
            self.build_horizontal_adjacency(mode=adjacency_type, 
                                            distance_threshold=args.distance_threshold)
            self.build_vertical_adjacency(mode=adjacency_type,
                                        min_overlap_area=0.05, min_weight=0)
            # Combine horizontal & vertical adjacency matrices
            self.build_combined_room_to_room_adjacency()
            
            # Build outside adjacency
            outside_vector = None
            if args.incorporate_weather:
                self.build_outside_adjacency(mode=adjacency_type)
                outside_vector = getattr(self, "combined_outside_adj", None)
            
            # Adding static attributes
            final_df = self.add_static_room_features_to_df(df=self.room_level_df_expanded)
            if args.make_and_save_plots:
                plot_missing_values(final_df, df_name=f"Room-level DF, Expanded {self.plot_suffix}",
                                    save=True, output_dir=self.missing_plots_dir)
            
            # Incorporate weather data as outside 'space' if specified
            if args.incorporate_weather:
                logger.info("Integrating outside node with weather data into homogeneous graph...")
                adj_matrix, room_URIs_str, masked_adjacency_matrices, final_df = self.incorporate_weather_as_an_outside_room(room_level_df=final_df)
                if args.make_and_save_plots:
                    plot_missing_values(final_df, df_name=f"Room-level DF, Expanded, with Weather {self.plot_suffix}",
                        save=True, output_dir=self.missing_plots_dir)
            else:
                adj_matrix = self.room_to_room_adj_matrix
                room_URIs_str = self.adj_matrix_room_URIs_str
                masked_adjacency_matrices = self.create_masked_adjacency_matrices(self.room_to_room_adj_matrix, self.room_URIs_str)
            
            # Saving the adjacency data
            graph_data = {}
            graph_data["room_URIRefs"] = room_URIs_str.copy()
            graph_data["n_nodes"] = len(room_URIs_str)
            graph_data[adjacency_type] = {
                "adjacency_matrix":             adj_matrix.copy(),
                "outside_adjacency_vector":     outside_vector.copy() if outside_vector is not None else None,
                "masked_adjacency_matrices":    {k: v.copy() for k, v in masked_adjacency_matrices.items()},
            }
        
        logger.info("========== Finished the homogenous graph pipeline. ==========")
        return final_df, graph_data
    
    #########################
    # Save Helpers
    #########################

    def save_df_as_parquet(self, df: pd.DataFrame, file_name: str) -> None:
        """Convenience helper to save DataFrames as parquet files."""
        if 'room_URIRef' in df.columns:
            df['room_URIRef'] = df['room_URIRef'].astype(str)
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


def main():
    """Main function to build both tabular and graph data in one go."""
    from ...config.args import parse_args
    args = parse_args()
    
    # Loading OfficeGraph data (based on task type)
    # if args.task_type == "workhour_classification":
    #     args.floors = [7]
    # else:
    #     args.floors = [1,2,3,4,5,6,7]
    
    from ..officegraph import OfficeGraph
    office_graph = OfficeGraph.from_pickles(floors_to_load = args.floors, data_dir=args.data_dir)
    
    # Setting up the builder
    builder = OfficeGraphBuilder(
        office_graph=office_graph, 
        processed_data_dir=args.processed_data_dir, 
        plots_dir=args.builder_plots_dir)     
    
    # Spatial pipeline
    builder.run_common_spatial_pipeline(args)

    # Temporal pipeline
    builder.run_temporal_pipeline(args)
    
    ### Creating DataFrames & Saving ###

    # Workhour classification:
    # - I has just one floor, so, completely seperate underlying data.
    # - There is always no weather for workhour_classification.
    # - It is always interval="1h".
    # For the other two, I can save them together in one go.
    if args.task_type == "workhour_classification":
        # Ensuring correct args for this task:
        args.incorporate_weather = False
        args.interval = "1h"

        # Tabular
        tabular_df = builder.run_tabular_pipeline(args=args, build_mode="workhour_classification")
        fname_base = get_data_filename(
            file_type="dataframe", task_type="workhour_classification",
            model_family="tabular")
        builder.save_df_as_parquet(tabular_df, file_name=f"{fname_base}.parquet")

        # Homogeneous Graph
        homograph_df, graph_data = builder.run_homograph_pipeline(args=args)
        fname_base = get_data_filename(
            file_type="dataframe", task_type="workhour_classification",
            model_family="graph")
        builder.save_df_as_parquet(homograph_df, file_name=f"{fname_base}.parquet")

        # Metadata
        fname_base = get_data_filename(file_type="metadata", task_type="workhour_classification")
        builder.save_metadata(graph_data=graph_data, file_name=f"{fname_base}.joblib")
        
    else:
        for build_mode in ["consumption_forecast", "measurement_forecast"]:

            tabular_df = builder.run_tabular_pipeline(args=args, build_mode=build_mode)
            fname_base = get_data_filename(
                file_type="dataframe", task_type=build_mode,
                model_family="tabular", 
                interval=args.interval, incorporate_weather=args.incorporate_weather)
            builder.save_df_as_parquet(tabular_df, file_name=f"{fname_base}.parquet")
                    
        # Homogeneous Graph
        # NOTE: The files for measurement and consumption are the same for HomoGraph, unlike Tabular.
        homograph_df, graph_data = builder.run_homograph_pipeline(args=args)
        fname_base = get_data_filename(
            file_type="dataframe",
            model_family="graph",
            interval=args.interval, incorporate_weather=args.incorporate_weather)
        builder.save_df_as_parquet(homograph_df, file_name=f"{fname_base}.parquet")
        
        # Metadata
        fname_base = get_data_filename(
            file_type="metadata",
            interval=args.interval, incorporate_weather=args.incorporate_weather)
        builder.save_metadata(graph_data=graph_data, file_name=f"{fname_base}.joblib")
        
if __name__ == "__main__":
    main()