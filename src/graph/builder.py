import os
import pickle
import torch
import networkx as nx
import numpy as np
from rdflib import URIRef
from typing import Dict, List, Tuple, Set, Any
from datetime import datetime, timedelta
import logging
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)

from ..data.FloorPlan import polygon_utils

class OfficeGraphBuilder:
    """
    Consolidated class to build and manipulate graphs from OfficeGraph data,
    including spatial relationships and temporal features.
    """
    
    def __init__(self, office_graph, consumption_dir: str = "data/consumption"):
        """
        Initialize the builder with the OfficeGraph instance.
        
        Args:
            office_graph: The OfficeGraph instance containing the data.
        """
        self.office_graph = office_graph
        self.consumption_dir = consumption_dir
        
        self.room_uris = list(self.office_graph.rooms.keys())
        self.room_names = {}
        for room_uri in self.room_uris:
            self.room_names[room_uri] = self.office_graph.rooms[room_uri].room_number

        # Time-related properties
        self.start_time = None
        self.end_time = None
        self.time_buckets = None
        self.temporal_graphs = None
        
        # Property configuration
        self.ignored_property_types = {"DeviceStatus", "BatteryLevel"}
        self.used_property_types = ["Temperature", "CO2Level", "Contact", "Humidity"]

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
            'polygons_doc.centroid',
            'polygons_doc.perimeter',
            'polygons_doc.width',
            'polygons_doc.height',
            'polygons_doc.compactness',
            'polygons_doc.rect_fit',
            'polygons_doc.aspect_ratio'
        ]

    #############################
    # Time-related methods
    #############################
        
    def initialize_time_parameters(self, 
                                  start_time: str = "2022-03-07 00:00:00", # Monday
                                  # The data starts at 03-01 (Tuesday), but we start at 03-07 (Monday)
                                  end_time: str = "2023-01-29 00:00:00", # 01-29, Sunday
                                  # The data ends at 01-31 (Tuesday), but we end at 01-29 (Monday)
                                  interval: str   = "1h",
                                  use_sundays: bool = False):
        """
        Initialize time-related parameters and create time buckets.
        
        Args:
            start_time: Start time for analysis in format "YYYY-MM-DD HH:MM:SS"
            end_time: End time for analysis in format "YYYY-MM-DD HH:MM:SS"
            interval: Frequency (15min, 30min, 1h, 2h…)
            use_sundays: Whether to include Sundays in time buckets
        """
        self.start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        self.end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        self.interval = interval
        self.use_sundays = use_sundays

        # Build buckets at arbitrary freq (15min, 30min, 1h, 2h, …)
        full_index = pd.date_range(self.start_time,
                                   self.end_time,
                                   freq=self.interval,
                                   inclusive="left")
        if not self.use_sundays:
            full_index = full_index[full_index.weekday != 6] # Sunday = 6
        
        # store as list of (start, end)
        off = pd.tseries.frequencies.to_offset(self.interval)
        self.time_buckets = [(ts, ts + off) for ts in full_index]
        self.time_buckets = [(ts.to_pydatetime(), (ts + off).to_pydatetime()) for ts in full_index]
        
        logger.info(f"Created {len(self.time_buckets)} buckets at frequency {self.interval}")
        
    def split_time_buckets(self,
                        train_blocks=4,
                        val_blocks=1,
                        test_blocks=1,
                        seed=2658918):
        """
        Split time buckets into train, validation, and test sets using block-wise sampling.
        Takes the direct number of blocks for each split and requires that the total requested
        blocks can divide evenly into the available blocks.
        
        Args:
            train_blocks: Number of blocks for training
            val_blocks: Number of blocks for validation
            test_blocks: Number of blocks for testing
            seed: Random seed for reproducibility
            
        Raises:
            ValueError: If the number of available blocks is not divisible by the 
                        total requested blocks
        """
        if not self.time_buckets:
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")
        if seed is not None:
            np.random.seed(seed)

        # Get all time indices
        time_indices = list(range(len(self.time_buckets)))
        
        # Compute how many buckets per day, then per week (or 6-day)
        # (so 96/day for 15min, 48/day for 30T, 24/day for 1H, etc.)
        offset = pd.Timedelta(self.interval)
        buckets_per_day = int(pd.Timedelta("1D") / offset)
        days_per_block = 7 if self.use_sundays else 6
        block_size = buckets_per_day * days_per_block
        logger.info(f"Using {days_per_block}-day blocks, {block_size} buckets at {self.interval} each.")

        # Create blocks of contiguous time points
        blocks = []
        for i in range(0, len(time_indices), block_size):
            # Take up to block_size indices (last block might be smaller)
            block = time_indices[i:i+block_size]
            blocks.append(block)
        
        n_blocks = len(blocks)
        logger.info(f"Created {n_blocks} blocks of data (each 1 week)")
        
        # Calculate total requested blocks
        total_requested_blocks = train_blocks + val_blocks + test_blocks
        
        # Check if the requested blocks divide evenly into available blocks
        n_extra_blocks = n_blocks % total_requested_blocks
        if n_extra_blocks != 0:
            logger.warning(f"The week block number ({n_blocks}) is not divisible by the "
                        f"requested block counts ({total_requested_blocks}). "
                        f"There are {n_extra_blocks} extra blocks. "
                        f"So, {n_extra_blocks} randomly chosen blocks will be assigned to train.")
            
            # Randomly select n_extra_blocks indices
            extra_block_indices = np.random.choice(range(n_blocks), n_extra_blocks, replace=False)
            
            # Initialize train_indices list for extra blocks
            train_indices = []
            
            # Add the selected blocks to training and remove them from blocks
            # We need to process in reverse order to avoid index shifting during removal
            for idx in sorted(extra_block_indices, reverse=True):
                train_indices.extend(blocks[idx])
                blocks.pop(idx)
            
            # Update n_blocks after removal
            n_blocks = len(blocks)
        else:
            # Initialize empty train_indices if no extra blocks
            train_indices = []
        
        # Initialize rest of the indices
        val_indices = []
        test_indices = []

        # Calculate the repeat factor - how many times to repeat the pattern
        repeat_factor = n_blocks // total_requested_blocks
        
        # Define the basic pattern
        basic_pattern = ["train"] * train_blocks + ["val"] * val_blocks + ["test"] * test_blocks
        
        # Repeat the pattern the necessary number of times
        sampling_pattern = basic_pattern * repeat_factor        
        
        # Shuffle the sampling pattern
        np.random.shuffle(sampling_pattern)
        
        # Assign blocks to splits
        for i, block in enumerate(blocks):
            split_type = sampling_pattern[i]
            
            # Assign the block to the corresponding split
            if split_type == "train":
                train_indices.extend(block)
            elif split_type == "val":
                val_indices.extend(block)
            else:  # "test"
                test_indices.extend(block)
        
        # Sort indices within each split to maintain temporal order
        train_indices.sort()
        val_indices.sort()
        test_indices.sort()
        
        # Store the indices in the class
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        logger.info(f"Data split: Train={len(train_indices)} samples ({(train_blocks * repeat_factor)+n_extra_blocks} blocks), "
                    f"Val={len(val_indices)} samples ({val_blocks * repeat_factor} blocks), "
                    f"Test={len(test_indices)} samples ({test_blocks * repeat_factor} blocks)")
        logger.info(f"Pattern [{','.join([str(b) for b in basic_pattern])}] repeated {repeat_factor} times")
    
    #############################
    # Polygons
    #############################
        
    def initialize_room_polygons(self, polygon_type='doc', simplify_polygons=False, simplify_epsilon=0.0):
        """
        Initialize shapely Polygon objects for each room based on the Room's polygon data.
        Also extracts and stores room areas for later use.
        
        Args:
            polygon_type (str): Which polygon data to use - 'geo' or 'doc'
            simplify_polygons (bool): Whether to simplify the polygons
            simplify_epsilon (float): Epsilon value for polygon simplification. 
                                    0.0 means no simplification.
        """
        self.room_polygons = {}
        self.areas = {}  # Store room areas
        self.polygon_type = polygon_type  # Store the polygon type for later reference
        
        for room_uri, room in self.office_graph.rooms.items():
            # Get the appropriate polygon data based on type
            if polygon_type == 'geo':
                points_2d = room.polygons_geo.get('points_2d', [])
                area = room.polygons_geo.get('area')
            elif polygon_type == 'doc':
                points_2d = room.polygons_doc.get('points_2d', [])
                area = room.polygons_doc.get('area')
            else:
                logger.warning(f"Invalid polygon_type: {polygon_type}. Using 'geo' as default.")
                points_2d = room.polygons_geo.get('points_2d', [])
                area = room.polygons_geo.get('area')
            
            # Store area if available
            if area is not None:
                self.areas[room_uri] = area
            
            # Check if we have valid polygon data
            if not points_2d or len(points_2d) < 3:
                logger.warning(f"Room {room_uri} has no valid polygon data for type '{polygon_type}'.")
                continue
            
            # Create a shapely Polygon
            polygon = Polygon(points_2d)
            self.room_polygons[room_uri] = polygon
            
            # Simplify if requested
            if simplify_polygons:
                simplified_coords = polygon_utils.simplify_polygon(points_2d, epsilon=simplify_epsilon)
                simplified_polygon = Polygon(simplified_coords)
                self.room_polygons[room_uri] = simplified_polygon
                logger.debug(f"Simplified room {room_uri} polygon from {len(points_2d)} to {len(simplified_coords)} vertices.")
            else:
                self.room_polygons[room_uri] = polygon
        
        logger.info(f"Initialized {len(self.room_polygons)} room polygons using '{polygon_type}' data.")
        logger.info(f"Extracted {len(self.areas)} room areas.")
    
    def normalize_room_areas(self) -> None:
        """
        Calculate both min-max and proportion normalized room areas and store them 
        as class attributes. Uses the areas stored during initialize_room_polygons().
        
        This method populates:
        - self.norm_areas_minmax: Dictionary mapping room URIs to min-max normalized areas (0-1 scale)
        - self.norm_areas_prop: Dictionary mapping room URIs to proportion normalized areas (fraction of total)
        
        Returns:
            tuple: (min_max_normalized_areas, proportion_normalized_areas) dictionaries
        """
        # Initialize empty dictionaries for normalized areas
        self.norm_areas_minmax = {}
        self.norm_areas_prop = {}
        
        if not hasattr(self, 'areas') or not self.areas:
            logger.warning("No areas available for normalization. Call initialize_room_polygons first.")
            return self.norm_areas_minmax, self.norm_areas_prop
        
        # Get total area and min/max values for calculations
        total_area = sum(self.areas.values())
        min_area = min(self.areas.values())
        max_area = max(self.areas.values())
        
        # Perform both normalizations at once
        for uri, area in self.areas.items():
            # Handle edge case where min and max are the same
            if max_area == min_area:
                self.norm_areas_minmax[uri] = 1.0
            else:
                # Min-max normalization: (value - min) / (max - min)
                self.norm_areas_minmax[uri] = (area - min_area) / (max_area - min_area)
            
            # Proportion normalization
            if total_area <= 0:
                self.norm_areas_prop[uri] = 0.0
            else:
                self.norm_areas_prop[uri] = area / total_area
        
        logger.info(f"Calculated min-max and proportion normalizations for {len(self.areas)} room areas")

    def plot_floor_plan(self, 
                    normalization='min_max',
                    show_room_ids=True,
                    figsize=(12, 10), 
                    colormap='turbo'):
        """
        Plot the floor plan with rooms colored according to their normalized areas.
        
        Args:
            normalization (str): Which normalization to use - 'min_max' or 'proportion'
            show_room_ids (bool): Whether to show room IDs in the plot
            figsize (tuple): Figure size
            colormap (str): Matplotlib colormap name for coloring rooms
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        
        # Check if we have the required data
        if not hasattr(self, 'room_polygons') or not self.room_polygons:
            logger.error("No polygons available for plotting. Call initialize_room_polygons first.")
            return None
            
        # Check if normalized areas are calculated
        if not hasattr(self, 'norm_areas_minmax') or not self.norm_areas_minmax:
            logger.info("Calculating area normalizations...")
            self.normalize_room_areas()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the color map
        cmap = plt.get_cmap(colormap)
        
        # Determine which normalization to use
        if normalization == 'min_max':
            normalized_areas = self.norm_areas_minmax
        elif normalization == 'proportion':
            normalized_areas = self.norm_areas_prop
        else:
            logger.warning(f"Unknown normalization type: {normalization}. Using 'min_max'.")
            normalized_areas = self.norm_areas_minmax
        
        # Plot each room
        room_ids = []
        values = []
        
        for room_uri, polygon in self.room_polygons.items():
            # Get room color based on normalized area (default to 0.5 if missing)
            norm_value = normalized_areas.get(room_uri, 0.5)
            values.append(norm_value)
            
            # Get room ID for display
            if hasattr(self.office_graph, 'rooms') and room_uri in self.office_graph.rooms:
                room = self.office_graph.rooms[room_uri]
                display_id = room.room_number or str(room_uri).split('/')[-1]
                room_ids.append(display_id)
            else:
                # Extract just the room number part if it's prefixed
                display_id = str(room_uri).split('/')[-1]
                if 'roomname_' in display_id:
                    display_id = display_id.split('roomname_')[-1]
                room_ids.append(display_id)
            
            # Plot the polygon
            color = cmap(norm_value)
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.7, fc=color, ec='black')
            
            # Show room ID if requested
            if show_room_ids:
                centroid = polygon.centroid
                ax.text(centroid.x, centroid.y, display_id,
                    ha='center', va='center', fontsize=8, 
                    color='black', fontweight='bold')
        
        # Set aspect equal to preserve shape
        ax.set_aspect('equal')
        
        # Get axis limits
        min_x = min(polygon.bounds[0] for polygon in self.room_polygons.values())
        max_x = max(polygon.bounds[2] for polygon in self.room_polygons.values())
        min_y = min(polygon.bounds[1] for polygon in self.room_polygons.values())
        max_y = max(polygon.bounds[3] for polygon in self.room_polygons.values())
        
        # Add some padding
        padding = 0.05 * max(max_x - min_x, max_y - min_y)
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)
        
        # Set title
        plt.title(f"Floor Plan with {normalization.replace('_', '-')} Normalized Areas")
        
        # Add color bar to show area scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'{normalization.capitalize()} Normalized Room Area')
        
        # Add axes labels based on polygon type
        coord_type = "Geographic" if self.polygon_type == "geo" else "Document"
        ax.set_xlabel(f"{coord_type} X Coordinate")
        ax.set_ylabel(f"{coord_type} Y Coordinate")
        
        plt.tight_layout()


    #############################
    # Adj Matrix
    #############################

    def calculate_binary_adjacency(
        self,
        distance_threshold: float = 5.0
        ) -> pd.DataFrame:
        """
        Calculate binary adjacency between rooms based on their polygons.
        Two rooms are considered adjacent if their polygons are within distance_threshold.
        
        Args:
            distance_threshold (float): Maximum distance (in meters) for rooms to be considered adjacent
            
        Returns:
            DataFrame: Adjacency matrix as a pandas DataFrame
        """
        if not self.room_polygons:
            raise ValueError("No room polygons found. Make sure to call initialize_room_polygons().")
        
        # Get list of room URIs as strings (to be used as DataFrame indices)
        room_uris_str = [str(uri) for uri in self.room_polygons.keys()]
        # Store mapping from string URI to original URI object for later use
        self.uri_str_to_obj = {str(uri): uri for uri in self.room_polygons.keys()}
        
        # Initialize adjacency matrix with zeros
        adj_df = pd.DataFrame(0, index=room_uris_str, columns=room_uris_str)
        
        # Fill adjacency matrix
        for i, room1_id in enumerate(room_uris_str):
            room1_uri = self.uri_str_to_obj[room1_id]
            room1_poly = self.room_polygons.get(room1_uri)
            
            if room1_poly is None:
                continue
                
            for j, room2_id in enumerate(room_uris_str[i+1:], i+1):
                room2_uri = self.uri_str_to_obj[room2_id]
                room2_poly = self.room_polygons.get(room2_uri)
                
                if room2_poly is None:
                    continue
                
                # Calculate distance between the polygons
                distance = room1_poly.distance(room2_poly)
                
                # Set adjacency based on distance threshold
                if distance <= distance_threshold:
                    adj_df.at[room1_id, room2_id] = 1
                    adj_df.at[room2_id, room1_id] = 1  # Symmetric

        return adj_df
    
    def calculate_proportional_boundary_adjacency(
        self,
        distance_threshold: float = 5.0,
        min_shared_length: float = 0.01,
        min_weight: float = 0.0
        ) -> pd.DataFrame:
        """
        Calculate a non-symmetric adjacency matrix where
        A[i,j] = (shared boundary length between room i and j) / perimeter(room i),
        but only if the two rooms are considered adjacent in the binary adjacency matrix.
        
        Args:
            distance_threshold: max distance (in meters) to consider adjacency
            min_weight: minimum weight to assign when binary adjacency exists but 
                        no significant boundary is shared (default: 0.001)
            
        Returns:
            DataFrame: adjacency matrix A (rows i to j)
        """
        if not self.room_polygons:
            raise ValueError("No room polygons found. Make sure to call initialize_room_polygons().")
        
        # Get binary adjacency
        binary_adj_df = self.calculate_binary_adjacency(distance_threshold=distance_threshold)
            
        # Reuse the room_ids from binary adjacency for consistency
        room_ids = binary_adj_df.index.tolist()
        
        # Prepare DataFrame
        adj_df = pd.DataFrame(0.0, index=room_ids, columns=room_ids)
        
        # Precompute perimeters
        perimeters = {}
        for rid in room_ids:
            uri = self.uri_str_to_obj[rid]
            if uri in self.room_polygons:
                perimeters[rid] = self.room_polygons[uri].length
            else:
                perimeters[rid] = 0.0
        
        # Process only room pairs that are adjacent according to binary adjacency
        for i, r1 in enumerate(room_ids):
            uri1 = self.uri_str_to_obj[r1]
            poly1 = self.room_polygons.get(uri1)
            if poly1 is None:
                continue
                
            p1 = perimeters[r1]
            if p1 <= 0:
                continue
            
            for j, r2 in enumerate(room_ids):
                if i == j:
                    continue
                    
                # Only process if binary adjacency exists
                if binary_adj_df.at[r1, r2] <= 0:
                    continue
                
                uri2 = self.uri_str_to_obj[r2]
                poly2 = self.room_polygons.get(uri2)
                if poly2 is None:
                    continue
                
                # Find shared boundary: intersect boundary of r1 with buffered boundary of r2
                try:
                    shared = poly1.boundary.intersection(
                        poly2.boundary.buffer(distance_threshold/2)
                    )
                except Exception as e:
                    logger.warning(f"Boundary intersection error {r1}-{r2}: {e}")
                    # Even if there's an error, we still want to preserve the binary adjacency
                    adj_df.at[r1, r2] = min_weight
                    continue
                
                # Compute total length of any lines
                length = 0.0
                if shared.geom_type == 'LineString':
                    length = shared.length
                elif shared.geom_type == 'MultiLineString':
                    length = sum(seg.length for seg in shared.geoms)
                
                if length > min_shared_length:
                    # Proportional weight: shared boundary / perimeter
                    adj_df.at[r1, r2] = length / p1
                else:
                    # assign the minimum weight if too small
                    adj_df.at[r1, r2] = min_weight
        
        return adj_df

    def build_room_to_room_adjacency(self, 
                                    matrix_type="binary", 
                                    distance_threshold=5.0):
        """
        Build the room-to-room adjacency matrix and store it in class attributes.
        
        Args:
            matrix_type: Kind of adjacency. Options:
                - 'binary': Basic binary adjacency based on proximity
                - 'weighted': Weighted adjacency where each room's influence is proportional to target's perimeter
            distance_threshold: Maximum distance for considering rooms adjacent (in meters)
            
        Returns:
            tuple: (adjacency_matrix, room_uris)
        """
        if not self.room_polygons:
            raise ValueError("No room polygons found. Make sure to call initialize_room_polygons().")
        
        if matrix_type == "binary":
            # Get binary adjacency
            adj_df = self.calculate_binary_adjacency(distance_threshold=distance_threshold)
            self.adjacency_type = "binary"
            
        elif matrix_type == "weighted":
            adj_df = self.calculate_proportional_boundary_adjacency(distance_threshold=distance_threshold)
            self.adjacency_type = "weighted"
        
        else:
            raise ValueError(f"Unknown adjacency kind: {matrix_type}. Use 'binary' or 'weighted'.")
        
        # Store both the DataFrame and the numpy array
        self.adjacency_matrix_df = adj_df
        self.room_to_room_adj_matrix = adj_df.values
        
        # Store the room URIs in the same order as in the adjacency matrix
        self.adj_matrix_room_uris = [self.uri_str_to_obj[uri_str] for uri_str in adj_df.index]
        
        logger.info(f"Built room-to-room adjacency matrix with shape {self.room_to_room_adj_matrix.shape}")
        return adj_df, self.adj_matrix_room_uris

    def plot_adjacency_matrix(self, figsize=(10, 8), title=None, show_room_ids=True, cmap='Blues'):
        """
        Plot the room-to-room adjacency matrix as a heatmap.
        
        Args:
            figsize (tuple): Figure size
            title (str, optional): Plot title. If None, a default title is used
            show_room_ids (bool): Whether to show room IDs on axes
            cmap (str): Matplotlib colormap name
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        
        if not hasattr(self, 'room_to_room_adj_matrix') or self.room_to_room_adj_matrix is None:
            raise ValueError("No room-to-room adjacency matrix found. Make sure to call build_room_to_room_adjacency().")
        
        if not hasattr(self, 'adj_matrix_room_uris') or self.adj_matrix_room_uris is None:
            raise ValueError("Room URIs in adjacency matrix order not found. Make sure to call build_room_to_room_adjacency().")
        
        fig = plt.figure(figsize=figsize)
        plt.imshow(self.room_to_room_adj_matrix, cmap=cmap)
        
        # Add colorbar with label based on adjacency type
        if hasattr(self, 'adjacency_type') and self.adjacency_type == 'weighted':
            plt.colorbar(label='Proportion of shared boundary')
        else:
            plt.colorbar(label='Connection strength')
        
        # Use pre-stored room names for labels
        if show_room_ids and len(self.adj_matrix_room_uris) <= 50:  # Only show labels if not too many rooms
            labels = [self.room_names.get(uri, str(uri)) for uri in self.adj_matrix_room_uris]
            plt.xticks(range(len(self.adj_matrix_room_uris)), labels, rotation=90)
            plt.yticks(range(len(self.adj_matrix_room_uris)), labels)
        elif not show_room_ids:
            plt.xticks([])
            plt.yticks([])
        else:
            # Too many rooms, just show indices
            plt.xticks(range(0, len(self.adj_matrix_room_uris), 5))
            plt.yticks(range(0, len(self.adj_matrix_room_uris), 5))
        
        # Set title
        if title is None:
            if hasattr(self, 'adjacency_type'):
                title = f"Room Adjacency Matrix ({self.adjacency_type})"
            else:
                title = "Room Adjacency Matrix"
        plt.title(title)
        
        # Add axis labels
        plt.xlabel("Room")
        plt.ylabel("Room")
        
        plt.tight_layout()

    def calculate_information_propagation_masks(self):
        """
        Calculate a series of masking matrices representing information propagation
        from rooms with devices to other rooms in the building.
        
        The function continues calculating propagation steps until equilibrium is reached
        (no new rooms can receive information).
        
        Returns:
            Dictionary mapping step indices (starting from 0) to masking matrices (numpy arrays)
            where 1 indicates a room can pass information and 0 indicates it is masked
        """
        if not hasattr(self, 'adjacency_matrix_df') or self.adjacency_matrix_df is None:
            raise ValueError("Room adjacency matrix not found. Run build_room_to_room_adjacency first.")
        
        # Get device presence information
        room_has_device = np.zeros(len(self.adj_matrix_room_uris), dtype=bool)
        for i, room_uri in enumerate(self.adj_matrix_room_uris):
            if room_uri in self.office_graph.rooms:
                room = self.office_graph.rooms[room_uri]
                if room.devices:  # Room has at least one device
                    room_has_device[i] = True
        
        logger.info(f"Found {room_has_device.sum()} rooms with devices out of {len(room_has_device)} total rooms")
        
        # Get adjacency matrix as numpy array
        adjacency = self.room_to_room_adj_matrix.copy()
        
        # Initialize masks dictionary
        masks = {}
        
        # Step 0: Only rooms with devices can pass information
        can_pass_info = room_has_device.copy()
        
        # Create initial mask matrix
        mask_matrix = np.zeros_like(adjacency)
        for i in range(len(can_pass_info)):
            if can_pass_info[i]:
                mask_matrix[i, :] = 1  # Room can pass information to others
        
        masks[0] = mask_matrix
        
        # Continue propagation until equilibrium (no new rooms activated)
        step = 1
        while True:
            # Identify newly activated rooms (those that receive information in this step)
            newly_activated = np.zeros_like(can_pass_info)
            
            for i in range(len(can_pass_info)):
                if not can_pass_info[i]:  # Room doesn't have information yet
                    # Check if it receives info from any room that can pass info
                    for j in range(len(can_pass_info)):
                        if can_pass_info[j] and adjacency[j, i] > 0:
                            newly_activated[i] = True
                            break
            
            # If no new rooms were activated, we've reached equilibrium
            if not np.any(newly_activated):
                logger.info(f"Information propagation reached equilibrium after {step} steps")
                break
                
            # Update the can_pass_info array
            can_pass_info = np.logical_or(can_pass_info, newly_activated)
            
            # Create new mask matrix
            mask_matrix = np.zeros_like(adjacency)
            for i in range(len(can_pass_info)):
                if can_pass_info[i]:
                    mask_matrix[i, :] = 1  # Room can pass information to others
            
            masks[step] = mask_matrix
            logger.info(f"Step {step}: {newly_activated.sum()} new rooms can pass information")
            
            step += 1
            
            # Safety check to prevent infinite loops
            if step > len(can_pass_info):
                logger.warning(f"Stopping propagation after {step} steps (exceeded number of rooms)")
                break
        
        # Log summary
        active_rooms = [np.sum(masks[step].sum(axis=1) > 0) for step in masks]
        logger.info(f"Created {len(masks)} information propagation masks")
        logger.info(f"Rooms that can pass information at each step: {active_rooms}")
        
        return masks

    def apply_masks_to_adjacency(self, masks=None):
        """
        Apply information propagation masks to the adjacency matrix 
        to create multiple adjacency matrices representing the progressive flow
        of information through the building.
        
        Args:
            masks: Dictionary mapping step indices to masking matrices.
                If None, calls calculate_information_propagation_masks
        
        Returns:
            Dictionary mapping step indices to masked adjacency matrices
        """
        if not hasattr(self, 'room_to_room_adj_matrix') or self.room_to_room_adj_matrix is None:
            raise ValueError("Room adjacency matrix not found. Run build_room_to_room_adjacency first.")
        
        # Calculate masks if not provided
        if masks is None:
            masks = self.calculate_information_propagation_masks()
        
        # Get adjacency matrix as numpy array
        adjacency = self.room_to_room_adj_matrix.copy()
        
        # Apply masks to create multiple adjacency matrices
        masked_adjacencies = {}
        
        for step, mask in masks.items():
            # Element-wise multiplication of mask and adjacency
            masked_adj = adjacency * mask
            masked_adjacencies[step] = masked_adj
        
        self.masked_adjacencies = masked_adjacencies
        logger.info(f"Created {len(masked_adjacencies)} masked adjacency matrices")
        
        return masked_adjacencies
    
    def create_interactive_plotly_visualization(self, output_file='output/builder/propagation_visualization.html'):
        """
        Create an interactive Plotly visualization of information propagation
        and export it to a standalone HTML file.
        
        Device rooms are colored green at Step 0, and subsequent information propagation
        is shown in shades of blue.
        
        Args:
            output_file: Path to save the HTML output file
            
        Returns:
            Plotly figure object
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        if not hasattr(self, 'masked_adjacencies') or not self.masked_adjacencies:
            raise ValueError("Masked adjacency matrices not found. Run apply_masks_to_adjacency first.")
            
        if not hasattr(self, 'room_polygons') or not self.room_polygons:
            raise ValueError("Room polygons not found. Run initialize_room_polygons first.")
        
        # Define the number of steps
        n_steps = len(self.masked_adjacencies)
        
        # Extract information about which rooms can pass info at each step
        room_info_by_step = {}
        for step in range(n_steps):
            mask = self.masked_adjacencies[step]
            can_pass_info = mask.sum(axis=1) > 0  # Rooms that can pass info
            room_info_by_step[step] = can_pass_info
        
        # Define colors
        device_color = '#2ca02c'  # Forest green for device rooms
        inactive_color = '#f0f0f0'  # Light gray for inactive rooms
        # Shades of blue for rooms activated in steps 1-N
        propagation_colors = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
        
        # Create a plotly figure with steps as frames
        fig = go.Figure()
        
        # Add frames for each step
        frames = []
        for step in range(n_steps):
            frame_data = []
            
            # For each room polygon
            for i, room_uri in enumerate(self.adj_matrix_room_uris):
                if room_uri in self.room_polygons:
                    polygon = self.room_polygons[room_uri]
                    
                    # Extract polygon coordinates
                    x, y = polygon.exterior.xy
                    
                    # Check if this is a device room (active at step 0)
                    is_device_room = room_info_by_step[0][i]
                    
                    # Determine when this room gets activated
                    activation_step = n_steps  # Default: not activated
                    for s in range(n_steps):
                        if room_info_by_step[s][i]:
                            activation_step = s
                            break
                    
                    # Determine color based on activation status for this step
                    if activation_step <= step:
                        # Room is active at this step
                        if is_device_room:
                            # Device rooms always show in green
                            color = device_color
                        else:
                            # Non-device rooms show in blue based on when they were activated
                            blue_idx = min(activation_step - 1, len(propagation_colors) - 1)
                            color = propagation_colors[blue_idx]
                    else:
                        # Not yet activated
                        color = inactive_color
                    
                    # Create room label
                    room_id = self.room_names.get(room_uri, str(room_uri).split('/')[-1])
                    if is_device_room:
                        room_id = f"{room_id}*"  # Mark device rooms
                    
                    # Create a polygon for this room
                    room_trace = go.Scatter(
                        x=list(x) + [x[0]],  # Close the polygon
                        y=list(y) + [y[0]],
                        fill='toself',
                        fillcolor=color,
                        line=dict(color='black', width=1),
                        text=room_id,
                        hoverinfo='text',
                        showlegend=False
                    )
                    
                    frame_data.append(room_trace)
            
            # Create frame for this step
            frame = go.Frame(
                data=frame_data,
                name=f"Step {step}",
                layout=go.Layout(
                    title=f"Information Propagation - Step {step}: "
                        f"{np.sum(room_info_by_step[step])}/{len(self.adj_matrix_room_uris)} rooms can pass information"
                )
            )
            frames.append(frame)
        
        # Add the initial data (step 0)
        fig.add_traces(frames[0].data)
        fig.frames = frames
        
        # Set up slider control
        sliders = [{
            'steps': [
                {
                    'method': 'animate',
                    'args': [
                        [f"Step {i}"],
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f"Step {i}"
                } for i in range(n_steps)
            ],
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Step: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0
        }]
        
        # Play and pause buttons
        updatemenus = [{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [
                        None,
                        {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                        }
                    ]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [
                        [None],
                        {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }
                    ]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'type': 'buttons',
            'x': 0.1,
            'y': 0,
            'xanchor': 'right',
            'yanchor': 'top'
        }]
        
        # Update figure layout
        fig.update_layout(
            title="Information Propagation Through Building",
            autosize=True,
            width=900,
            height=700,
            margin=dict(l=50, r=50, t=100, b=100),
            sliders=sliders,
            updatemenus=updatemenus
        )
        
        # Set axes properties
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        
        # Make sure the aspect ratio is preserved
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))
        
        # Add legend
        legend_traces = [
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=device_color),
                name='Device Rooms',
                showlegend=True
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=propagation_colors[0]),
                name='First Propagation',
                showlegend=True
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=propagation_colors[1]),
                name='Second Propagation',
                showlegend=True
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=propagation_colors[2]),
                name='Third Propagation',
                showlegend=True
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=inactive_color),
                name='Inactive Rooms',
                showlegend=True
            )
        ]
        
        for trace in legend_traces:
            fig.add_trace(trace)
        
        # Add annotation for device rooms
        fig.add_annotation(
            text="* Rooms with devices",
            xref="paper", yref="paper",
            x=0.01, y=0.01,
            showarrow=False,
            font=dict(size=12)
        )
        
        # Save as HTML
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Interactive visualization saved to {output_file}")
        
        return fig

    #############################
    # Graph
    #############################

    def build_homogeneous_graph(self, static_room_attributes: List[str] = None) -> nx.Graph:
        """
        Build a homogeneous graph with rooms as nodes, including specified room attributes.
        
        Args:
            attributes: List of room attributes to include as node features. 
                    Format can be:
                    - Simple attribute name: 'room_number', 'is_support_zone'
                    - Nested attribute with dot notation: 'polygons_geo.area', 'polygons_doc.centroid'
                    - Special attributes: 'norm_area_minmax', 'norm_area_prop' for normalized areas
                    
                    If None, uses default attributes.
        Returns:
            A NetworkX Graph with room nodes and their attributes.
        """
        # Check if adjacency matrix is available
        if not hasattr(self, 'adjacency_matrix_df') or self.adjacency_matrix_df is None:
            raise ValueError("Room adjacency matrix not found. Run build_room_to_room_adjacency first.")
        
        if not hasattr(self, 'adj_matrix_room_uris') or self.adj_matrix_room_uris is None:
            raise ValueError("Room URIs in adjacency matrix order not found. Run build_room_to_room_adjacency first.")

        # Set default attributes if none provided
        if static_room_attributes is not None:
            self.static_room_attributes = static_room_attributes

        # Check if we need to compute normalized areas
        if ('norm_area_minmax' in self.static_room_attributes or 'norm_area_prop' in self.static_room_attributes) and (
                not hasattr(self, 'norm_areas_minmax') or not self.norm_areas_minmax):
            logger.info("Computing normalized areas for graph attributes...")
            self.normalize_room_areas()

        # Initialize an undirected graph
        G = nx.Graph()
        
        # Add room nodes with selected attributes
        for uri in self.office_graph.rooms.keys():
            room = self.office_graph.rooms[uri]
            
            # Start with devices as default attribute
            node_attrs = {'devices': list(room.devices)}
            
            # Add additional requested attributes
            for attr in self.static_room_attributes:
                # Handle special normalized area attributes
                if attr == 'norm_area_minmax':
                    if hasattr(self, 'norm_areas_minmax') and uri in self.norm_areas_minmax:
                        node_attrs['norm_area_minmax'] = self.norm_areas_minmax[uri]
                elif attr == 'norm_area_prop':
                    if hasattr(self, 'norm_areas_prop') and uri in self.norm_areas_prop:
                        node_attrs['norm_area_prop'] = self.norm_areas_prop[uri]
                # Handle nested attributes
                elif '.' in attr:
                    parts = attr.split('.')
                    if len(parts) == 2:
                        container, key = parts
                        if hasattr(room, container) and isinstance(getattr(room, container), dict):
                            container_dict = getattr(room, container)
                            if key in container_dict:
                                node_attrs[f"{container}.{key}"] = container_dict[key]
                # Handle regular attributes
                elif hasattr(room, attr):
                    node_attrs[attr] = getattr(room, attr)
            
            # Add node with all collected attributes
            G.add_node(uri, **node_attrs)
        
        # Use the DataFrame directly to maintain indices
        adj_df = self.adjacency_matrix_df
        
        # Iterate through the upper triangle of the adjacency matrix
        for i, room1_str in enumerate(adj_df.index):
            room1_uri = self.uri_str_to_obj[room1_str]
            for j, room2_str in enumerate(adj_df.columns[i+1:], i+1):
                room2_uri = self.uri_str_to_obj[room2_str]
                w = adj_df.at[room1_str, room2_str]
                if w != 0:
                    G.add_edge(room1_uri,
                            room2_uri,
                            weight=w) # if we've fed binary adjacency, everything gets 1 essentially
        
        self.homogeneous_graph = G
        
        logger.info(f"Built homogeneous graph with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    def plot_network_graph(self,
                        graph=None,
                        figsize=(12, 10),
                        layout='spring',
                        node_size_based_on='area',
                        node_size_factor=1000,
                        node_color='lightblue',
                        device_node_color='salmon',
                        edge_width=1.0,
                        show_room_ids=True):
        """
        Plot the room adjacency as a network graph.
        
        Args:
            graph (nx.Graph, optional): The networkx graph to plot. If None, uses 
                                    the graph from build_simple_homogeneous_graph()
            figsize (tuple): Figure size
            layout (str): Graph layout type ('spring', 'kamada_kawai', 'planar', 'spatial')
            node_size_based_on (str): 'area' or 'degree' to determine node sizes
            node_size_factor (float): Factor to control node sizes
            node_color (str): Color of nodes without devices
            device_node_color (str): Color of nodes with devices
            edge_width (float): Width of edges
            show_room_ids (bool): Whether to show room IDs in the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """        
        # Use provided graph or build a new one if not available
        if self.homogeneous_graph is None:
            raise ValueError("Build homogenous graph first.")
        else:
            graph = self.homogeneous_graph

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get positions for nodes based on chosen layout
        if layout == 'spring':
            pos = nx.spring_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif layout == 'planar':
            # Try planar layout, but fall back to spring if not possible
            try:
                pos = nx.planar_layout(graph)
            except nx.NetworkXException:
                logger.warning("Planar layout not possible. Falling back to spring layout.")
                pos = nx.spring_layout(graph)
        elif layout == 'spatial':
            # Use actual spatial positions from room polygons if available
            if hasattr(self, 'room_polygons') and self.room_polygons:
                pos = {}
                for node in graph.nodes():
                    if node in self.room_polygons:
                        centroid = self.room_polygons[node].centroid
                        pos[node] = (centroid.x, centroid.y)
                    else:
                        logger.warning(f"No polygon found for room {node}. Using centroid (0,0).")
                        pos[node] = (0, 0)
            else:
                logger.warning("Spatial layout requested but no room polygons available. Using spring layout.")
                pos = nx.spring_layout(graph)
        else:
            logger.warning(f"Unknown layout: {layout}. Using spring layout.")
            pos = nx.spring_layout(graph)
        
        # Get node sizes based on selected criteria
        node_sizes = []
        
        if node_size_based_on == 'area':
            # Use room areas if available
            if hasattr(self, 'areas') and self.areas:
                max_area = max(self.areas.values()) if self.areas else 1.0
                for node in graph.nodes():
                    # Get the area, defaulting to median if not found
                    if node in self.areas:
                        area = self.areas[node]
                        node_size = area * node_size_factor / max_area
                    else:
                        # Use median value if area not found
                        area = np.median(list(self.areas.values()))
                        node_size = area * node_size_factor / max_area
                    node_sizes.append(node_size)
            else:
                # If no areas available, use degree centrality instead
                logger.warning("Room areas not available. Using node degrees for sizing.")
                degrees = dict(graph.degree())
                max_degree = max(degrees.values()) if degrees else 1
                node_sizes = [degrees[node] * node_size_factor / max_degree for node in graph.nodes()]
        elif node_size_based_on == 'degree':
            # Size based on node degree (number of connections)
            degrees = dict(graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            node_sizes = [degrees[node] * node_size_factor / max_degree for node in graph.nodes()]
        else:
            # Default size if no valid option
            logger.warning(f"Unknown node_size_based_on value: {node_size_based_on}. Using constant size.")
            node_sizes = [node_size_factor * 0.3] * len(graph.nodes())
        
        # Separate nodes with and without devices
        nodes_with_devices = []
        nodes_without_devices = []
        
        for node in graph.nodes():
            if 'devices' in graph.nodes[node] and graph.nodes[node]['devices']:
                nodes_with_devices.append(node)
            else:
                nodes_without_devices.append(node)
        
        # Draw nodes without devices
        if nodes_without_devices:
            nx.draw_networkx_nodes(graph, pos, 
                            nodelist=nodes_without_devices,
                            node_size=[node_sizes[i] for i, node in enumerate(graph.nodes()) if node in nodes_without_devices],
                            node_color=node_color, 
                            alpha=0.8, 
                            ax=ax)
        
        # Draw nodes with devices
        if nodes_with_devices:
            nx.draw_networkx_nodes(graph, pos, 
                            nodelist=nodes_with_devices,
                            node_size=[node_sizes[i] for i, node in enumerate(graph.nodes()) if node in nodes_with_devices],
                            node_color=device_node_color, 
                            alpha=0.8, 
                            ax=ax)
        
        # Draw edges with weight consideration if available
        if nx.get_edge_attributes(graph, 'weight'):
            weights = [graph[u][v]['weight'] for u, v in graph.edges()]
            # Normalize weights for visualization
            if weights:
                max_weight = max(weights)
                min_weight = min(weights)
                if max_weight > min_weight:
                    norm_weights = [(w - min_weight) / (max_weight - min_weight) * 2 + 0.5 for w in weights]
                    nx.draw_networkx_edges(graph, pos, width=norm_weights, alpha=0.6, ax=ax)
                else:
                    nx.draw_networkx_edges(graph, pos, width=edge_width, alpha=0.5, ax=ax)
            else:
                nx.draw_networkx_edges(graph, pos, width=edge_width, alpha=0.5, ax=ax)
        else:
            nx.draw_networkx_edges(graph, pos, width=edge_width, alpha=0.5, ax=ax)
        
        # Draw labels if requested - now using the pre-stored room names
        if show_room_ids:
            labels = {node: self.room_names.get(node, str(node)) for node in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
        
        # Add title based on adjacency type
        title = "Room Adjacency Graph"
        if hasattr(self, 'adjacency_type'):
            title += f" ({self.adjacency_type})"
        plt.title(title)
        
        # Add legend for node colors
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color, markersize=10, label='Rooms without devices'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=device_node_color, markersize=10, label='Rooms with devices')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add legend for node sizes
        if node_size_based_on == 'area':
            legend_text = "Node size proportional to room area"
        elif node_size_based_on == 'degree':
            legend_text = "Node size proportional to number of connections"
        else:
            legend_text = "Uniform node size"
        plt.figtext(0.5, 0.01, legend_text, ha='center')
        
        plt.axis('off')
        plt.tight_layout()
        return fig



    #############################
    # Temporal Feature Extraction
    #############################
    
    def get_room_properties(self) -> Dict[URIRef, Set[str]]:
        """
        Get property types available for each room based on its devices,
        using the NetworkX graph exclusively.
        
        Returns:
            Dictionary mapping room URIs to sets of property types
        
        Raises:
            ValueError: If the homogeneous graph has not been built yet
        """
        # Check if we have a graph to use
        if not hasattr(self, 'homogeneous_graph') or self.homogeneous_graph is None:
            raise ValueError("Homogeneous graph not available. Call build_homogeneous_graph first.")
        
        graph = self.homogeneous_graph
        room_properties = {}
        
        # Get property types from the graph nodes
        for room_uri in graph.nodes():
            property_types = set()
            
            # Get device list from graph node
            devices = graph.nodes[room_uri].get('devices', [])
            
            # For each device in the room
            for device_uri in devices:
                if device_uri in self.office_graph.devices:
                    device = self.office_graph.devices[device_uri]
                    
                    # For each property in the device
                    for property_uri in device.properties:
                        # Find its property type
                        for prop_type, uris in self.office_graph.property_type_mappings.items():
                            if property_uri in uris:
                                property_types.add(prop_type)
                                break
            
            room_properties[room_uri] = property_types
        
        return room_properties
    
    def get_room_measurements(self) -> Dict[URIRef, Dict[str, List]]:
        """
        Get all measurements for each room organized by property type,
        using the NetworkX graph exclusively.
        
        Returns:
            Dictionary mapping room URIs to dictionaries of property types to lists of measurements
        
        Raises:
            ValueError: If the homogeneous graph has not been built yet
        """
        # Check if we have a graph to use
        if not hasattr(self, 'homogeneous_graph') or self.homogeneous_graph is None:
            raise ValueError("Homogeneous graph not available. Call build_homogeneous_graph first.")
        
        graph = self.homogeneous_graph
        room_measurements = {}
        
        # For each room
        for room_uri in graph.nodes():
            room_measurements[room_uri] = {}
            
            # Get device list from graph node
            devices = graph.nodes[room_uri].get('devices', [])
            
            # For each device in the room
            for device_uri in devices:
                if device_uri in self.office_graph.devices:
                    device = self.office_graph.devices[device_uri]
                    
                    # For each property in the device
                    for property_uri, measurements in device.measurements_by_property.items():
                        # Find the property type
                        for prop_type, uris in self.office_graph.property_type_mappings.items():
                            if property_uri in uris:
                                # Skip ignored property types
                                if prop_type in self.ignored_property_types:
                                    continue
                                
                                # Only include used property types
                                if prop_type not in self.used_property_types:
                                    continue
                                
                                # Initialize if not already
                                if prop_type not in room_measurements[room_uri]:
                                    room_measurements[room_uri][prop_type] = []
                                
                                # Add measurements
                                room_measurements[room_uri][prop_type].extend(measurements)
                                break
        
        return room_measurements
    
    #############################
    # Spatio-Temporal Graph Building
    #############################
    
    def build_temporal_graph_snapshots(self) -> Dict[int, nx.Graph]:
        """
        Build a temporal graph for each time bucket, using the weighted non-symmetric
        adjacency matrix calculated by calculate_proportional_boundary_adjacency.
        Shows a progress bar during processing.
        
        Returns:
            Dictionary mapping time bucket indices to NetworkX graphs
            
        Raises:
            ValueError: If required components are missing or not initialized
        """        
        # Check if time buckets are available
        if not self.time_buckets:
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")
        
        # Ensure homogeneous graph is available
        if not hasattr(self, 'homogeneous_graph') or self.homogeneous_graph is None:
            # Check if adjacency matrix exists
            if not hasattr(self, 'adjacency_matrix_df') or self.adjacency_matrix_df is None:
                raise ValueError("Room adjacency matrix not found. Run build_room_to_room_adjacency first.")
                
            # Build the homogeneous graph
            logger.info("Building homogeneous graph with static features")
            self.homogeneous_graph = self.build_homogeneous_graph(weighted=True)
        
        base_graph = self.homogeneous_graph
        
        # Get room properties and measurements
        logger.info("Retrieving room properties and measurements")
        room_properties = self.get_room_properties()
        room_measurements = self.get_room_measurements()
        
        # Initialize temporal graphs
        temporal_graphs = {}
        
        # Create iterator with progress bar
        time_bucket_iter = tqdm(
            enumerate(self.time_buckets), 
            total=len(self.time_buckets),
            desc="Building temporal graphs",
            unit="bucket"
        )
        
        # For each time bucket
        for bucket_idx, (bucket_start, bucket_end) in time_bucket_iter:
            # Create a copy of the base graph
            time_graph = base_graph.copy()
            
            # For each room (node)
            for room_uri in time_graph.nodes():
                # Initialize temporal feature dictionary for this room
                temporal_features = {}
                
                # For each property type
                for prop_type in self.used_property_types:
                    # Initialize feature dictionary for this property
                    if prop_type == "Contact":
                        prop_features = {
                            "sum": 0.0,
                            "has_property": 0.0,
                            "measurement_count": 0
                        }
                    else:
                        prop_features = {
                            "mean": 0.0,
                            "std": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                            "has_property": 0.0,
                            "measurement_count": 0
                        }
                    
                    # Check if room has this property type
                    if room_uri in room_properties and prop_type in room_properties[room_uri]:
                        prop_features["has_property"] = 1.0
                        
                        # Get measurements for this property type in this room
                        measurement_values = []
                        if room_uri in room_measurements and prop_type in room_measurements[room_uri]:
                            for meas in room_measurements[room_uri][prop_type]:
                                # Check if measurement falls within this time bucket
                                if bucket_start <= meas.timestamp < bucket_end:
                                    measurement_values.append(meas.value)

                        if measurement_values:
                            if prop_type == "Contact":
                                prop_features["sum"] = float(sum(measurement_values))
                                prop_features["measurement_count"] = len(measurement_values)
                            else:
                                # Calculate statistics if there are measurements
                                prop_features["mean"] = float(np.mean(measurement_values))
                                prop_features["std"] = float(np.std(measurement_values)) if len(measurement_values) > 1 else 0.0
                                prop_features["min"] = float(np.min(measurement_values))
                                prop_features["max"] = float(np.max(measurement_values))
                                prop_features["measurement_count"] = len(measurement_values)
                    
                    # Add property features to temporal features
                    temporal_features[prop_type] = prop_features
                
                # Add temporal features to node (preserve existing static features)
                time_graph.nodes[room_uri]["temporal_features"] = temporal_features
                
                # Add time bucket information
                time_graph.nodes[room_uri]["time_bucket"] = bucket_idx
                time_graph.nodes[room_uri]["time_start"] = bucket_start
                time_graph.nodes[room_uri]["time_end"] = bucket_end
            
            # Add to temporal graphs
            temporal_graphs[bucket_idx] = time_graph
        
        self.temporal_graphs = temporal_graphs
        logger.info(f"Built {len(temporal_graphs)} temporal graph snapshots")
        return temporal_graphs

    
    #############################
    # Feature Matrix Generation
    #############################
    
    def generate_feature_matrices(self) -> Tuple[Dict[int, np.ndarray], List[URIRef]]:
        """
        Generate feature matrices for each time bucket from temporal graphs,
        including both static and temporal features.
        Shows a progress bar during processing.
        
        Returns:
            Tuple of (feature_matrices, room_uris)
                - feature_matrices: Dictionary mapping time bucket indices to feature matrices
                - room_uris: List of room URIs in the same order as used in the matrices
                
        Raises:
            ValueError: If temporal graphs have not been created yet
        """        
        # Check if temporal graphs are available
        if not hasattr(self, 'temporal_graphs') or not self.temporal_graphs:
            raise ValueError("Temporal graphs not available. Call build_temporal_graph_snapshots first.")
        
        # Get first temporal graph to extract node info
        first_graph = next(iter(self.temporal_graphs.values()))
        
        # Get ordered list of room URIs (to ensure consistency across all time buckets)
        room_uris = list(first_graph.nodes())
                
        # Define feature types based on property types
        temporal_feature_types = []
        for prop_type in self.used_property_types:
            if prop_type == "Contact":
                # Contact only has sum and has_property features
                temporal_feature_types.extend([f"{prop_type}.sum", f"{prop_type}.has_property"])
            else:
                # Other properties have mean, std, min, max, has_property
                temporal_feature_types.extend([
                    f"{prop_type}.mean", f"{prop_type}.std", f"{prop_type}.min", 
                    f"{prop_type}.max", f"{prop_type}.has_property"
                ])
        
        # Calculate dimensions for feature matrices
        n_rooms = len(room_uris)
        n_static_features = len(self.static_room_attributes)
        n_temporal_features = len(temporal_feature_types)
        
        logger.info(f"Extracting features for {n_rooms} rooms: {n_static_features} static and {n_temporal_features} temporal features")
        
        # Extract static features once (they don't change over time)
        static_features = np.zeros((n_rooms, n_static_features), dtype=float)
        
        # For each room
        for room_idx, room_uri in enumerate(room_uris):
            # Extract static features
            for feat_idx, feat_name in enumerate(self.static_room_attributes):
                # Handle nested attributes with dot notation
                if '.' in feat_name:
                    container, key = feat_name.split('.')
                    if container in first_graph.nodes[room_uri] and key in first_graph.nodes[room_uri][container]:
                        static_features[room_idx, feat_idx] = first_graph.nodes[room_uri][container][key]
                # Handle regular attributes
                elif feat_name in first_graph.nodes[room_uri]:
                    static_features[room_idx, feat_idx] = first_graph.nodes[room_uri][feat_name]
        
        # Initialize feature matrices dictionary
        feature_matrices = {}
        
        # Create iterator with progress bar
        time_indices = list(self.temporal_graphs.keys())
        time_iter = tqdm(
            time_indices, 
            desc="Generating feature matrices",
            unit="bucket"
        )
        
        # For each time bucket
        for time_idx in time_iter:
            graph = self.temporal_graphs[time_idx]
            
            # Initialize temporal feature matrix for this time bucket
            temporal_features = np.zeros((n_rooms, n_temporal_features), dtype=float)
            
            # For each room
            for room_idx, room_uri in enumerate(room_uris):
                # Get room temporal features
                if "temporal_features" in graph.nodes[room_uri]:
                    room_temp_features = graph.nodes[room_uri]["temporal_features"]
                    
                    # Extract temporal features - track feature index separately
                    feat_idx = 0
                    for prop_type in self.used_property_types:
                        if prop_type in room_temp_features:
                            prop_feats = room_temp_features[prop_type]
                            
                            # Extract features based on property type
                            if prop_type == "Contact":
                                # For Contact, we store sum and has_property
                                temporal_features[room_idx, feat_idx] = prop_feats["sum"]
                                temporal_features[room_idx, feat_idx + 1] = prop_feats["has_property"]
                                # Increment feature index by 2
                                feat_idx += 2
                            else:
                                # For other properties, store mean, std, min, max, has_property
                                temporal_features[room_idx, feat_idx] = prop_feats["mean"]
                                temporal_features[room_idx, feat_idx + 1] = prop_feats["std"]
                                temporal_features[room_idx, feat_idx + 2] = prop_feats["min"]
                                temporal_features[room_idx, feat_idx + 3] = prop_feats["max"]
                                temporal_features[room_idx, feat_idx + 4] = prop_feats["has_property"]
                                # Increment feature index by 5
                                feat_idx += 5
                        else:
                            # If property not available for this room, skip the appropriate number of features
                            if prop_type == "Contact":
                                feat_idx += 2
                            else:
                                feat_idx += 5
            
            # Combine static and temporal features
            combined_features = np.hstack([static_features, temporal_features])
            
            # Store the combined feature matrix for this time bucket
            feature_matrices[time_idx] = combined_features
        
        # After filling `feature_matrices[t] = combined_features` for each t:
        time_indices = sorted(self.temporal_graphs.keys())
        # Stack into array of shape (T, R, F)
        all_feats = np.stack([feature_matrices[t] for t in time_indices], axis=0)

        # Save dimensions & names
        self.feature_matrices = {
            t: all_feats[i] for i, t in enumerate(time_indices)
        }
        self.room_uris = room_uris
        self.feature_names = self.static_room_attributes + temporal_feature_types
        self.static_feature_count  = n_static_features
        self.temporal_feature_count = n_temporal_features

        # Now standardize continuous features (in-place)
        self.standardize_continuous_features()

        logger.info(f"Generated and standardized feature matrices for {len(time_indices)} time buckets")
    
    #############################
    # Normalization
    #############################
    
    def standardize_continuous_features(self) -> Dict[str, Dict[str, float]]:
        """
        Z-score normalization of only the continuous features, computed on the train set.
        Binary flags (static or '.has_property') are left untouched.
        """
        if not hasattr(self, 'feature_matrices') or not self.feature_matrices:
            raise ValueError("Feature matrices not available. Run generate_feature_matrices first.")
        if not hasattr(self, 'train_indices') or not self.train_indices:
            raise ValueError("Train indices not available. Run split_time_buckets first.")
        if not hasattr(self, 'feature_names') or not self.feature_names:
            raise ValueError("Feature names not available. Run generate_feature_matrices first.")

        feature_names = self.feature_names
        F = len(feature_names)
        time_indices = sorted(self.feature_matrices.keys())
        T, R = len(time_indices), next(iter(self.feature_matrices.values())).shape[0]

        # --- Build binary-mask explicitly ---
        static_binary = {
            'hasWindows','has_multiple_windows','hasBackWindows','hasFrontWindows',
            'hasRightWindows','hasLeftWindows','isProperRoom'
        }
        continuous_mask = np.ones(F, dtype=bool)
        for i, fn in enumerate(feature_names):
            if fn in static_binary or fn.endswith('.has_property'):
                continuous_mask[i] = False
        continuous_indices = np.where(continuous_mask)[0]
        logger.info(f"Standardizing {len(continuous_indices)}/{F} continuous features")

        # --- Stack train data and compute mean/std ---
        # shape (T, R, F) → select train buckets → reshape to (T_train*R, F)
        all_feats = np.stack([self.feature_matrices[t] for t in time_indices], axis=0)
        train_feats = all_feats[self.train_indices]             # (T_train, R, F)
        train_flat  = train_feats.reshape(-1, F)               # (T_train*R, F)

        means = train_flat[:, continuous_indices].mean(axis=0)
        stds  = train_flat[:, continuous_indices].std(axis=0)
        # guard zero‐variance
        stds[stds < 1e-6] = 1.0

        # Save for inference
        self.continuous_indices = continuous_indices
        self.feature_mean  = means
        self.feature_std   = stds

        # --- Apply normalization to every bucket ---
        # vectorized over rooms:
        all_feats[:, :, continuous_indices] = (
            (all_feats[:, :, continuous_indices] - means[None, None, :])
            / stds[None, None, :]
        )

        # Check:
        for i in continuous_indices:
            logger.info(
                f"After norm – feature '{feature_names[i]}' has "
                f"µ≈{all_feats[:,:,i].mean():.2f}, σ≈{all_feats[:,:,i].std():.2f}; "
                f"min/max = {all_feats[:,:,i].min():.2f}/{all_feats[:,:,i].max():.2f}"
            )

        # write back into dict
        for idx, t in enumerate(time_indices):
            self.feature_matrices[t] = all_feats[idx]

        # Build detailed params
        detailed = {}
        for i, fn in enumerate(feature_names):
            if i in continuous_indices:
                detailed[fn] = {'mean': float(means[np.where(continuous_indices==i)[0][0]]),
                                'std':  float(stds [np.where(continuous_indices==i)[0][0]]),
                                'is_continuous': True}
            else:
                detailed[fn] = {'mean': 0.0, 'std': 1.0, 'is_continuous': False}

        self.detailed_normalization_params = detailed
        logger.info("Finished standardizing continuous features.")
        return detailed


    #############################
    # Task-Specific Data Preparation
    #############################
    
    def get_classification_labels(self, country_code: str = 'NL') -> np.ndarray:
        """
        Generate work hour classification labels for each time bucket.
        
        Args:
            country_code: Country code for holidays
            
        Returns:
            Binary labels for each time bucket (1 for work hour, 0 for non-work hour)
        """
        # Check if time buckets are available
        if not self.time_buckets:
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")
        
        # Initialize work hour classifier
        from ..data.TimeSeries.workhours import WorkHourClassifier
        work_hour_classifier = WorkHourClassifier(
            country_code=country_code,
            start_year=self.start_time.year,
            end_year=self.end_time.year
        )
        
        # Classify time buckets
        labels_list = work_hour_classifier.classify_time_buckets(self.time_buckets)
        
        # Convert to numpy array
        labels = np.array(labels_list, dtype=int)
        
        logger.info(f"Generated {len(labels)} classification labels")
        logger.info(f"Work hours: {labels.sum()} ({labels.sum()/len(labels):.1%} of time buckets)")
        
        return labels
    
    def get_forecasting_values(self, normalize: bool = True) -> Dict[int, float]:
        """
        Load, aggregate (to time buckets), and optionally Z-score normalize
        consumption data for forecasting.

        Args:
            normalize: if True, return (cons - mean_train)/std_train,
                       else return raw consumption.

        Returns:
            Dictionary mapping time bucket index to consumption value
            (normalized if normalize=True).
        """
        # 1) load and aggregate exactly as before
        from ..data.TimeSeries.consumption import (
            load_consumption_files,
            aggregate_consumption_to_time_buckets
        )
        consumption_data = load_consumption_files(
            self.consumption_dir,
            self.start_time,
            self.end_time
        )
        bucket_consumption = aggregate_consumption_to_time_buckets(
            consumption_data,
            self.time_buckets,
            self.interval
        )
        # now bucket_consumption: { idx: raw_value }

        # 2) If this is the first time through, compute & store raw series as an array
        if not hasattr(self, '_raw_consumption_array'):
            T = len(self.time_buckets)
            arr = np.zeros(T, dtype=float)
            for i in range(T):
                arr[i] = bucket_consumption[i]
            self._raw_consumption_array = arr

        # 3) If asked for raw, just return the dict
        if not normalize:
            logger.info(f"Returning raw consumption for {len(bucket_consumption)} buckets")
            return bucket_consumption

        # 4) Otherwise—fit Z-score on train split once, then reuse:
        if not hasattr(self, 'target_mean'):
            train_arr = self._raw_consumption_array[self.train_indices]
            mu    = train_arr.mean()
            sigma = train_arr.std() if train_arr.std()>1e-6 else 1.0
            self.target_mean = float(mu)
            self.target_std  = float(sigma)
            logger.info(f"Fitted target norm: mean={mu:.3f}, std={sigma:.3f}")

        # 5) Build normalized dict and return
        normed = {}
        for idx, raw in bucket_consumption.items():
            normed[idx] = float((raw - self.target_mean) / self.target_std)
        logger.info(f"Returning normalized consumption for {len(normed)} buckets")
        return normed

    #############################
    # STGCN Input Preparation
    #############################
    
    def prepare_stgcn_input(self) -> Dict[str, Any]:
        """
        Prepare all necessary inputs for a STGCN model, including weighted non-symmetric
        adjacency matrix and combined static/temporal features.
                    
        Returns:
            Dictionary containing all STGCN inputs
            
        Raises:
            ValueError: If required components are missing or not initialized
        """                # Make sure everything is there
        if not hasattr(self, 'temporal_graphs') or not self.temporal_graphs:
            raise ValueError("Temporal graphs not available. Run build_temporal_graph_snapshots first.")
        
        if not hasattr(self, 'feature_matrices') or not self.feature_matrices:
            raise ValueError("Feature matrices not available. Run generate_feature_matrices first.")

        # Get time indices
        time_indices = list(range(len(self.time_buckets)))
        
        # Package everything into a dictionary
        stgcn_input = {
            "adjacency_matrix": self.room_to_room_adj_matrix,
            "dynamic_adjacencies": self.masked_adjacencies,
            "room_uris": self.room_uris,
            "property_types": self.used_property_types,
            "feature_matrices": self.feature_matrices,
            "feature_names": self.feature_names,
            "static_feature_count": self.static_feature_count,
            "temporal_feature_count": self.temporal_feature_count,
            "time_indices": time_indices,
            "time_buckets": self.time_buckets,
            "train_idx": self.train_indices,
            "val_idx":   self.val_indices,
            "test_idx":  self.test_indices,
            "workhour_labels": self.get_classification_labels(),
            "consumption_values": self.get_forecasting_values()
        }
            
        logger.info("STGCN input preparation complete")
        return stgcn_input
    
    def prepare_tabular_input(self) -> Dict[str, Any]:
        """
        Prepare aggregated (per time-bucket) feature matrix for only the rooms
        that have devices (i.e. measurements).  Each row is one time-bucket; 
        columns are [room1_feat1, … room1_featF, room2_feat1, …].
        
        Returns dict with:
          - "X" : np.ndarray, shape (T, R_dev * F)
          - "y" : np.ndarray, shape (T,)
          - "train_idx","val_idx","test_idx" : List[int] into 0…T-1
          - metadata: feature_names, device_room_uris
        """
        # Make sure everything is there
        if not hasattr(self, 'homogeneous_graph') or self.homogeneous_graph is None:
            raise ValueError("Homogeneous graph not available. Run build_homogeneous_graph first.")

        if not hasattr(self, 'temporal_graphs') or not self.temporal_graphs:
            raise ValueError("Temporal graphs not available. Run build_temporal_graph_snapshots first.")
        
        if not hasattr(self, 'feature_matrices') or not self.feature_matrices:
            raise ValueError("Feature matrices not available. Run generate_feature_matrices first.")

        G = self.homogeneous_graph
        device_room_uris: List[Any] = [
            uri for uri, data in G.nodes(data=True) 
            if data.get('devices')
        ]
        if not device_room_uris:
            raise ValueError("No rooms with devices found in the graph.")

        # Map those URIs back to indices in the all_room_uris list
        device_room_indices = [
            self.room_uris.index(uri) for uri in device_room_uris
        ]

        # 3) Build X: for each time‐bucket, grab only those room‐rows and flatten
        time_indices = list(range(len(self.time_buckets)))
        X_list = []
        for t in time_indices:
            mat = self.feature_matrices[t]                # shape (R_all, F)
            sub = mat[device_room_indices, :]        # shape (R_dev, F)
            row = sub.flatten(order='C')             # shape (R_dev * F,)
            X_list.append(row)
        X = np.vstack(X_list)                        # shape (T, R_dev * F)


        bucket_values_workhour = self.get_classification_labels()
        bucket_values_consumption = self.get_forecasting_values()

        time_indices = list(range(len(self.time_buckets)))

        y_workhour = np.array([bucket_values_workhour[i] for i in time_indices])
        y_consumption = np.array([bucket_values_consumption[i] for i in time_indices])

        # 5) (Optional) Build nice column names
        col_names: List[str] = []
        for uri in device_room_uris:
            room_id = (self.office_graph.rooms[uri].room_number
                       or str(uri).split('/')[-1])
            for feat in self.feature_names:
                # sanitize feat if it contains dots
                feat_clean = feat.replace('.', '_')
                col_names.append(f"{room_id}_{feat_clean}")

        return {
            "X": X,                                   # (T, R_dev*F)
            "y": {
                "workhour": y_workhour,                 # (T,)
                "consumption": y_consumption,           # (T,)
            },
            "feature_names": col_names,               # length R_dev*F
            "device_room_uris": device_room_uris,     # length R_dev
            "train_idx": self.train_indices,          # into 0…T-1
            "val_idx":   self.val_indices,
            "test_idx":  self.test_indices
        }

    def convert_to_torch_tensors(self, stgcn_input: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
        """
        Convert numpy arrays to PyTorch tensors for model input.
        
        Args:
            stgcn_input: Dictionary output from prepare_stgcn_input
            device: PyTorch device to move tensors to
            
        Returns:
            Dictionary with numpy arrays converted to PyTorch tensors
        """
        torch_input = {}
        
        # Adjacency matrices
        torch_input["adjacency_matrix"] = torch.tensor(stgcn_input["adjacency_matrix"], 
                                                     dtype=torch.float32, 
                                                     device=device)
        
        torch_input["dynamic_adjacencies"] = {}
        for step, masked_adj in stgcn_input["dynamic_adjacencies"].items():
            torch_input["dynamic_adjacencies"][step] = torch.tensor(masked_adj, 
                                                                  dtype=torch.float32,
                                                                  device=device)

        # Convert feature matrices
        torch_input["feature_matrices"] = {}
        for time_idx, feature_matrix in stgcn_input["feature_matrices"].items():
            torch_input["feature_matrices"][time_idx] = torch.tensor(feature_matrix, 
                                                                   dtype=torch.float32, 
                                                                   device=device)
        
        # Classification task
        torch_input["workhour_labels"] = torch.tensor(stgcn_input["workhour_labels"], 
                                                    dtype=torch.long,
                                                    device=device)
        # Forecasting task
        consumption = np.zeros(len(stgcn_input["time_indices"]))
        for idx, value in stgcn_input["consumption_values"].items():
            consumption[idx] = value
        
        torch_input["consumption_values"] = torch.tensor(consumption,
                                            dtype=torch.float32,
                                            device=device)

        # Copy non-tensor data
        torch_input["room_uris"] = stgcn_input["room_uris"]
        torch_input["property_types"] = stgcn_input["property_types"]
        torch_input["feature_names"] = stgcn_input["feature_names"]
        torch_input["static_feature_count"] = stgcn_input["static_feature_count"]
        torch_input["temporal_feature_count"] = stgcn_input["temporal_feature_count"]
        torch_input["time_indices"] = stgcn_input["time_indices"]
        torch_input["time_buckets"] = stgcn_input["time_buckets"]
        torch_input["train_idx"] = stgcn_input["train_idx"]
        torch_input["val_idx"] = stgcn_input["val_idx"]
        torch_input["test_idx"] = stgcn_input["test_idx"]

        logger.info("Converted data to PyTorch tensors on device: " + str(device))
        return torch_input

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Prepare OfficeGraph data for analysis and visualization')
    
    # Input/output arguments
    parser.add_argument('--officegraph_path', type=str, 
                        default='data/processed/officegraph_extracted.pkl',
                        help='Path to the pickled OfficeGraph')
    
    parser.add_argument('--output_dir', type=str, 
                        default='data/processed',
                        help='Directory to save the PyTorch tensors')
    
    # Static Room attributes
    parser.add_argument('--static_attr_preset', type=str,
                    choices=['minimal', 'standard', 'all'],
                    default='standard',
                    help='Preset for static room attributes: minimal, standard, or all')

    # Time-related arguments
    parser.add_argument('--start_time', type=str, 
                        default="2022-03-07 00:00:00",
                        help='Start time for analysis (YYYY-MM-DD HH:MM:SS)')

    parser.add_argument('--end_time', type=str, 
                        default="2023-01-30 00:00:00",
                        help='End time for analysis (YYYY-MM-DD HH:MM:SS)')
    
    parser.add_argument('--interval', type=str, 
                        default="1h",
                        help='Frequency of time buckets as a pandas offset string e.g., ("15min", "30min", "1h", "2h")')
    
    parser.add_argument('--use_sundays', action='store_true',
                        help='Include Sundays in the analysis')
    
    parser.add_argument(
        "--split",
        type=int,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[3, 1, 1],
        help="train/val/test split in number of blocks (default: 3 1 1)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility"
    )

    # Polygon-related arguments
    parser.add_argument('--polygon_type', type=str,
                        choices=['geo', 'doc'],
                        default='doc',
                        help='Type of polygon data to use: geo or doc')
    
    parser.add_argument('--simplify_polygons', action='store_true',
                        dest='simplify_polygons',
                        help='Polygon simplification (off by default)')
    
    parser.add_argument('--simplify_epsilon', type=float,
                        default=0.1,
                        help='Epsilon value for polygon simplification (higher = more simplification)')
    
    # Adjacency-related arguments
    parser.add_argument('--adjacency_type', type=str,
                        choices=['binary', 'weighted'],
                        default='weighted',
                        help='Type of adjacency to use: binary or weighted')
    
    parser.add_argument('--distance_threshold', type=float,
                        default=5.0,
                        help='Distance threshold (in meters) for room adjacency')
    
    # Graph visualization arguments
    parser.add_argument('--plot_floor_plan', action='store_true',
                        help='Generate floor plan visualization')
    
    parser.add_argument('--plot_adjacency', action='store_true',
                        help='Generate adjacency matrix visualization')
    
    parser.add_argument('--plot_network', action='store_true',
                        help='Generate network graph visualization')
    
    parser.add_argument('--network_layout', type=str,
                        choices=['spring', 'kamada_kawai', 'planar', 'spatial'],
                        default='spring',
                        help='Layout for network graph visualization')
    
    # Task-specific arguments    
    parser.add_argument('--consumption_dir', type=str, 
                        default='data/consumption',
                        help='Directory containing consumption data (for forecasting)')
    
    parser.add_argument('--save_plots', action='store_true',
                        help='Save plot images instead of displaying them')
    
    parser.add_argument('--plots_dir', type=str,
                        default='output/builder',
                        help='Directory to save plot images')
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load OfficeGraph
    from .officegraph import OfficeGraph
    from .extraction import OfficeGraphExtractor
    logger.info(f"Loading OfficeGraph from {args.officegraph_path}")
    with open(args.officegraph_path, 'rb') as f:
        office_graph = pickle.load(f)
    
    # Initialize builder
    builder = OfficeGraphBuilder(office_graph, consumption_dir=args.consumption_dir)
    
    # Define presets
    static_attr_presets = {
        'minimal': ['isProperRoom', 'norm_area_minmax'],
        'standard': ['hasWindows', 'has_multiple_windows', 'window_direction_sin', 'window_direction_cos', 
                     'isProperRoom', 
                     'norm_area_minmax'],
        'all': ['hasWindows', 'has_multiple_windows', 
                'window_direction_sin', 'window_direction_cos', 
                'hasBackWindows', 'hasFrontWindows', 'hasRightWindows', 'hasLeftWindows', 
                'isProperRoom', 
                'norm_area_minmax', 'norm_area_prop', 
                'polygons_doc.centroid',
                'polygons_doc.width', 'polygons_doc.height',
                'polygons_doc.compactness', 'polygons_doc.rect_fit', 'polygons_doc.aspect_ratio', 'polygons_doc.perimeter']
    }
    builder.static_room_attributes = static_attr_presets[args.static_attr_preset]

    # Initialize time parameters
    builder.initialize_time_parameters(
        start_time=args.start_time,
        end_time=args.end_time,
        interval=args.interval,
        use_sundays=args.use_sundays
    )
    
    train_blocks, val_blocks, test_blocks = args.split
    builder.split_time_buckets(
        train_blocks=train_blocks,
        val_blocks=val_blocks,
        test_blocks=test_blocks,
        seed=args.seed
    )

    # Initialize room polygons
    builder.initialize_room_polygons(
        polygon_type=args.polygon_type,
        simplify_polygons=args.simplify_polygons,
        simplify_epsilon=args.simplify_epsilon
    )
    
    # Normalize room areas
    builder.normalize_room_areas()
    
    # Plot floor plan if requested
    if args.plot_floor_plan:
        logger.info("Plotting floor plan")
        fig = builder.plot_floor_plan(
            normalization='min_max',
            show_room_ids=True
        )
        if args.save_plots:
            os.makedirs(args.plots_dir, exist_ok=True)
            fig.savefig(f"{args.plots_dir}/floor_plan.png", dpi=300)
        else:
            plt.show()
    
    # Build room-to-room adjacency
    builder.build_room_to_room_adjacency(
        matrix_type=args.adjacency_type,
        distance_threshold=args.distance_threshold
    )
        
    # Plot adjacency matrix if requested
    if args.plot_adjacency:
        logger.info("Plotting adjacency matrix")
        fig = builder.plot_adjacency_matrix(
            show_room_ids=True
        )
        if args.save_plots:
            os.makedirs(args.plots_dir, exist_ok=True)
            fig.savefig(f"{args.plots_dir}/adjacency_matrix.png", dpi=300)
        else:
            plt.show()
    
    # Creating (masked) adjacency matrices
    builder.apply_masks_to_adjacency()

    # Build homogeneous graph
    logger.info("Building homogeneous graph")
    builder.build_homogeneous_graph()
    
    # Plot network graph if requested
    if args.plot_network:
        logger.info("Plotting network graph")
        fig = builder.plot_network_graph(
            layout=args.network_layout,
            node_size_based_on='area',
            show_room_ids=True
        )
        if args.save_plots:
            os.makedirs(args.plots_dir, exist_ok=True)
            fig.savefig(f"{args.plots_dir}/network_graph.png", dpi=300)
        else:
            plt.show()
    
    # Build temporal graphs (shared for both tasks)
    logger.info("Building temporal graph snapshots")
    builder.build_temporal_graph_snapshots()
    
    # Feature matrices
    builder.generate_feature_matrices()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare STGCN input
    stgcn_input = builder.prepare_stgcn_input()
    torch_tensors = builder.convert_to_torch_tensors(stgcn_input)
    output_path = os.path.join(args.output_dir, f"torch_input_{args.adjacency_type}_{args.interval}.pt")
    torch.save(torch_tensors, output_path)
    logger.info(f"Saved tensors to {output_path}")

    # — Tabular baseline inputs —
    logger.info("Preparing data for tabular baseline")
    tab = builder.prepare_tabular_input()
    tabular_path = os.path.join(args.output_dir, f"tab_input_{args.interval}.npz")
    np.savez_compressed(
        tabular_path,
        X=tab["X"],
        y_workhour=tab["y"]["workhour"],
        y_consumption=tab["y"]["consumption"],
        train_idx=tab["train_idx"],
        val_idx=tab["val_idx"],
        test_idx=tab["test_idx"],
        feature_names=np.array(tab["feature_names"], dtype=object),
        device_room_uris=np.array(tab["device_room_uris"], dtype=object)
    )
    logger.info(f"Saved tabular baseline inputs to {tabular_path}")

    logger.info("Processing complete")
