import pandas as pd
from shapely.geometry import Polygon
import numpy as np
from typing import Dict, List
from collections import defaultdict
from rdflib.term import URIRef

from ...data.FloorPlan import polygon_utils

import logging
logger = logging.getLogger(__name__)


class SpatialBuilderMixin:
    
    def initialize_canonical_ordering(self) -> None:
        """
        Establishes a canonical ordering of rooms based on their room number.

        This method populates 
        - self.room_URIs_str: a sorted list of string URIs.
        - self.room_names: a {str_uri: room_number} dict.
        """
        # Create a temporary mapping from string URI to room number
        room_URIRefs_temp = self.office_graph.rooms.keys()
        room_names_temp = {
            str(uri): self.office_graph._map_RoomURIRef_to_RoomNumber(uri) 
            for uri in room_URIRefs_temp
        }
        
        # Store the canonically sorted list of string URIs
        self.room_URIs_str = sorted(
            room_names_temp.keys(), 
            key=lambda uri_str: room_names_temp[uri_str]
        )
        # Store the name mapping
        self.room_names = room_names_temp

        # Logging
        logger.info(f"Established canonical room ordering for {len(self.room_URIs_str)} rooms")
        logger.debug(f"First 5 rooms in order: {[self.room_names[uri] for uri in self.room_URIs_str[:5]]}")
        logger.debug(f"Last 5 rooms in order: {[self.room_names[uri] for uri in self.room_URIs_str[-5:]]}")
    
    
    #############################
    # Polygons
    #############################

    def initialize_room_polygons(
        self, 
        polygon_type='doc', 
        simplify_polygons=False, 
        simplify_epsilon=0.0
        ) -> None:
        """
        Initialize shapely Polygon objects for each room based on the Room's polygon data.
        Also extracts and stores room areas for later use.
        
        Args:
            polygon_type (str): Which polygon data to use - 'geo' or 'doc'
            simplify_polygons (bool): Whether to simplify the polygons
            simplify_epsilon (float): Epsilon value for polygon simplification. 
                                    0.0 means no simplification.
        """
        ###### Polygons ######

        # Store the polygon type for later reference
        self.polygon_type = polygon_type  

        self.polygons = defaultdict(dict)  # [floor_number][room_uri] = polygon
        self.areas = defaultdict(dict)     # [floor_number][room_uri] = area
        
        # Store room floor mapping for multi-floor processing
        self.room_to_floor = {}
        self.floor_to_rooms = defaultdict(list)
        
        # Process rooms in canonical order
        for room_URI_str in self.room_URIs_str:
            room_uri_obj = URIRef(room_URI_str)
            room = self.office_graph.rooms[room_uri_obj]
            
            # Get the appropriate polygon data based on type
            if polygon_type == 'geo':
                points_2d = room.polygons_geo.get('points_2d', [])
                area = room.polygons_geo.get('area')
            elif polygon_type == 'doc':
                points_2d = room.polygons_doc.get('points_2d', [])
                area = room.polygons_doc.get('area')
            else:
                logger.warning(f"Invalid polygon_type: {polygon_type}. Using 'doc' as default.")
                points_2d = room.polygons_doc.get('points_2d', [])
                area = room.polygons_doc.get('area')
            
            # Get the floor number using the helper in OfficeGraph
            floor_URIRef = self.office_graph._map_RoomURIRef_to_FloorURIRef(room_uri_obj)
            floor_number = self.office_graph._map_FloorURIRef_to_FloorNumber(floor_URIRef)
            
            # Quick access dictionaries
            self.room_to_floor[room_URI_str] = floor_number
            self.floor_to_rooms[floor_number].append(room_URI_str)
                        
            # Area data
            if area is not None:
                self.areas[floor_number][room_URI_str] = area
            else:
                raise ValueError(f"Room {room_URI_str} has no area data.")

            # Check if we have valid polygon data
            if not points_2d or len(points_2d) < 3:
                logger.warning(f"Room {room_URI_str} has no valid polygon data for type '{polygon_type}'.")
                continue
            
            # Create a shapely Polygon (simplify if requested)
            if simplify_polygons:
                simplified_coords = polygon_utils.simplify_polygon(points_2d, epsilon=simplify_epsilon)
                polygon = Polygon(simplified_coords)
                logger.debug(f"Simplified room {room_URI_str} polygon from {len(points_2d)} to {len(simplified_coords)} vertices.")
            else:
                polygon = Polygon(points_2d)
            
            self.polygons[floor_number][room_URI_str] = polygon

        # Sort floor_to_rooms lists to maintain canonical ordering within floors
        for floor_num in self.floor_to_rooms:
            self.floor_to_rooms[floor_num].sort(key=lambda uri_str: self.room_names[uri_str])
        
        # Calculate totals for proper logging
        total_polygons = sum(len(floor_polys) for floor_polys in self.polygons.values())
        total_areas = sum(len(floor_areas) for floor_areas in self.areas.values())
        
        logger.info(f"Initialized {total_polygons} room polygons using '{polygon_type}' data.")
        logger.info(f"Extracted {total_areas} room areas.")
        logger.info(f"Found rooms on {len(self.floor_to_rooms)} floors: {list(self.floor_to_rooms.keys())}")
        
        # Log floor-by-floor breakdown
        for floor_num in sorted(self.polygons.keys()):
            logger.info(f"  Floor {floor_num}: {len(self.polygons[floor_num])} rooms")
        
        return None
    
    ##############################
    # Normalization of Areas
    ##############################
    
    def normalize_room_areas(self) -> None:
        """
        Calculate both min-max and proportion normalized room areas and store them as class attributes. 
        Uses the areas stored during initialize_room_polygons().
        
        This method populates:
        - self.norm_areas_minmax: Hierarchical dictionary [floor_number][room_uri] = normalized_area (0-1 scale)
        - self.norm_areas_prop: Hierarchical dictionary [floor_number][room_uri] = proportion (fraction of total)
        """
        # Initialize hierarchical dictionaries for normalized areas
        self.norm_areas_minmax = defaultdict(dict)
        self.norm_areas_prop = defaultdict(dict)
        
        if not hasattr(self, 'areas') or not self.areas:
            raise ValueError("No areas available for normalization. Call initialize_room_polygons first.")
        
        # Flatten all areas across floors for global min/max/total calculations
        all_areas = []
        for floor_areas in self.areas.values():
            all_areas.extend(floor_areas.values())
        
        if not all_areas:
            raise ValueError("No area data found across all floors.")
        
        # Get total area and min/max values for calculations
        total_area = sum(all_areas)
        min_area = min(all_areas)
        max_area = max(all_areas)
        
        # Perform both normalizations for each floor
        total_normalized = 0
        for floor_number, floor_areas in self.areas.items():
            for room_uri, area in floor_areas.items():
                # Handle edge case where min and max are the same
                if max_area == min_area:
                    self.norm_areas_minmax[floor_number][room_uri] = 1.0
                else:
                    # Min-max normalization: (value - min) / (max - min)
                    self.norm_areas_minmax[floor_number][room_uri] = (area - min_area) / (max_area - min_area)
                
                # Proportion normalization
                if total_area <= 0:
                    self.norm_areas_prop[floor_number][room_uri] = 0.0
                else:
                    self.norm_areas_prop[floor_number][room_uri] = area / total_area
                
                total_normalized += 1
        
        # Convert defaultdicts to regular dicts
        self.norm_areas_minmax = dict(self.norm_areas_minmax)
        self.norm_areas_prop = dict(self.norm_areas_prop)
        
        logger.info(f"Calculated min-max and proportion normalizations for {total_normalized} room areas across {len(self.areas)} floors")
        
        # Log floor-by-floor breakdown
        for floor_num in sorted(self.norm_areas_minmax.keys()):
            logger.info(f"  Floor {floor_num}: normalized {len(self.norm_areas_minmax[floor_num])} rooms")

        return None
    
    
    #############################
    # Horizontal Adjacency
    #############################
    
    def calculate_binary_adjacency(
        self,
        floor_number,
        distance_threshold: float = 5.0
        ) -> pd.DataFrame:
        """
        Calculate binary adjacency between rooms on a floor based on their polygons.
        Two rooms are considered adjacent if their polygons are within distance_threshold.
        
        Args:
            floor_number (int): Floor number to calculate adjacency for
            distance_threshold (float): Maximum distance (in meters) for rooms to be considered adjacent
        
        Returns:
            DataFrame: Adjacency matrix as a pandas DataFrame (rows i to j)
        """
        # Get room polygons for the specified floor
        if not hasattr(self, 'polygons') or floor_number not in self.polygons:
            raise ValueError(f"Floor {floor_number} not found. Available floors: {list(self.polygons.keys()) if hasattr(self, 'polygons') else []}")
        
        # Get the rooms on the requested floor
        room_polygons = self.polygons[floor_number]        
        room_URIs_str = list(room_polygons.keys()) 
        if not room_polygons:
            raise ValueError(f"No room polygons found for floor {floor_number}")

        # Initialize adjacency matrix with zeros
        adj_df = pd.DataFrame(0, index=room_URIs_str, columns=room_URIs_str)
        
        # Fill adjacency matrix
        for i, room1_id in enumerate(room_URIs_str):
            room1_poly = room_polygons.get(room1_id)
            if not room1_poly: continue
                
            for j, room2_id in enumerate(room_URIs_str[i+1:], i+1):
                room2_poly = room_polygons.get(room2_id)
                if not room2_poly: continue
                
                if room1_poly.distance(room2_poly) <= distance_threshold:
                    adj_df.at[room1_id, room2_id] = 1
                    adj_df.at[room2_id, room1_id] = 1

        return adj_df

    def calculate_proportional_boundary_adjacency(
        self,
        floor_number,
        distance_threshold: float = 5.0,
        min_shared_length: float = 0.01,
        min_weight: float = 0.0
        ) -> pd.DataFrame:
        """
        Calculate a non-symmetric adjacency matrix where
        A[i,j] = (shared boundary length between room i and j) / perimeter(room i),
        but only if the two rooms are considered adjacent in the binary adjacency matrix.
        
        Args:
            floor_number (int): Floor number to calculate adjacency for
            distance_threshold: max distance (in meters) to consider adjacency
            min_shared_length: minimum shared boundary length to consider
            min_weight: minimum weight to assign when binary adjacency exists but 
                        no significant boundary is shared (default: 0.001)
            
        Returns:
            DataFrame: Adjacency matrix as a pandas DataFrame (rows i to j)
        """
        # Get room polygons for the specified floor
        if not hasattr(self, 'polygons') or floor_number not in self.polygons:
            raise ValueError(f"Floor {floor_number} not found. Available floors: {list(self.polygons.keys()) if hasattr(self, 'polygons') else []}")
        
        # Get the room polygons on the requested floor
        room_polygons = self.polygons[floor_number]
        if not room_polygons:
            raise ValueError(f"No room polygons found for floor {floor_number}")
        
        # Get binary adjacency for this floor
        binary_adj_df = self.calculate_binary_adjacency(floor_number=floor_number, distance_threshold=distance_threshold)
            
        # Reuse the room_ids from binary adjacency for consistency
        room_URIs_str = binary_adj_df.index.tolist()
        
        # Prepare DataFrame
        adj_df = pd.DataFrame(0.0, index=room_URIs_str, columns=room_URIs_str)
        
        # Create URI mapping for this floor
        perimeters = {
            rid: room_polygons[rid].length
            for rid in room_URIs_str if rid in room_polygons and room_polygons[rid].is_valid
        }

        # Process only room pairs that are adjacent according to binary adjacency
        for r1 in room_URIs_str:
            poly1 = room_polygons.get(r1)
            p1 = perimeters.get(r1, 0)
            
            # Skip if the source room has no valid polygon or perimeter
            if not poly1 or p1 <= 0:
                continue

            for r2 in room_URIs_str:
                # Only calculate for pairs that are adjacent in the binary check
                if r1 == r2 or binary_adj_df.at[r1, r2] == 0:
                    continue

                poly2 = room_polygons.get(r2)
                if not poly2:
                    continue

                try:
                    # Find the length of the shared boundary
                    shared = poly1.boundary.intersection(
                        poly2.boundary.buffer(distance_threshold / 2)
                    )
                    length = sum(seg.length for seg in getattr(shared, 'geoms', [shared]))

                    # Assign weight: proportional to perimeter or a minimum value
                    if length > min_shared_length:
                        adj_df.at[r1, r2] = length / p1
                    else:
                        adj_df.at[r1, r2] = min_weight

                except Exception as e:
                    logger.warning(f"Boundary intersection error between {r1} and {r2}: {e}")
                    adj_df.at[r1, r2] = min_weight
                    continue

        return adj_df
    
    def build_horizontal_adjacency(self, 
                                   mode="weighted", 
                                   distance_threshold=5.0) -> None:
        """
        Build horizontal room-to-room adjacency matrices for all floors and store them in class attributes.
        
        Args:
            mode: Kind of adjacency. Options:
                - 'binary': Basic binary adjacency based on proximity
                - 'weighted': Weighted adjacency where each room's influence is proportional to target's perimeter
            distance_threshold: Maximum distance for considering rooms adjacent (in meters)
        """
        if not hasattr(self, 'polygons') or not self.polygons:
            raise ValueError("No room polygons found. Make sure to call initialize_room_polygons().")
        
        # Initialize the horizontal adjacency storage
        self.horizontal_adj = {}
        self.horizontal_adj_type = mode

        # Select the appropriate adjacency function
        if mode == "binary":
            adj_func = self.calculate_binary_adjacency
        elif mode == "weighted":
            adj_func = self.calculate_proportional_boundary_adjacency
        else:
            raise ValueError(f"Unknown adjacency kind: {mode}. Use 'binary' or 'weighted'.")
        
        logger.info(f"Building {mode} horizontal adjacency for {len(self.polygons)} floors...")

        for floor_number in sorted(self.polygons.keys()):
            adj_df = adj_func(floor_number=floor_number, distance_threshold=distance_threshold)
            
            # CHANGE: Simplified. adj_df.index is already a list of strings. No obj mapping.
            self.horizontal_adj[floor_number] = {
                "df": adj_df,
                "matrix": adj_df.values,
                "room_URIs_str": adj_df.index.tolist(),
            }
            logger.info(f"  Floor {floor_number}: {adj_df.shape} matrix with {(adj_df > 0).sum().sum()} connections")
        
        self._combine_horizontal_adjacencies()
        return None
    
    def _combine_horizontal_adjacencies(self) -> None:
        """
        Combine per-floor horizontal adjacency matrices into one large matrix.
        Uses canonical room ordering established during initialization.
        """
        if not hasattr(self, 'horizontal_adj') or not self.horizontal_adj:
            raise ValueError("No horizontal adjacency matrices found. Call build_horizontal_adjacency() first.")
        
        # Get all room URIs in canonical order that appear in adjacency matrices
        rooms_in_adj_matrices = set()
        for floor_data in self.horizontal_adj.values():
            rooms_in_adj_matrices.update(floor_data["df"].index)
                
        # Keep canonical order but only for rooms that have adjacency data
        all_room_URIs_str = [uri for uri in self.room_URIs_str if uri in rooms_in_adj_matrices]
        
        logger.info(f"Creating combined matrix with {len(all_room_URIs_str)} rooms in canonical order")
        
        # Create combined matrix
        self.combined_horizontal_adj_df = pd.DataFrame(0.0, index=all_room_URIs_str, columns=all_room_URIs_str)
        
        # Fill in the floor-specific adjacencies
        for floor_data in self.horizontal_adj.values():
            floor_df = floor_data["df"]
            self.combined_horizontal_adj_df.loc[floor_df.index, floor_df.columns] = floor_df
        
        self.horizontal_adj_matrix = self.combined_horizontal_adj_df.values
        self.adj_matrix_room_URIs_str = self.combined_horizontal_adj_df.index.tolist()
        logger.info(f"Combined horizontal adjacencies into {self.combined_horizontal_adj_df.shape} matrix.")
        return None
    

    #############################
    # Vertical Adjacency
    #############################

    def calculate_binary_vertical_adjacency(
        self,
        min_overlap_area: float = 0.1
        ) -> pd.DataFrame:
        """
        Build a full-building binary vertical adjacency matrix.
        A[i,j] = 1 if room i (on floor f) and room j (on floor f+1 or f-1)
        have an overlap area greater than min_overlap_area. Otherwise 0.

        Args:
            min_overlap_area (float): The minimum area (in square meters) of polygon
                                    intersection to be considered an overlap.

        Returns:
            pd.DataFrame: A binary adjacency matrix for the entire building.
        """
        # Initialize an empty adjacency matrix with zeros
        adj_df = pd.DataFrame(0, index=self.room_URIs_str, columns=self.room_URIs_str, dtype=int)

        # Iterate over all rooms in their canonical order
        for uri1_str in self.room_URIs_str:
            # Get the floor number and polygon for the first room
            f1 = self.room_to_floor.get(uri1_str)
            poly1 = self.polygons.get(f1, {}).get(uri1_str)
            # Skip if the room has no valid polygon
            if poly1 is None: continue

            # Check for adjacency on the floor above and below
            for delta in [-1, 1]:
                f2 = f1 + delta
                # Continue if the adjacent floor doesn't exist
                if f2 not in self.polygons: continue
                # Iterate through all rooms on the adjacent floor
                for uri2_str in self.floor_to_rooms.get(f2, []):
                    poly2 = self.polygons[f2].get(uri2_str)
                    # Skip if the second room has no valid polygon
                    if not poly2: continue
                    # Add binary adjacency if above the min_overlap_area threshold
                    if poly1.intersection(poly2).area > min_overlap_area:
                        adj_df.at[uri1_str, uri2_str] = 1
        return adj_df
        
    def calculate_proportional_vertical_adjacency(
        self,
        min_overlap_area: float = 0.1,
        min_weight: float = 0.0
        ) -> pd.DataFrame:
        """
        Build a full-building vertical adjacency matrix:
        A[i,j] = overlap_area(poly_i, poly_j) / area(poly_i)
        only if rooms are on floors f and f+1 (adjacent).
        Otherwise 0.
        """
        # Initialize empty adjacency (with canonical order)
        adj_df = pd.DataFrame(0.0, index=self.room_URIs_str, columns=self.room_URIs_str)

        for uri1_str in self.room_URIs_str:
            # Get the floor number and polygon for the first room
            f1 = self.room_to_floor.get(uri1_str)
            poly1 = self.polygons.get(f1, {}).get(uri1_str)
            # Skip if the room has no valid polygon
            if poly1 is None: continue
            
            # Check for adjacency on the floor above and below
            for delta in [-1, 1]:
                f2 = f1 + delta
                # Continue if the adjacent floor doesn't exist
                if f2 not in self.polygons: continue
                # Iterate through all rooms on the adjacent floor
                for uri2_str in self.floor_to_rooms.get(f2, []):
                    poly2 = self.polygons[f2].get(uri2_str)
                    # Skip if the second room has no valid polygon
                    if not poly2: continue
                        # Add binary adjacency if above the min_overlap_area threshold

                    overlap = poly1.intersection(poly2).area
                    adj_df.at[uri1_str, uri2_str] = (overlap / poly1.area) if overlap > min_overlap_area else min_weight
        return adj_df

    def build_vertical_adjacency(
        self,
        mode: str = "weighted",
        min_overlap_area: float = 0.1,
        min_weight: float = 0.0
        ) -> None:
        """Builds the combined vertical adjacency matrix."""
        logger.info(f"Building '{mode}' vertical adjacency for entire building...")
        
        # Select the appropriate calculation function based on the mode
        if mode == "weighted":
            v_df = self.calculate_proportional_vertical_adjacency(
                min_overlap_area=min_overlap_area,
                min_weight=min_weight
            )
        elif mode == "binary":
            v_df = self.calculate_binary_vertical_adjacency(
                min_overlap_area=min_overlap_area
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'weighted' or 'binary'.")
        
        # Store the results
        self.vertical_adj_type = mode
        self.combined_vertical_adj_df = v_df
        self.vertical_adj_matrix = v_df.values
        self.adj_matrix_room_URIs_str = list(self.room_URIs_str)

        total_connections = (v_df > 0).sum().sum()
        logger.info(
            f"Built '{mode}' vertical adjacency matrix {v_df.shape} with "
            f"{int(total_connections)} non-zero connections."
        )
        return None

    #############################
    # Combined Adjacency
    #############################


    def build_combined_room_to_room_adjacency(self) -> None:
        """
        Combines the horizontal and vertical adjacency matrices.
        The result is stored in self.room_to_room_adj_matrix.
        
        Raises:
            ValueError: If horizontal or vertical adjacency matrices are not yet computed.
        """
        if not hasattr(self, 'horizontal_adj_matrix') or self.horizontal_adj_matrix is None:
            raise ValueError(
                "Horizontal adjacency matrix not found. "
                "Run build_horizontal_adjacency() and combine_horizontal_adjacencies() first."
            )
        if not hasattr(self, 'vertical_adj_matrix') or self.vertical_adj_matrix is None:
            raise ValueError(
                "Vertical adjacency matrix not found. "
                "Run build_vertical_adjacency() first."
            )

        horizontal = self.horizontal_adj_matrix
        vertical = self.vertical_adj_matrix
        
        self.room_to_room_adj_matrix = horizontal + vertical
        
        logger.info(
            f"Successfully combined horizontal and vertical adjacency matrices. "
            f"Shape: {self.room_to_room_adj_matrix.shape}"
        )
        return None



    #############################
    # Information propagation
    #############################


    def create_masked_adjacency_matrices(
        self, 
        adjacency_matrix: np.ndarray, 
        uri_str_list: List[str]
        ) -> Dict[int, np.ndarray]:
        """
        Calculate a series of masking matrices representing information propagation
        from rooms with devices to other rooms in the building, using BOTH
        horizontal and vertical adjacency.

        Step 0: only rooms with devices can pass info.
        Step k: rooms reachable in k hops (horizonal or vertical).
        Continues until no new rooms are added.

        Then, using the propagation masks, produce a series of
        masked adjacency matrices showing the network at each step.

        Args:
            adjacency_matrix (np.ndarray): The adjacency matrix to use for calculations.
            uri_str_list (List[str]): The list of URI strings corresponding to the matrix rows/columns.
        
        Returns:
            Dict[int, np.ndarray]: A dictionary of {step: masked_adjacency_matrix}.
        """
        n_rooms = adjacency_matrix.shape[0]
        room_has_device = np.zeros(n_rooms, dtype=bool)

        # Find which rooms have devices
        for i, uri_str in enumerate(uri_str_list):
            if uri_str == "outside":
                continue
            room = self.office_graph.rooms[URIRef(uri_str)]
            if room and getattr(room, "devices", None):
                room_has_device[i] = True
                
        logger.info(f"{room_has_device.sum()} of {n_rooms} rooms have devices.")
                
        # Calculating propagation masks
        masks, can_pass, step = {}, room_has_device.copy(), 0        
        while True:
            masks[step] = np.tile(can_pass, (n_rooms, 1)).T
            newly_reachable = (adjacency_matrix.T @ can_pass > 0) & ~can_pass
            if not newly_reachable.any() or step > n_rooms: break
            can_pass |= newly_reachable
            step += 1
        logger.info(f"Generated {len(masks)} propagation masks, reaching equilibrium after {step} steps.")

        # Building the masked adjacency matrices
        masked_adjs = {
            step: adjacency_matrix * mask
            for step, mask in masks.items()
        }
        logger.info(f"Created {len(masked_adjs)} masked adjacency matrices.")

        return masked_adjs
    
    
    #############################
    # Outside Adjacency
    #############################

    def calculate_outside_adjacency(self, floor_number: int, mode: str = "weighted") -> np.ndarray:
        """
        Compute the outside‐to‐room adjacency vector for one floor.

        Args:
            floor_number: which floor to process
            mode: "binary" or "weighted"
              - "binary": 1 for rooms with hasWindows, else 0
              - "weighted": for windowed rooms, weight = max(0, 1 – sum(horizontal adj));
                            non-windowed rooms get 0

        Returns:
            np.ndarray of length = rooms on that floor (in the same order as self.horizontal_adj[floor_number]['room_URIs_str'])
        """
        if not hasattr(self, 'horizontal_adj') or floor_number not in self.horizontal_adj:
            raise ValueError(f"No horizontal adjacency for floor {floor_number}. Run build_horizontal_adjacency() first.")
        
        floor_data = self.horizontal_adj[floor_number]
        df = floor_data['df']
        room_uris_str = floor_data['room_URIs_str']
        vec = np.zeros(len(room_uris_str), dtype=float)

        for i, uri_str in enumerate(room_uris_str):
            room = self.office_graph.rooms.get(URIRef(uri_str))
            if not room or not room.hasWindows: continue
            
            if mode == "binary":
                vec[i] = 1.0
            elif mode == "weighted":
                vec[i] = max(0.0, 1.0 - df.loc[uri_str].sum())
        return vec
    
    def build_outside_adjacency(self, mode: str = "weighted") -> None:
        """
        Build per‐floor outside adjacency vectors and store in self.outside_adj.

        After calling this, we have:
          self.outside_adj[floor_number] = {
             "vector": np.ndarray,
             "room_URIs_str": str,
          }
        """
        self.outside_adj = {}
        self.outside_adj_mode = mode

        for floor_number in sorted(self.polygons.keys()):
            self.outside_adj[floor_number] = {
                "vector": self.calculate_outside_adjacency(floor_number, mode=mode),
                "room_URIs_str": self.horizontal_adj[floor_number]['room_URIs_str']
            }
        logger.info(f"Built outside adjacency on {len(self.outside_adj)} floors (mode={mode})")

        # Combine all per‐floor outside adjacency vectors into one building‐wide vector
        self._combine_outside_adjacencies()
        return None

    def _combine_outside_adjacencies(self) -> None:
        """
        Combine all per‐floor outside adjacency vectors into one building‐wide vector
        in the canonical self.room_uris order.

        Stores:
          - self.combined_outside_adj: numpy array, length = total rooms
          - self.combined_outside_adj_series: pandas.Series indexed by str(uri)
          - self.room_to_outside_adjacency: same numpy array (back‐compat)
        """
        if not hasattr(self, 'outside_adj') or not self.outside_adj:
            raise ValueError("No outside adjacency found. Run build_outside_adjacency() first.")

        combined = pd.Series(0.0, index=self.room_URIs_str)
        for floor_num, floor_data in self.outside_adj.items():
            floor_series = pd.Series(floor_data["vector"], index=floor_data["room_URIs_str"])
            combined.update(floor_series)

        self.combined_outside_adj = combined.values
        self.combined_outside_adj_series = combined
        self.room_to_outside_adjacency = combined.values # backward compatibility
        logger.info(f"Combined outside adjacency into vector of length {len(combined)}.")
        return None