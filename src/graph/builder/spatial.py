import logging
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
from rdflib import URIRef
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

from ...data.FloorPlan import polygon_utils

from ..officegraph import OfficeGraph

class SpatialBuilderMixin:
    office_graph: OfficeGraph
    room_uris: List[Any]
    room_names: Dict[Any, str]

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

        ###### Rooms ######

        # Get all room URIs and create room names mapping
        all_room_uris = list(self.office_graph.rooms.keys())
        room_names_temp = {}
        
        for room_uri in all_room_uris:
            room = self.office_graph.rooms[room_uri]
            room_number = room.room_number
            room_names_temp[room_uri] = room_number
        
        # Create canonical ordering - simple string sort
        self.room_uris = sorted(all_room_uris, key=lambda uri: room_names_temp[uri])
        self.room_names = {uri: room_names_temp[uri] for uri in self.room_uris}

        logger.info(f"Established canonical room ordering for {len(self.room_uris)} rooms")
        logger.debug(f"First 5 rooms in order: {[self.room_names[uri] for uri in self.room_uris[:5]]}")
        logger.debug(f"Last 5 rooms in order: {[self.room_names[uri] for uri in self.room_uris[-5:]]}")


        ###### Polygons ######

        # Store the polygon type for later reference
        self.polygon_type = polygon_type  

        self.polygons = defaultdict(dict)  # [floor_number][room_uri] = polygon
        self.areas = defaultdict(dict)     # [floor_number][room_uri] = area
        
        # Store room floor mapping for multi-floor processing
        self.room_to_floor = {}
        self.floor_to_rooms = defaultdict(list)
        
        # Process rooms in canonical order
        for room_uri in self.room_uris:
            room = self.office_graph.rooms[room_uri]

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
            
            # Floor data
            if room.floor is not None:
                # room.floor is a URIRef, need to get the actual Floor object
                floor_obj = self.office_graph.floors.get(room.floor)
                if floor_obj and floor_obj.floor_number is not None:
                    floor_number = floor_obj.floor_number
                else:
                    # Fallback: extract floor number from URI
                    try:
                        floor_str = str(room.floor)
                        if 'floor_' in floor_str:
                            floor_number = int(floor_str.split('floor_')[-1])
                        else:
                            floor_number = int(floor_str.split('/')[-1])
                    except:
                        raise ValueError(f"Room {room_uri} has no valid floor information. Floor URI: {room.floor}")
            else:
                raise ValueError(f"Room {room_uri} has no associated floor information.")
            
            # Quick access dictionaries
            self.room_to_floor[room_uri] = floor_number
            self.floor_to_rooms[floor_number].append(room_uri)
            
            # Area data
            if area is not None:
                self.areas[floor_number][room_uri] = area
            else:
                raise ValueError(f"Room {room_uri} has no area data.")

            # Check if we have valid polygon data
            if not points_2d or len(points_2d) < 3:
                logger.warning(f"Room {room_uri} has no valid polygon data for type '{polygon_type}'.")
                continue
            
            # Create a shapely Polygon (simplify if requested)
            if simplify_polygons:
                simplified_coords = polygon_utils.simplify_polygon(points_2d, epsilon=simplify_epsilon)
                polygon = Polygon(simplified_coords)
                logger.debug(f"Simplified room {room_uri} polygon from {len(points_2d)} to {len(simplified_coords)} vertices.")
            else:
                polygon = Polygon(points_2d)
            
            self.polygons[floor_number][room_uri] = polygon

        # Sort floor_to_rooms lists to maintain canonical ordering within floors
        for floor_num in self.floor_to_rooms:
            self.floor_to_rooms[floor_num].sort(key=lambda uri: self.room_names[uri])

        # Calculate totals for proper logging
        total_polygons = sum(len(floor_polys) for floor_polys in self.polygons.values())
        total_areas = sum(len(floor_areas) for floor_areas in self.areas.values())
        
        logger.info(f"Initialized {total_polygons} room polygons using '{polygon_type}' data.")
        logger.info(f"Extracted {total_areas} room areas.")
        logger.info(f"Found rooms on {len(self.floor_to_rooms)} floors: {list(self.floor_to_rooms.keys())}")
        
        # Log floor-by-floor breakdown
        for floor_num in sorted(self.polygons.keys()):
            logger.info(f"  Floor {floor_num}: {len(self.polygons[floor_num])} rooms")
    


    ##############################
    # Normalization of Areas
    ##############################


    def normalize_room_areas(self) -> None:
        """
        Calculate both min-max and proportion normalized room areas and store them 
        as class attributes. Uses the areas stored during initialize_room_polygons().
        
        This method populates:
        - self.norm_areas_minmax: Hierarchical dictionary [floor_number][room_uri] = normalized_area (0-1 scale)
        - self.norm_areas_prop: Hierarchical dictionary [floor_number][room_uri] = proportion (fraction of total)
        
        Returns:
            None
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
        Calculate binary adjacency between rooms based on their polygons.
        Two rooms are considered adjacent if their polygons are within distance_threshold.
        
        Args:
            floor_number (int): Floor number to calculate adjacency for
            distance_threshold (float): Maximum distance (in meters) for rooms to be considered adjacent
            
        Returns:
            DataFrame: Adjacency matrix as a pandas DataFrame
        """
        # Get room polygons for the specified floor
        if not hasattr(self, 'polygons') or floor_number not in self.polygons:
            raise ValueError(f"Floor {floor_number} not found. Available floors: {list(self.polygons.keys()) if hasattr(self, 'polygons') else []}")
        
        room_polygons = self.polygons[floor_number]
        
        if not room_polygons:
            raise ValueError(f"No room polygons found for floor {floor_number}")
        
        # Get list of room URIs as strings (to be used as DataFrame indices)
        room_uris_str = [str(uri) for uri in room_polygons.keys()]
        # Store mapping from string URI to original URI object for later use
        uri_str_to_obj = {str(uri): uri for uri in room_polygons.keys()}
        
        # Initialize adjacency matrix with zeros
        adj_df = pd.DataFrame(0, index=room_uris_str, columns=room_uris_str)
        
        # Fill adjacency matrix
        for i, room1_id in enumerate(room_uris_str):
            room1_uri = uri_str_to_obj[room1_id]
            room1_poly = room_polygons.get(room1_uri)
            
            if room1_poly is None:
                continue
                
            for j, room2_id in enumerate(room_uris_str[i+1:], i+1):
                room2_uri = uri_str_to_obj[room2_id]
                room2_poly = room_polygons.get(room2_uri)
                
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
            DataFrame: adjacency matrix A (rows i to j)
        """
        # Get room polygons for the specified floor
        if not hasattr(self, 'polygons') or floor_number not in self.polygons:
            raise ValueError(f"Floor {floor_number} not found. Available floors: {list(self.polygons.keys()) if hasattr(self, 'polygons') else []}")
        
        room_polygons = self.polygons[floor_number]
        
        if not room_polygons:
            raise ValueError(f"No room polygons found for floor {floor_number}")
        
        # Get binary adjacency for this floor
        binary_adj_df = self.calculate_binary_adjacency(floor_number=floor_number, distance_threshold=distance_threshold)
            
        # Reuse the room_ids from binary adjacency for consistency
        room_ids = binary_adj_df.index.tolist()
        
        # Prepare DataFrame
        adj_df = pd.DataFrame(0.0, index=room_ids, columns=room_ids)
        
        # Create URI mapping for this floor
        uri_str_to_obj = {str(uri): uri for uri in room_polygons.keys()}
        
        # Precompute perimeters
        perimeters = {}
        for rid in room_ids:
            uri = uri_str_to_obj[rid]
            if uri in room_polygons:
                perimeters[rid] = room_polygons[uri].length
            else:
                perimeters[rid] = 0.0
        
        # Process only room pairs that are adjacent according to binary adjacency
        for i, r1 in enumerate(room_ids):
            uri1 = uri_str_to_obj[r1]
            poly1 = room_polygons.get(uri1)
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
                
                uri2 = uri_str_to_obj[r2]
                poly2 = room_polygons.get(uri2)
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

    def build_horizontal_adjacency(self, 
                                    matrix_type="binary", 
                                    distance_threshold=5.0) -> None:
        """
        Build horizontal room-to-room adjacency matrices for all floors and store them in class attributes.
        
        Args:
            matrix_type: Kind of adjacency. Options:
                - 'binary': Basic binary adjacency based on proximity
                - 'weighted': Weighted adjacency where each room's influence is proportional to target's perimeter
            distance_threshold: Maximum distance for considering rooms adjacent (in meters)
            
        Returns:
            None
        """
        if not hasattr(self, 'polygons') or not self.polygons:
            raise ValueError("No room polygons found. Make sure to call initialize_room_polygons().")
        
        # Initialize the horizontal adjacency storage
        self.horizontal_adj = {}
        self.horizontal_adj_type = matrix_type

        # Select the appropriate adjacency function
        if matrix_type == "binary":
            adj_func = self.calculate_binary_adjacency
        elif matrix_type == "weighted":
            adj_func = self.calculate_proportional_boundary_adjacency
        else:
            raise ValueError(f"Unknown adjacency kind: {matrix_type}. Use 'binary' or 'weighted'.")
        
        logger.info(f"Building {matrix_type} horizontal adjacency for {len(self.polygons)} floors...")
        
        for floor_number in sorted(self.polygons.keys()):
            logger.info(f"Processing floor {floor_number}...")
            
            # Calculate adjacency for this floor
            adj_df = adj_func(floor_number=floor_number, distance_threshold=distance_threshold)
            
            # Create URI mapping for this floor
            uri_str_to_obj = {str(uri): uri for uri in self.polygons[floor_number].keys()}
            
            # Store the results for this floor
            self.horizontal_adj[floor_number] = {
                "df": adj_df,
                "matrix": adj_df.values,
                "room_uris": [uri_str_to_obj[uri_str] for uri_str in adj_df.index],
                "uri_str_to_obj": uri_str_to_obj
            }
            
            non_zero_connections = (adj_df > 0).sum().sum()
            logger.info(f"  Floor {floor_number}: {adj_df.shape} matrix with {non_zero_connections} connections")
        
        logger.info(f"Completed horizontal adjacency calculation for {len(self.horizontal_adj)} floors")
        
        return None

    def combine_horizontal_adjacencies(self) -> None:
        """
        Combine per-floor horizontal adjacency matrices into one large matrix.
        Uses canonical room ordering established during initialization.
        
        Returns:
            None (stores results in self.combined_horizontal_adj_df and related attributes)
        """
        if not hasattr(self, 'horizontal_adj') or not self.horizontal_adj:
            raise ValueError("No horizontal adjacency matrices found. Call build_horizontal_adjacency() first.")
        
        # Get all room URIs in canonical order that appear in adjacency matrices
        rooms_in_adj_matrices = set()
        for floor_data in self.horizontal_adj.values():
            rooms_in_adj_matrices.update(floor_data["df"].index)
        
        # Keep canonical order but only for rooms that have adjacency data
        all_room_uris_str = [str(uri) for uri in self.room_uris if str(uri) in rooms_in_adj_matrices]
        
        logger.info(f"Creating combined matrix with {len(all_room_uris_str)} rooms in canonical order")
        
        # Create combined matrix
        self.combined_horizontal_adj_df = pd.DataFrame(0.0, index=all_room_uris_str, columns=all_room_uris_str)
        
        # Fill in the floor-specific adjacencies
        for floor_num, floor_data in self.horizontal_adj.items():
            floor_df = floor_data["df"]
            floor_rooms = list(floor_df.index)
            
            # Copy the floor adjacency into the appropriate block of the combined matrix
            for i, room1 in enumerate(floor_rooms):
                for j, room2 in enumerate(floor_rooms):
                    self.combined_horizontal_adj_df.at[room1, room2] = floor_df.at[room1, room2]
        
        # Store combined results
        self.horizontal_adjacency_matrix_df = self.combined_horizontal_adj_df
        self.horizontal_adj_matrix = self.combined_horizontal_adj_df.values
        
        # Create URI mapping in canonical order
        self.uri_str_to_obj = {str(uri): uri for uri in self.room_uris}
        
        # Room URIs in the same order as the combined matrix (canonical order)
        self.adj_matrix_room_uris = [self.uri_str_to_obj[uri_str] for uri_str in self.combined_horizontal_adj_df.index]
        
        total_connections = (self.combined_horizontal_adj_df > 0).sum().sum()
        logger.info(f"Combined horizontal adjacencies into {self.combined_horizontal_adj_df.shape} matrix with {total_connections} total connections")
        
        # Log ordering verification
        logger.info("Room ordering verification:")
        for floor_num in sorted(self.polygons.keys()):
            floor_rooms_in_matrix = [self.room_names[uri] for uri in self.adj_matrix_room_uris 
                                   if self.room_to_floor.get(uri) == floor_num]
            if floor_rooms_in_matrix:
                logger.info(f"  Floor {floor_num}: {floor_rooms_in_matrix[:3]}...{floor_rooms_in_matrix[-3:]} ({len(floor_rooms_in_matrix)} total)")

        return None


    #############################
    # Vertical Adjacency
    #############################


    def calculate_proportional_vertical_adjacency(
        self,
        min_overlap_area: float = 0.0,
        min_weight: float = 0.0
    ) -> pd.DataFrame:
        """
        Build a full-building vertical adjacency matrix:
        A[i,j] = overlap_area(poly_i, poly_j) / area(poly_i)
        only if rooms are on floors f and f+1 (adjacent).
        Otherwise 0.
        """
        # Prepare index/columns in canonical order
        room_strs = [str(uri) for uri in self.room_uris]
        uri_str_to_obj = {str(uri): uri for uri in self.room_uris}
        # Initialize empty adjacency
        adj_df = pd.DataFrame(0.0, index=room_strs, columns=room_strs)

        for i, uri1 in enumerate(self.room_uris):
            f1 = self.room_to_floor[uri1]
            poly1 = self.polygons.get(f1, {}).get(uri1)
            if poly1 is None or poly1.area == 0:
                continue
            area1 = poly1.area

            # Only consider rooms on floor f1+1 and f1-1
            for delta in (-1, 1):
                f2 = f1 + delta
                if f2 not in self.polygons:
                    continue

                for uri2 in self.floor_to_rooms[f2]:
                    poly2 = self.polygons[f2].get(uri2)
                    if poly2 is None:
                        continue

                    # compute overlap area
                    try:
                        overlap = poly1.intersection(poly2).area
                    except Exception as e:
                        logger.warning(f"Vertical overlap error {uri1}-{uri2}: {e}")
                        overlap = 0.0

                    # assign weight
                    if overlap > min_overlap_area:
                        weight = overlap / area1
                    else:
                        weight = min_weight

                    adj_df.at[str(uri1), str(uri2)] = weight

        return adj_df

    def build_vertical_adjacency(
        self,
        min_overlap_area: float = 0.05,
        min_weight: float = 0.0
    ) -> None:
        """
        Build and store the combined vertical adjacency matrix for the entire building.
        Stores:
          - self.combined_vertical_adj_df
          - self.vertical_adj_matrix  (NumPy array)
          - self.adj_matrix_room_uris      (list of URIRefs)
        """
        logger.info("Building vertical adjacency for entire building...")
        # Calculate full-building vertical adjacency
        v_df = self.calculate_proportional_vertical_adjacency(
            min_overlap_area=min_overlap_area,
            min_weight=min_weight
        )

        # Store
        self.combined_vertical_adj_df = v_df
        self.vertical_adj_matrix = v_df.values
        # room_uris in the same canonical order
        self.adj_matrix_room_uris = list(self.room_uris)

        total_connections = (v_df > 0).sum().sum()
        logger.info(
            f"Built vertical adjacency matrix {v_df.shape} with "
            f"{int(total_connections)} non-zero connections"
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


    def calculate_information_propagation_masks(self):
        """
        Calculate a series of masking matrices representing information propagation
        from rooms with devices to other rooms in the building, using BOTH
        horizontal and vertical adjacency.

        Step 0: only rooms with devices can pass info.
        Step k: rooms reachable in k hops (horizonal or vertical).
        Continues until no new rooms are added.
        
        Returns:
            Dict[int, np.ndarray]: step → mask matrix (1 = can pass, 0 = masked)
        """
        # --- ensure both adjacencies exist ---
        if not hasattr(self, 'horizontal_adj_matrix') or self.horizontal_adj_matrix is None:
            raise ValueError("Horizontal adjacency matrix not found. Run build_horizontal_adjacency() first.")
        if not hasattr(self, 'vertical_adj_matrix') or self.vertical_adj_matrix is None:
            raise ValueError("Vertical adjacency matrix not found. Run build_vertical_adjacency() first.")
        
        # --- Use or build combined adjacency ---
        if not hasattr(self, 'room_to_room_adj_matrix') or self.room_to_room_adj_matrix is None:
            logger.info("Combined adjacency matrix not found for propagation masks. Building it now.")
            self.build_combined_room_to_room_adjacency()
        
        adjacency = self.room_to_room_adj_matrix
        
        # --- find which rooms start with devices ---
        n_rooms = len(self.adj_matrix_room_uris)
        room_has_device = np.zeros(n_rooms, dtype=bool)
        for i, uri in enumerate(self.adj_matrix_room_uris):
            room = self.office_graph.rooms.get(uri)
            if room and getattr(room, "devices", None):
                room_has_device[i] = True
        
        logger.info(f"{room_has_device.sum()} rooms have devices / {n_rooms} total")
        
        # --- propagation masks ---
        masks = {}
        # step 0: only device rooms can PASS
        can_pass = room_has_device.copy()
        mask0 = np.zeros_like(adjacency)
        mask0[can_pass, :] = 1
        masks[0] = mask0
        
        step = 1
        while True:
            # who newly gets reachability this round?
            newly = np.zeros_like(can_pass)
            for tgt in range(n_rooms):
                if not can_pass[tgt]:
                    # any neighbor j that can_pass and adjacency[j,tgt]>0?
                    if np.any(can_pass & (adjacency[:, tgt] > 0)):
                        newly[tgt] = True
            
            if not newly.any():
                logger.info(f"Reached equilibrium after {step} steps")
                break
            
            can_pass = can_pass | newly
            mask = np.zeros_like(adjacency)
            mask[can_pass, :] = 1
            masks[step] = mask
            
            logger.info(f"Step {step}: +{newly.sum()} newly active rooms")
            step += 1
            if step > n_rooms:
                logger.warning("Stopping early to avoid infinite loop")
                break
        
        logger.info(f"Generated {len(masks)} propagation masks")
        return masks

    def build_masked_adjacencies(self, masks=None) -> None:
        """
        Using the propagation masks (horizontal+vertical), produce a series of
        masked adjacency matrices showing the network at each step.
        
        Args:
            masks (dict): step→mask from calculate_information_propagation_masks.
                          If None, that method will be called.
        
        Returns:
            Dict[int, np.ndarray]: step→(adjacency * mask)
        """
        # ensure adjacency exists
        if not hasattr(self, 'horizontal_adj_matrix') or self.horizontal_adj_matrix is None:
            raise ValueError("Adjacency matrix not found. Run both build_*_adjacency() first.")
        
        if masks is None:
            masks = self.calculate_information_propagation_masks()
        
        adjacency = self.room_to_room_adj_matrix
        self.masked_adjacencies = {
            step: adjacency * mask
            for step, mask in masks.items()
        }
        
        logger.info(f"Created {len(self.masked_adjacencies)} masked adjacency matrices")
        return None



    #############################
    # Outside Adjacency
    #############################


    def calculate_outside_adjacency(self, floor_number: int, mode: str = "weighted") -> np.ndarray:
        """
        Compute the outside‐to‐room adjacency vector **for one floor**.

        Args:
            floor_number: which floor to process
            mode: "binary" or "weighted"
              - "binary": 1 for rooms with hasWindows, else 0
              - "weighted": for windowed rooms, weight = max(0, 1 – sum(horizontal adj));
                            non-windowed rooms get 0

        Returns:
            np.ndarray of length = rooms on that floor (in the same order as self.horizontal_adj[floor_number]['room_uris'])
        """
        # Ensure horizontal adjacency is built
        if not hasattr(self, 'horizontal_adj') or floor_number not in self.horizontal_adj:
            raise ValueError(f"No horizontal adjacency for floor {floor_number}. Run build_horizontal_adjacency() first.")

        floor_data = self.horizontal_adj[floor_number]
        df = floor_data['df']                  # pandas DataFrame of horizontal adj for this floor
        room_uris = floor_data['room_uris']    # List[URIRef] in the same order

        if mode == "binary":
            vec = np.array([
                1.0 if self.office_graph.rooms[uri].hasWindows else 0.0
                for uri in room_uris
            ], dtype=float)

        elif mode == "weighted":
            # sum of each row = total horizontal weight per room
            row_sums = df.sum(axis=1)
            weights = []
            for uri in room_uris:
                room = self.office_graph.rooms[uri]
                if room.hasWindows:
                    # leftover up to 1.0
                    w = max(0.0, 1.0 - row_sums[str(uri)])
                else:
                    w = 0.0
                weights.append(w)
            vec = np.array(weights, dtype=float)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'binary' or 'weighted'.")

        return vec


    def build_outside_adjacency(self, mode: str = "weighted") -> None:
        """
        Build per‐floor outside adjacency vectors and store in self.outside_adj.

        After calling this, you’ll have:
          self.outside_adj[floor_number] = {
             "vector": np.ndarray,
             "room_uris": [...URIRefs in floor order...]
          }
        """
        self.outside_adj = {}
        self.outside_adj_mode = mode

        for floor_number in sorted(self.polygons.keys()):
            vec = self.calculate_outside_adjacency(floor_number, mode=mode)
            room_uris = self.horizontal_adj[floor_number]['room_uris']
            self.outside_adj[floor_number] = {
                "vector": vec,
                "room_uris": room_uris
            }
            logger.info(f"  Floor {floor_number}: outside vector ({mode}) length={len(vec)}")

        logger.info(f"Built outside adjacency on {len(self.outside_adj)} floors (mode={mode})")
        
        return None

    def combine_outside_adjacencies(self) -> None:
        """
        Combine all per‐floor outside adjacency vectors into one building‐wide vector
        in your canonical self.room_uris order.

        Stores:
          - self.combined_outside_adj: numpy array, length = total rooms
          - self.combined_outside_adj_series: pandas.Series indexed by str(uri)
          - self.room_to_outside_adjacency: same numpy array (back‐compat)
        """

        if not hasattr(self, 'outside_adj') or not self.outside_adj:
            raise ValueError("No outside adjacency found. Run build_outside_adjacency() first.")

        combined = []
        index = []
        for uri in self.room_uris:
            floor = self.room_to_floor[uri]
            floor_data = self.outside_adj.get(floor)
            if floor_data is None:
                combined.append(0.0)
            else:
                try:
                    idx = floor_data["room_uris"].index(uri)
                    combined.append(float(floor_data["vector"][idx]))
                except ValueError:
                    combined.append(0.0)
            index.append(str(uri))

        arr = np.array(combined, dtype=float)
        series = pd.Series(arr, index=index)

        # store
        self.combined_outside_adj = arr
        self.combined_outside_adj_series = series
        self.room_to_outside_adjacency = arr  # for backward compatibility

        logger.info(
            f"Combined outside adjacency into length-{len(arr)} vector; "
            f"total weight={series.sum():.3f}"
        )
        return None