import logging
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from rdflib import URIRef
from typing import Dict, Any, List

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
            raise ValueError("No areas available for normalization. Call initialize_room_polygons first.")
        
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
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1)
        )
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
        fig.update_layout(
            title="Information Propagation Through Building",
            autosize=True,
            width=900,
            height=700,
            margin=dict(l=50, r=50, t=100, b=100),
            sliders=sliders,
            updatemenus=updatemenus,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        )
        
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
                
        # Save as HTML
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Interactive visualization saved to {output_file}")
        
        return fig

    #############################
    # Outside Adjacency
    #############################

    def calculate_outside_adjacency(self, mode: str = "weighted"):
        """
        Compute the outside‐to‐room adjacency vector.

        Args:
            mode: "binary" or "weighted"
                - "binary": simply 1 for every room with hasWindows == True, else 0
                - "weighted": for rooms with windows, 
                  weight = max(0, 1 – sum(weighted adjacencies to other rooms));
                  rooms without windows get 0.
            distance_threshold: passed to build weighted adjacency if needed.

        Returns:
            np.ndarray of length N_rooms, ordered the same as self.adj_matrix_room_uris
        """
        # Make sure we have everything we need
        if not hasattr(self, 'adjacency_matrix_df') or self.adjacency_matrix_df is None:
            raise ValueError("Room adjacency matrix not found. Run build_room_to_room_adjacency first.")
        if not hasattr(self, 'adj_matrix_room_uris') or self.adj_matrix_room_uris is None:
            raise ValueError("Room URIs in adjacency matrix order not found. Run build_room_to_room_adjacency first.")

        # Binary case: just hasWindows
        if mode == "binary":
            outside = np.array([
                1.0 if self.office_graph.rooms[URIRef(uri)].hasWindows else 0.0
                for uri in self.adj_matrix_room_uris
            ], dtype=float)

        # Weighted case: leftover from perimeter‐based adjacency
        elif mode == "weighted":

            # Sum each row of the weighted adjacency DataFrame
            row_sums = self.adjacency_matrix_df.sum(axis=1)

            outside_weights = []
            for uri, row_sum in zip(self.adjacency_matrix_df.index, row_sums):
                room = self.office_graph.rooms[URIRef(uri)]
                if room.hasWindows:
                    # leftover up to a max of 1.0
                    w = max(0.0, 1.0 - row_sum)
                else:
                    w = 0.0
                outside_weights.append(w)

            outside = np.array(outside_weights, dtype=float)

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'binary' or 'weighted'.")

        self.room_to_outside_adjacency = outside

    def plot_outside_adjacency(self,
                               show_room_ids: bool = True,
                               figsize: tuple = (12, 10),
                               colormap: str = "turbo"):
        """
        Plot the floor plan with rooms colored according to their outside-adjacency.
        
        Args:
            mode (str): "binary" or "weighted" — passed to calculate_outside_adjacency()
            show_room_ids (bool): Whether to annotate each room with its ID
            figsize (tuple): Figure size
            colormap (str): Matplotlib colormap name for coloring rooms
            
        Returns:
            matplotlib.figure.Figure
        """
        # Ensure polygons are initialized
        if not hasattr(self, "room_polygons") or not self.room_polygons:
            raise ValueError("No polygons available for plotting. Call initialize_room_polygons() first.")

        if not hasattr(self, "room_to_outside_adjacency") or self.room_to_outside_adjacency is None:
            raise ValueError("No outside adjacency available for plotting. Call calculate_outside_adjacency() first.")

        # Build a mapping from room_uri → outside weight
        # Assumes calculate_outside_adjacency returns in same order as adj_matrix_room_uris
        uri_list = self.adj_matrix_room_uris
        outside_map = {uri: w for uri, w in zip(uri_list, self.room_to_outside_adjacency)}

        # Set up plot
        fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap(colormap)

        # Plot each room
        for uri, poly in self.room_polygons.items():
            w = outside_map.get(uri, 0.0)
            color = cmap(w)
            x, y = poly.exterior.xy
            ax.fill(x, y, alpha=0.7, fc=color, ec="black")

            if show_room_ids:
                # derive display id
                room = self.office_graph.rooms.get(uri, None)
                if room:
                    disp = room.room_number or str(uri).split("/")[-1]
                else:
                    disp = str(uri).split("/")[-1]
                centroid = poly.centroid
                ax.text(centroid.x, centroid.y, disp,
                        ha="center", va="center", fontsize=8,
                        color="black", fontweight="bold")

        # equal aspect
        ax.set_aspect("equal")

        # axis limits + padding
        mins = np.array([poly.bounds[:2] for poly in self.room_polygons.values()]).min(axis=0)
        maxs = np.array([poly.bounds[2:] for poly in self.room_polygons.values()]).max(axis=0)
        dx, dy = maxs - mins
        pad = 0.05 * max(dx, dy)
        ax.set_xlim(mins[0] - pad, maxs[0] + pad)
        ax.set_ylim(mins[1] - pad, maxs[1] + pad)

        # title & colorbar
        plt.title(f"Floor Plan Colored by Outside Adjacency")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Outside Adjacency Weight")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.tight_layout()
        return fig
