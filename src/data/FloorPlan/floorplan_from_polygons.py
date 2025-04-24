import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt
import networkx as nx
import os
import importlib.util


class FloorPlanFromPolygons:
    """
    A class for processing room polygons from floor plans and generating 
    features useful for spatial-temporal graph convolutional network modeling.
    """
    
    def __init__(self, polygons_file_path=None, all_floors=None):
        """
        Initialize the processor with either a file path or a dictionary of room polygons.
        
        Args:
            polygons_file_path (str, optional): Path to the Python file containing room polygons.
            all_floors (dict, optional): Dictionary containing floor dictionaries with room polygons.
        """
        self.all_floors = {}
        self.floor_names = []
        self.polygons = {}  # Shapely polygons for each room
        self.centroids = {}  # Centroids for each room
        self.areas = {}  # Areas for each room
        self.normalized_areas = {}  # Min-max normalized areas
        self.proportional_areas = {}  # Proportional areas
        self.adjacency = {}  # Binary adjacency for each room
        self.weighted_adjacency_distance = None  # Weighted adjacency matrix based on distance
        self.weighted_adjacency_boundary = None  # Weighted adjacency matrix based on boundary
        
        # Load room polygons
        if polygons_file_path:
            self.load_from_file(polygons_file_path)
        elif all_floors:
            self.all_floors = all_floors
            self.floor_names = list(all_floors.keys())
            self._initialize_data_structures()
    
    def load_from_file(self, file_path):
        """
        Load room polygons from a Python file.
        
        Args:
            file_path (str): Path to the Python file containing room polygons
        """
        # Load the module dynamically
        try:
            # Get the module name from the file path
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Load the module spec
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if the module has an all_floors attribute
            if hasattr(module, 'all_floors'):
                self.all_floors = module.all_floors
                self.floor_names = list(self.all_floors.keys())
            else:
                # Look for individual floor dictionaries in the module
                for attr_name in dir(module):
                    if attr_name.startswith('floor_') and isinstance(getattr(module, attr_name), dict):
                        floor_name = attr_name
                        self.floor_names.append(floor_name)
                        self.all_floors[floor_name] = getattr(module, floor_name)
            
            # If no floors were found, check if there's a room_polygons variable
            if not self.all_floors and hasattr(module, 'room_polygons'):
                # For backward compatibility with the old format
                self.all_floors = {'floor_1': module.room_polygons}
                self.floor_names = ['floor_1']
            
            # Initialize data structures
            self._initialize_data_structures()
            
            print(f"Loaded {len(self.floor_names)} floors with a total of {len(self.polygons)} rooms")
            
        except Exception as e:
            print(f"Error loading room polygons from {file_path}: {e}")
            raise
    
    def _initialize_data_structures(self):
        """Initialize Shapely polygons and other data structures."""
        # Process each floor's polygons
        for floor_name in self.floor_names:
            floor_polygons = self.all_floors[floor_name]
            
            # Create Shapely polygons and compute areas
            for room_name, coords in floor_polygons.items():
                # Create a unique identifier for the room that includes the floor
                room_id = f"{floor_name}_{room_name}" if not room_name.startswith(floor_name) else room_name
                
                try:
                    # Create polygon - Shapely will ensure it's valid
                    polygon = Polygon(coords)
                    
                    # Fix invalid polygons if needed
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)  # This often fixes self-intersections
                    
                    self.polygons[room_id] = polygon
                    self.centroids[room_id] = polygon.centroid
                    self.areas[room_id] = polygon.area
                except Exception as e:
                    print(f"Error processing room {room_id}: {e}")
    
    def get_floor_rooms(self, floor_name):
        """
        Get all rooms for a specific floor.
        
        Args:
            floor_name (str): The name of the floor
            
        Returns:
            dict: Dictionary with room names as keys and Shapely polygons as values
        """
        if floor_name not in self.floor_names:
            raise ValueError(f"Floor {floor_name} not found. Available floors: {self.floor_names}")
        
        # Filter polygons by floor name
        return {room_id: polygon for room_id, polygon in self.polygons.items() 
                if room_id.startswith(floor_name)}
    
    def calculate_room_areas(self):
        """
        Calculate the area of each room polygon.
        
        Returns:
            dict: Dictionary with room names as keys and areas as values
        """
        # Areas are already calculated during initialization
        return self.areas
    
    def min_max_normalize_areas(self):
        """
        Scale areas to range [0,1] using min-max normalization.
        
        Returns:
            dict: Dictionary with room names as keys and normalized areas as values
        """
        areas_values = list(self.areas.values())
        min_area = min(areas_values)
        max_area = max(areas_values)
        
        # Avoid division by zero
        if max_area == min_area:
            self.normalized_areas = {room: 0.5 for room in self.areas}
        else:
            self.normalized_areas = {
                room: (area - min_area) / (max_area - min_area) 
                for room, area in self.areas.items()
            }
        
        return self.normalized_areas
    
    def proportion_normalize_areas(self):
        """
        Express each area as a proportion of the total area.
        
        Returns:
            dict: Dictionary with room names as keys and proportional areas as values
        """
        total_area = sum(self.areas.values())
        
        # Avoid division by zero
        if total_area == 0:
            self.proportional_areas = {room: 0 for room in self.areas}
        else:
            self.proportional_areas = {
                room: area / total_area 
                for room, area in self.areas.items()
            }
        
        return self.proportional_areas
    
    def calculate_floor_specific_area_normalizations(self):
        """
        Calculate area normalizations for each floor separately.
        
        Returns:
            dict: Dictionary with floor names as keys and dictionaries of normalized areas as values
        """
        floor_normalizations = {}
        
        for floor_name in self.floor_names:
            # Get rooms for this floor
            floor_rooms = self.get_floor_rooms(floor_name)
            
            # Calculate areas for these rooms
            floor_areas = {room_id: self.areas[room_id] for room_id in floor_rooms}
            
            # Calculate min-max normalization
            areas_values = list(floor_areas.values())
            min_area = min(areas_values)
            max_area = max(areas_values)
            
            # Avoid division by zero
            if max_area == min_area:
                floor_norm_areas = {room: 0.5 for room in floor_areas}
            else:
                floor_norm_areas = {
                    room: (area - min_area) / (max_area - min_area) 
                    for room, area in floor_areas.items()
                }
            
            floor_normalizations[floor_name] = floor_norm_areas
        
        return floor_normalizations
    
    def calculate_adjacency(self, distance_threshold=5.0, min_boundary_length=10.0):
        """
        Calculate room adjacency based on proximity and shared boundary length.
        
        Args:
            distance_threshold (float): Maximum distance to consider rooms adjacent
            min_boundary_length (float): Minimum length of nearby boundaries to consider rooms adjacent
            
        Returns:
            dict: Dictionary with room names as keys and lists of adjacent rooms as values
        """
        # Dictionary to store adjacency lists
        self.adjacency = {}
        
        # Check each pair of rooms
        rooms = list(self.polygons.keys())
        for i, room1 in enumerate(rooms):
            self.adjacency[room1] = []
            
            # Get boundary of room1 with a buffer
            boundary1 = self.polygons[room1].boundary
            buffered1 = boundary1.buffer(distance_threshold)
            
            for j, room2 in enumerate(rooms):
                if i != j:  # Don't compare a room with itself
                    # Get boundary of room2
                    boundary2 = self.polygons[room2].boundary
                    
                    # Check if boundaries are close and the intersection is substantial
                    if buffered1.intersects(boundary2):
                        # Calculate how much of boundary2 is near boundary1
                        intersection = buffered1.intersection(boundary2)
                        
                        # If intersection is a single point, length will be 0
                        # If it's a linestring or multilinestring, we can get its length
                        if hasattr(intersection, 'length') and intersection.length > min_boundary_length:
                            self.adjacency[room1].append(room2)
        
        return self.adjacency
    
    def calculate_weighted_adjacency_distance(self, max_distance=400.0):
        """
        Calculate weighted adjacency based on inverse distance between room centroids.
        
        Args:
            max_distance (float): Maximum distance to consider (rooms further than this get weight 0)
            
        Returns:
            pandas DataFrame: Weighted adjacency matrix
        """
        # Create empty adjacency matrix
        rooms = list(self.polygons.keys())
        n_rooms = len(rooms)
        adj_matrix = np.zeros((n_rooms, n_rooms))
        
        # Calculate weights based on inverse distance
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms):
                if i != j:  # Don't calculate for same room
                    # Calculate Euclidean distance between centroids
                    distance = self.centroids[room1].distance(self.centroids[room2])
                    
                    # Convert to weight (inverse distance)
                    if distance < max_distance:
                        # Exponential decay weight
                        weight = np.exp(-distance / (max_distance/3))
                        adj_matrix[i, j] = weight
        
        # Create DataFrame for easier visualization/manipulation
        self.weighted_adjacency_distance = pd.DataFrame(adj_matrix, index=rooms, columns=rooms)
        return self.weighted_adjacency_distance
    
    def calculate_weighted_adjacency_boundary(self, distance_threshold=5.0):
        """
        Calculate weighted adjacency based on shared boundary length.
        
        Args:
            distance_threshold (float): Maximum distance to consider boundaries as "shared"
            
        Returns:
            pandas DataFrame: Weighted adjacency matrix
        """
        # Create empty adjacency matrix
        rooms = list(self.polygons.keys())
        n_rooms = len(rooms)
        adj_matrix = np.zeros((n_rooms, n_rooms))
        
        # Calculate weights based on shared boundary length
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms):
                if i != j:  # Don't calculate for same room
                    # Create buffer around room1's boundary
                    boundary1 = self.polygons[room1].boundary
                    buffered1 = boundary1.buffer(distance_threshold)
                    
                    # Get room2's boundary
                    boundary2 = self.polygons[room2].boundary
                    
                    # If boundaries are close, calculate intersection length
                    if buffered1.intersects(boundary2):
                        intersection = buffered1.intersection(boundary2)
                        
                        # If intersection is a line or multiline, get length
                        if hasattr(intersection, 'length'):
                            # Normalize by perimeter of smaller room
                            perimeter1 = boundary1.length
                            perimeter2 = boundary2.length
                            min_perimeter = min(perimeter1, perimeter2)
                            
                            # Weight is proportion of boundary that's shared
                            weight = intersection.length / min_perimeter
                            adj_matrix[i, j] = weight
        
        # Create DataFrame for easier visualization/manipulation
        self.weighted_adjacency_boundary = pd.DataFrame(adj_matrix, index=rooms, columns=rooms)
        return self.weighted_adjacency_boundary
    
    def calculate_floor_specific_adjacency(self, distance_threshold=5.0, min_boundary_length=10.0):
        """
        Calculate adjacency matrices for each floor separately.
        
        Args:
            distance_threshold (float): Maximum distance to consider rooms adjacent
            min_boundary_length (float): Minimum length of nearby boundaries to consider rooms adjacent
            
        Returns:
            dict: Dictionary with floor names as keys and adjacency dictionaries as values
        """
        floor_adjacency = {}
        
        for floor_name in self.floor_names:
            # Get rooms for this floor
            floor_rooms = list(self.get_floor_rooms(floor_name).keys())
            
            # Calculate adjacency for this floor only
            floor_adj = {}
            
            for room1 in floor_rooms:
                floor_adj[room1] = []
                
                # Get boundary of room1 with a buffer
                boundary1 = self.polygons[room1].boundary
                buffered1 = boundary1.buffer(distance_threshold)
                
                for room2 in floor_rooms:
                    if room1 != room2:  # Don't compare a room with itself
                        # Get boundary of room2
                        boundary2 = self.polygons[room2].boundary
                        
                        # Check if boundaries are close and the intersection is substantial
                        if buffered1.intersects(boundary2):
                            intersection = buffered1.intersection(boundary2)
                            
                            if hasattr(intersection, 'length') and intersection.length > min_boundary_length:
                                floor_adj[room1].append(room2)
            
            floor_adjacency[floor_name] = floor_adj
        
        return floor_adjacency
    
    def generate_adjacency_matrix(self):
        """
        Generate a binary adjacency matrix from the adjacency dictionary.
        
        Returns:
            pandas DataFrame: Binary adjacency matrix
        """
        if not self.adjacency:
            self.calculate_adjacency()
        
        # Create empty matrix
        rooms = list(self.adjacency.keys())
        n_rooms = len(rooms)
        adj_matrix = np.zeros((n_rooms, n_rooms))
        
        # Fill matrix based on adjacency dictionary
        for i, room in enumerate(rooms):
            for adjacent_room in self.adjacency[room]:
                j = rooms.index(adjacent_room)
                adj_matrix[i, j] = 1
        
        # Create DataFrame
        return pd.DataFrame(adj_matrix, index=rooms, columns=rooms)
    
    def plot_floor_plan(self, floor_name=None, figsize=(12, 10), show_room_ids=True, 
                         show_centroids=False, highlight_adjacency=None, colormap='viridis'):
        """
        Plot the floor plan with optional features.
        
        Args:
            floor_name (str, optional): The specific floor to plot, or None for all floors
            figsize (tuple): Figure size
            show_room_ids (bool): Whether to show room IDs in the plot
            show_centroids (bool): Whether to show centroids of rooms
            highlight_adjacency (str, optional): Room ID to highlight its adjacencies
            colormap (str): Matplotlib colormap name for coloring rooms
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Determine which rooms to plot
        if floor_name:
            if floor_name not in self.floor_names:
                raise ValueError(f"Floor {floor_name} not found. Available floors: {self.floor_names}")
            rooms_to_plot = self.get_floor_rooms(floor_name)
        else:
            rooms_to_plot = self.polygons
        
        # Get the color map
        cmap = plt.get_cmap(colormap)
        
        # Calculate room areas if not already done
        if not self.normalized_areas:
            self.min_max_normalize_areas()
        
        # Plot each room
        for room_id, polygon in rooms_to_plot.items():
            # Get room color based on normalized area
            color = cmap(self.normalized_areas.get(room_id, 0.5))
            
            # If highlighting adjacency, set color differently
            if highlight_adjacency and room_id in self.adjacency.get(highlight_adjacency, []):
                color = 'red'
            elif highlight_adjacency and room_id == highlight_adjacency:
                color = 'green'
            
            # Plot the polygon
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=color, ec='black')
            
            # Show room ID if requested
            if show_room_ids:
                # Extract just the room number part if it's prefixed with floor name
                display_id = room_id.split('_')[-1] if '_' in room_id else room_id
                centroid = polygon.centroid
                ax.text(centroid.x, centroid.y, display_id, 
                        ha='center', va='center', fontsize=8)
            
            # Show centroids if requested
            if show_centroids:
                centroid = polygon.centroid
                ax.plot(centroid.x, centroid.y, 'ro', markersize=4)
        
        # Set aspect equal to preserve shape
        ax.set_aspect('equal')
        
        # Set title
        if floor_name:
            plt.title(f"Floor Plan: {floor_name}")
        else:
            plt.title("Combined Floor Plans")
        
        # Add color bar to show area scale
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Normalized Room Area')
        
        plt.tight_layout()
        return fig
    
    def plot_adjacency_graph(self, floor_name=None, figsize=(12, 10), 
                            layout='spring', node_size_factor=1000):
        """
        Plot the room adjacency as a network graph.
        
        Args:
            floor_name (str, optional): The specific floor to plot, or None for all floors
            figsize (tuple): Figure size
            layout (str): Graph layout type ('spring', 'kamada_kawai', 'planar')
            node_size_factor (float): Factor to control node sizes
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        # Calculate adjacency if not already done
        if not self.adjacency:
            self.calculate_adjacency()
        
        # Determine which rooms to include
        if floor_name:
            if floor_name not in self.floor_names:
                raise ValueError(f"Floor {floor_name} not found. Available floors: {self.floor_names}")
            
            # Filter adjacency to only include rooms from this floor
            floor_rooms = set(self.get_floor_rooms(floor_name).keys())
            adj = {room: [r for r in adjacent if r in floor_rooms] 
                   for room, adjacent in self.adjacency.items() if room in floor_rooms}
        else:
            adj = self.adjacency
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes (rooms)
        for room in adj:
            # Calculate node size based on room area
            if self.areas:
                # Get the area, defaulting to median if not found
                area = self.areas.get(room, np.median(list(self.areas.values())))
                node_size = area * node_size_factor / max(self.areas.values())
            else:
                node_size = 300
            
            # Add node with size attribute
            G.add_node(room, size=node_size)
        
        # Add edges (adjacencies)
        for room, adjacent_rooms in adj.items():
            for adj_room in adjacent_rooms:
                G.add_edge(room, adj_room)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get positions for nodes based on chosen layout
        if layout == 'spring':
            pos = nx.spring_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'planar':
            # Try planar layout, but fall back to spring if not possible
            try:
                pos = nx.planar_layout(G)
            except nx.NetworkXException:
                pos = nx.spring_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Get node sizes from attributes
        node_sizes = [data.get('size', 300) for _, data in G.nodes(data=True)]
        
        # Draw the graph
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, ax=ax)
        
        # Draw labels with just the room number part if it's prefixed with floor name
        labels = {node: node.split('_')[-1] if '_' in node else node for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        
        # Set title
        if floor_name:
            plt.title(f"Room Adjacency Graph: {floor_name}")
        else:
            plt.title("Room Adjacency Graph")
        
        plt.axis('off')
        plt.tight_layout()
        
        return fig
    
    def calculate_graph_features(self):
        """
        Calculate graph-theoretic features for each room based on adjacency.
        
        Returns:
            pandas DataFrame: DataFrame with room IDs and various graph features
        """
        # Calculate adjacency if not already done
        if not self.adjacency:
            self.calculate_adjacency()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges
        for room, adjacent_rooms in self.adjacency.items():
            G.add_node(room)
            for adj_room in adjacent_rooms:
                G.add_edge(room, adj_room)
        
        # Calculate graph features
        features = {
            'room_id': [],
            'degree': [],
            'betweenness_centrality': [],
            'closeness_centrality': [],
            'eigenvector_centrality': [],
            'clustering_coefficient': []
        }
        
        # Add areas if available
        if self.areas:
            features['area'] = []
            features['normalized_area'] = []
        
        # Calculate centrality measures
        betweenness = nx.betweenness_centrality(G)
        closeness = nx.closeness_centrality(G)
        
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            # Fall back to a simpler calculation if eigenvector centrality fails
            eigenvector = {node: 0.0 for node in G.nodes()}
        
        clustering = nx.clustering(G)
        
        # Compile features for each room
        for room in self.adjacency:
            features['room_id'].append(room)
            features['degree'].append(len(self.adjacency[room]))
            features['betweenness_centrality'].append(betweenness.get(room, 0.0))
            features['closeness_centrality'].append(closeness.get(room, 0.0))
            features['eigenvector_centrality'].append(eigenvector.get(room, 0.0))
            features['clustering_coefficient'].append(clustering.get(room, 0.0))
            
            # Add areas if available
            if self.areas:
                features['area'].append(self.areas.get(room, 0.0))
                features['normalized_area'].append(self.normalized_areas.get(room, 0.0))
        
        # Create DataFrame
        return pd.DataFrame(features)
    
    def export_for_graph_learning(self, output_dir, format='pytorch_geometric'):
        """
        Export room data in a format suitable for graph learning frameworks.
        
        Args:
            output_dir (str): Directory to save the exported files
            format (str): Export format ('pytorch_geometric', 'networkx', 'adjacency_matrix')
            
        Returns:
            dict: Dictionary with paths to exported files
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate required data if not already done
        if not self.areas:
            self.calculate_room_areas()
        
        if not self.normalized_areas:
            self.min_max_normalize_areas()
        
        if not self.adjacency:
            self.calculate_adjacency()
        
        # Calculate graph features
        graph_features = self.calculate_graph_features()
        
        # Prepare node features matrix
        node_features = graph_features.set_index('room_id')
        
        # Export based on requested format
        export_paths = {}
        
        if format == 'pytorch_geometric':
            # Export node features
            node_features_path = os.path.join(output_dir, 'node_features.csv')
            node_features.to_csv(node_features_path)
            export_paths['node_features'] = node_features_path
            
            # Export edge index (COO format)
            edge_index = []
            node_mapping = {node: i for i, node in enumerate(node_features.index)}
            
            for source, targets in self.adjacency.items():
                for target in targets:
                    # Only include edges where both nodes are in node_features
                    if source in node_mapping and target in node_mapping:
                        edge_index.append((node_mapping[source], node_mapping[target]))
            
            edge_index_path = os.path.join(output_dir, 'edge_index.csv')
            pd.DataFrame(edge_index, columns=['source', 'target']).to_csv(edge_index_path, index=False)
            export_paths['edge_index'] = edge_index_path
            
            # Export node mapping
            node_mapping_path = os.path.join(output_dir, 'node_mapping.csv')
            pd.DataFrame(list(node_mapping.items()), columns=['room_id', 'index']).to_csv(node_mapping_path, index=False)
            export_paths['node_mapping'] = node_mapping_path
            
        elif format == 'networkx':
            # Create and export NetworkX graph
            G = nx.Graph()
            
            # Add nodes with features
            for room_id, features in node_features.iterrows():
                G.add_node(room_id, **features.to_dict())
            
            # Add edges
            for source, targets in self.adjacency.items():
                for target in targets:
                    G.add_edge(source, target)
            
            # Save as GraphML
            graph_path = os.path.join(output_dir, 'room_graph.graphml')
            nx.write_graphml(G, graph_path)
            export_paths['graph'] = graph_path
            
        elif format == 'adjacency_matrix':
            # Export adjacency matrix
            adj_matrix = self.generate_adjacency_matrix()
            adj_matrix_path = os.path.join(output_dir, 'adjacency_matrix.csv')
            adj_matrix.to_csv(adj_matrix_path)
            export_paths['adjacency_matrix'] = adj_matrix_path
            
            # Export node features separately
            node_features_path = os.path.join(output_dir, 'node_features.csv')
            node_features.to_csv(node_features_path)
            export_paths['node_features'] = node_features_path
        
        print(f"Exported room graph data in {format} format to {output_dir}")
        return export_paths
    
    def calculate_distance_matrix(self):
        """
        Calculate a distance matrix between room centroids.
        
        Returns:
            pandas DataFrame: Distance matrix
        """
        # Get list of rooms
        rooms = list(self.centroids.keys())
        n_rooms = len(rooms)
        
        # Create empty distance matrix
        dist_matrix = np.zeros((n_rooms, n_rooms))
        
        # Fill distance matrix
        for i, room1 in enumerate(rooms):
            for j, room2 in enumerate(rooms):
                if i != j:
                    dist_matrix[i, j] = self.centroids[room1].distance(self.centroids[room2])
        
        # Create DataFrame
        return pd.DataFrame(dist_matrix, index=rooms, columns=rooms)
    
    def find_path_between_rooms(self, start_room, end_room):
        """
        Find the shortest path between two rooms using the adjacency graph.
        
        Args:
            start_room (str): Starting room ID
            end_room (str): Ending room ID
            
        Returns:
            list: List of room IDs representing the path
        """
        # Calculate adjacency if not already done
        if not self.adjacency:
            self.calculate_adjacency()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes and edges
        for room, adjacent_rooms in self.adjacency.items():
            G.add_node(room)
            for adj_room in adjacent_rooms:
                G.add_edge(room, adj_room)
        
        # Check if rooms exist
        if start_room not in G:
            raise ValueError(f"Start room {start_room} not found")
        if end_room not in G:
            raise ValueError(f"End room {end_room} not found")
        
        # Find shortest path
        try:
            path = nx.shortest_path(G, source=start_room, target=end_room)
            return path
        except nx.NetworkXNoPath:
            print(f"No path exists between {start_room} and {end_room}")
            return []
