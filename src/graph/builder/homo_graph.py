import torch
import networkx as nx
import numpy as np
from rdflib import URIRef
from typing import Dict, List, Tuple, Set, Any
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

logger = logging.getLogger(__name__)

class HomogGraphBuilderMixin:

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
