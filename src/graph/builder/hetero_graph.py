from typing import Dict, Any, Optional, List
from rdflib import URIRef
import numpy as np
import torch
from torch_geometric.data import HeteroData

import logging; logger = logging.getLogger(__name__)


class HeteroGraphBuilderMixin:
    """
    Mixin class for building heterogeneous graphs with PyTorch Geometric.
        
    Node types:
    - 'room': Room nodes with static features
    - 'device': Device nodes (intermediary between rooms and properties)
    - 'property': Property nodes with temporal measurements (one per device-property combination)
    - 'outside': Single outside node with weather data
    - 'time': Single time information node (time features)
    
    Edge types:
    - ('room', 'adjacent_horizontal', 'room'): Horizontal room-to-room adjacency (same floor, directed)
    - ('room', 'adjacent_vertical', 'room'): Vertical room-to-room adjacency (between floors, directed)
    - ('room', 'contains', 'device'): Room-to-device connections # (No message passing)
    - ('device', 'contained_in', 'room'): Device-to-room connections
    - ('device', 'measures', 'property'): Device-to-property connections # (No message passing)
    - ('property', 'measured_by', 'device'): Property-to-device connections
    - ('outside', 'influences', 'room'): Outside-to-room connections (for rooms with windows, weighted)
    - ('time', 'affects', 'room'): Time-to-room connections
    - ('time', 'affects', 'device'): Time-to-device connections
    - ('time', 'affects', 'property'): Time-to-property connections
    """
    
    def __init__(self):
        """Initialize hetero-specific attributes"""
        self.base_hetero_graph = None
        self.hetero_temporal_graphs = None
        self.node_mappings = {}
        self.reverse_node_mappings = {}
        self.feature_names: Dict[str, List[str]] = {}

    def build_base_hetero_graph(
            self, 
            horizontal_adj_matrix: np.ndarray,
            vertical_adj_matrix: np.ndarray,
            outside_adj_vector: np.ndarray
    ) -> HeteroData:
        """
        Build the base heterogeneous graph structure with static features.
        This graph will be duplicated for each time bucket.
        
        Returns:
            HeteroData: Base heterogeneous graph
        """
        # Ensure required components are available
        if not hasattr(self, 'device_level_df'):
            raise ValueError("Device-level DataFrame not found. Run build_device_level_df first.")
        
        # Initialize heterogeneous graph
        hetero_data = HeteroData()
        
        # Build node mappings and features
        self._build_node_mappings()
        self._add_room_nodes(hetero_data)
        self._add_device_nodes(hetero_data)
        self._add_property_nodes(hetero_data)
        self._add_outside_node(hetero_data)
        self._add_time_node(hetero_data)
        
        # Build edges
        self._add_horizontal_room_edges(hetero_data, horizontal_adj_matrix)
        self._add_vertical_room_edges(hetero_data, vertical_adj_matrix)
        self._add_room_device_edges(hetero_data)
        self._add_device_property_edges(hetero_data)
        self._add_outside_to_room_edges(hetero_data, outside_adj_vector)
        self._add_time_edges(hetero_data)

        self.base_hetero_graph = hetero_data
        logger.info("Built base heterogeneous graph")
        self._log_graph_statistics(hetero_data)
        
        return hetero_data
    
    def _build_node_mappings(self) -> None:
        """Build mappings between original URIs/IDs and node indices."""
        self.node_mappings = {
            'room': {},
            'device': {},
            'property': {},
            'outside': {},
            'time': {}
        }
        self.reverse_node_mappings = {
            'room': {},
            'device': {},
            'property': {},
            'outside': {},
            'time': {}
        }
        
        # Room nodes: Use the same order as adjacency matrix
        for idx, room_uri_str in enumerate(self.room_URIs_str):
            self.node_mappings['room'][room_uri_str] = idx
            self.reverse_node_mappings['room'][idx] = room_uri_str
        
        # Device nodes: Extract unique devices from the DataFrame
        unique_devices = self.device_level_df['device_uri_str'].unique()
        for idx, device_uri_str in enumerate(unique_devices):
            self.node_mappings['device'][device_uri_str] = idx
            self.reverse_node_mappings['device'][idx] = device_uri_str

        # Property nodes: One node per unique (device, property_type) combination
        property_idx = 0
        self.property_type_lookup = {}
        
        # Get unique device-property combinations from the DataFrame
        unique_combinations = self.device_level_df[['device_uri_str', 'property_type']].drop_duplicates()
        
        for _, row in unique_combinations.iterrows():
            device_uri_str = row['device_uri_str']
            property_type = row['property_type']
                        
            # Create key
            device_property_key = (device_uri_str, property_type)
            self.node_mappings['property'][device_property_key] = property_idx
            self.reverse_node_mappings['property'][property_idx] = device_property_key
            self.property_type_lookup[device_property_key] = property_type
            property_idx += 1
        
        # Outside node (single node)
        self.node_mappings['outside']['outside'] = 0
        self.reverse_node_mappings['outside'][0] = 'outside'
        
        # Time node (single node)
        self.node_mappings['time']['time'] = 0
        self.reverse_node_mappings['time'][0] = 'time'
        
        logger.info(f"Created node mappings: {len(self.node_mappings['room'])} rooms, "
                f"{len(self.node_mappings['device'])} devices, "
                f"{len(self.node_mappings['property'])} device-property combinations, "
                f"1 outside node, 1 time node")
        return None
    
    #############################
    # Nodes
    #############################

    def _add_room_nodes(self, hetero_data: HeteroData) -> None:
        """Add room nodes with static features."""
        if not hasattr(self, 'static_room_features_df'):
            raise ValueError("Static room features DataFrame not found. Run build_static_room_features_df() first.")
        static_df = self.static_room_features_df.copy()
        # Vectorize static features
        static_features_np = static_df.to_numpy(dtype=np.float32)
        hetero_data["room"].x = torch.from_numpy(static_features_np)
        n_r, n_f = static_features_np.shape
        logger.info("Added %d room nodes with %d static features (vectorised)", n_r, n_f)
        # Save feature names
        self.feature_names['room'] = static_df.columns.tolist()
        logger.info(f"Stored {len(self.feature_names['room'])} feature names for 'room' nodes.")
        return None
    
    def _add_device_nodes(self, hetero_data: HeteroData) -> None:
        """Add device nodes with features including one-hot device type encoding."""
        n_devices = len(self.node_mappings['device'])
        
        # First pass: collect all unique device types
        device_types = set()
        for device_uri_str in self.node_mappings['device'].keys():
            device = self.office_graph.devices[URIRef(device_uri_str)]
            if hasattr(device, 'device_type') and device.device_type:
                device_types.add(device.device_type)
        
        # Sort for consistent ordering
        device_types = sorted(list(device_types))
        n_device_types = len(device_types)
        
        # Create device type to index mapping
        self.device_type_to_idx = {dtype: idx for idx, dtype in enumerate(device_types)}
        self.idx_to_device_type = {idx: dtype for dtype, idx in self.device_type_to_idx.items()}
        
        logger.info(f"Found {n_device_types} unique device types: {device_types}")
        
        ### Static Device Features: [num_properties] + [device_type_one_hot_encoding] ###
        
        # Define and store feature names for device nodes
        device_static_feature_list = ['num_properties'] + [f"device_type_{dtype}" for dtype in device_types]
        self.feature_names['device'] = device_static_feature_list
        logger.info(f"Stored {len(self.feature_names['device'])} feature names for 'device' nodes.")
        
        # NOTE: We can zero-impute here, since this information is fully-deduced.
        n_features = 1 + n_device_types
        device_features = torch.zeros(n_devices, n_features, dtype=torch.float32)
        
        for device_uri_str, device_idx in self.node_mappings['device'].items():
            device = self.office_graph.devices[URIRef(device_uri_str)]
            
            # Feature 0: Number of properties this device measures
            device_features[device_idx, 0] = len(device.properties)
            
            # Features 1 to n_device_types: One-hot encoding of device type
            if hasattr(device, 'device_type') and device.device_type:
                device_type = device.device_type
                if device_type in self.device_type_to_idx:
                    type_idx = self.device_type_to_idx[device_type]
                    device_features[device_idx, 1 + type_idx] = 1.0
                else:
                    logger.warning(f"Unknown device type: {device_type}")
        
        hetero_data['device'].x = device_features
        
        logger.info(f"Added {n_devices} device nodes with {n_features} features")
        logger.info(f"Device features: [num_properties] + one-hot[{', '.join(device_types)}]")
        return None
    
    def _add_property_nodes(self, hetero_data: HeteroData) -> None:
        """Add property nodes with one-hot encoded property type features."""
        n_properties = len(self.node_mappings['property'])
        
        # Get unique property types from the DataFrame
        property_types = sorted(self.device_level_df['property_type'].unique())
        n_property_types = len(property_types)
        
        # Create property type to index mapping
        self.property_type_to_idx = {prop_type: idx for idx, prop_type in enumerate(property_types)}
        self.idx_to_property_type = {idx: prop_type for prop_type, idx in self.property_type_to_idx.items()}
        
        # Store for later use in temporal updates
        self.actual_property_types = property_types
        
        logger.info(f"Found {n_property_types} unique property types: {property_types}")
        
        ### Static Property Features: [property_type_one_hot_encoding] ###

        # Define and store feature names for property nodes
        static_prop_features = [f"property_type_{ptype}" for ptype in property_types]
        temporal_prop_features = self.device_level_df_temporal_feature_names
        self.feature_names['property'] = static_prop_features + temporal_prop_features
        logger.info(f"Stored {len(self.feature_names['property'])} feature names for 'property' nodes.")
        
        # NOTE: We can zero-impute here, since this information is fully-deduced.
        n_features = n_property_types
        property_features = torch.zeros(n_properties, n_features, dtype=torch.float32)
        
        # Fill in the one-hot encoding
        for (device_uri_str, prop_type), prop_idx in self.node_mappings['property'].items():
            type_idx = self.property_type_to_idx[prop_type]
            property_features[prop_idx, type_idx] = 1.0

        hetero_data['property'].x = property_features
        logger.info(f"Added {n_properties} property nodes with {n_property_types} features")
        logger.info(f"Property features: one-hot[{', '.join(property_types)}]")
        return None
    
    def _add_outside_node(self, hetero_data: HeteroData) -> None:
        """
        Add single outside node for weather data.

        Initializes with placeholder for weather features.
        Will be filled with actual weather data per time bucket.
        """        
        # Initialize with placeholder zero features
        num_weather_feats = len(self.weather_feature_names)
        hetero_data['outside'].x = torch.full(
            (1, num_weather_feats), 
            float('nan'), dtype=torch.float32)
        logger.info(f"Added 1 outside node with placeholder dim={num_weather_feats} "
                    "(actual features will be filled per time bucket)")
        # Save feature names
        self.feature_names['outside'] = self.weather_feature_names
        logger.info(f"Stored {len(self.feature_names['outside'])} feature names for 'outside' node.")
        return None
    
    def _add_time_node(self, hetero_data: HeteroData) -> None:
        """
        Add single time information node for time features.
        
        Initializes with placeholder features. 
        Will be filled later per time bucket.
        """
        # Initialize with placeholder zero features
        num_time_features = len(self.time_feature_names)
        hetero_data['time'].x = torch.full(
            (1, num_time_features), 
            float('nan'), dtype=torch.float32)
        logger.info(f"Added 1 time node with {num_time_features} temporal features")
        # Save feature names
        self.feature_names['time'] = self.time_feature_names
        logger.info(f"Stored {len(self.feature_names['time'])} feature names for 'time' node.")
        return None
    
    #############################
    # Edges
    #############################

    def _add_adjacency_edges(
            self,
            hetero_data: HeteroData,
            adj_matrix: np.ndarray,
            edge_name: str
    ) -> None:
        """
        A generic helper to add weighted, directed edges to a HeteroData object
        from an adjacency matrix.

        Args:
            hetero_data: The graph data object to modify.
            adj_matrix: The (N, N) numpy array representing the adjacency matrix.
            edge_name: The name for the edge type (e.g., 'adjacent_horizontal').
        """
        # Edge type is defined as a tuple of (source_type, edge_name, target_type)
        edge_type = ('room', edge_name, 'room')

        # Find the indices [row, col] of all non-zero entries
        edge_idx = np.argwhere(adj_matrix > 0)
        if edge_idx.size == 0:
            logger.warning("No %s edges found", edge_name)
            return None
        
        # Get edge weights
        edge_weights = adj_matrix[edge_idx[:, 0], edge_idx[:, 1]].astype(np.float32)

        # Convert to PyTorch tensors and assign to the hetero_data object
        hetero_data[edge_type].edge_index = torch.as_tensor(edge_idx.T, dtype=torch.long)
        hetero_data[edge_type].edge_attr  = torch.from_numpy(edge_weights).unsqueeze(1)
        
        # Use the edge_name to create a user-friendly log message
        logger.info(f"Added {len(edge_idx)} {edge_name} room-to-room edges")
        return None
    
    def _add_horizontal_room_edges(
            self, 
            hetero_data: HeteroData, 
            horizontal_adj_matrix: np.ndarray
    ) -> None:
        """Add directed horizontal room-to-room edges based on the horizontal adjacency matrix."""
        self._add_adjacency_edges(
            hetero_data=hetero_data,
            adj_matrix=horizontal_adj_matrix,
            edge_name='adjacent_horizontal'
        )
        return None
    
    def _add_vertical_room_edges(
            self, 
            hetero_data: HeteroData, 
            vertical_adj_matrix: np.ndarray
    ) -> None:
        """Add directed vertical room-to-room edges based on the vertical adjacency matrix."""
        self._add_adjacency_edges(
            hetero_data=hetero_data,
            adj_matrix=vertical_adj_matrix,
            edge_name='adjacent_vertical'
        )
        return None

    def _add_room_device_edges(self, hetero_data: HeteroData) -> None:
        """Add bidirectional edges between rooms and devices."""
        room_to_device_edges = []
        
        for room_uri_str, room_idx in self.node_mappings['room'].items():
            room = self.office_graph.rooms[URIRef(room_uri_str)]
            
            for device_uri_str in room.devices:
                if device_uri_str in self.node_mappings['device']:
                    device_idx = self.node_mappings['device'][device_uri_str]
                    room_to_device_edges.append([room_idx, device_idx])
        
        if room_to_device_edges:
            edge_index = torch.tensor(room_to_device_edges, dtype=torch.long).t().contiguous()
            hetero_data['room', 'contains', 'device'].edge_index = edge_index
            hetero_data['device', 'contained_in', 'room'].edge_index = edge_index.flip([0])
            logger.info(f"Added {len(room_to_device_edges)} bidirectional room-device edges")
        return None

    def _add_device_property_edges(self, hetero_data: HeteroData) -> None:
        """Add bidirectional edges between devices and properties."""
        device_to_property_edges = []
        
        for (device_uri_str, _), prop_idx in self.node_mappings['property'].items():
            if device_uri_str in self.node_mappings['device']:
                device_idx = self.node_mappings['device'][device_uri_str]
                device_to_property_edges.append([device_idx, prop_idx])
        
        if device_to_property_edges:
            edge_index = torch.tensor(device_to_property_edges, dtype=torch.long).t().contiguous()
            hetero_data['device', 'measures', 'property'].edge_index = edge_index
            hetero_data['property', 'measured_by', 'device'].edge_index = edge_index.flip([0])
            logger.info(f"Added {len(device_to_property_edges)} bidirectional device-property edges")
        return None

    def _add_outside_to_room_edges(self, hetero_data: HeteroData, outside_adj_vector: np.ndarray) -> None:
        """Add weighted edges from outside node to rooms (no reverse edges)."""        
        # Edge type is defined as a tuple of (source_type, edge_name, target_type)
        edge_type = ('outside', 'influences', 'room')

        # Single outside node
        outside_idx = 0

        # Find the indices of all rooms that have a connection from the outside
        room_idx = np.nonzero(outside_adj_vector > 0)[0]
        if room_idx.size == 0:
            logger.warning("No outside-to-room edges found")
            return None
        
        # Get the corresponding weights for those connections
        edge_weights = outside_adj_vector[room_idx].astype(np.float32)

        # Convert to PyTorch tensors and assign to the hetero_data object
        edge_index = torch.vstack((
            torch.full_like(torch.as_tensor(room_idx), outside_idx, dtype=torch.long),
            torch.as_tensor(room_idx, dtype=torch.long),
        ))  # shape (2, E)
        hetero_data[edge_type].edge_index = edge_index
        hetero_data[edge_type].edge_attr  = torch.from_numpy(edge_weights).unsqueeze(1)
        logger.info(f"Added {len(room_idx)} weighted outside-to-room edges")
        return None

    def _add_time_edges(self, hetero_data: HeteroData) -> None:
        """Add edges from time node to all other nodes."""
        # Single time node
        time_node_idx = 0

        # Connect time node to all other node types
        node_types_to_connect = ['room', 'device', 'property']
        for node_type in node_types_to_connect:
            num_nodes = hetero_data[node_type].num_nodes
            if num_nodes == 0:
                logger.warning(f"No {node_type} nodes to connect from time node")
                continue
            
            # Create edge index for time to node_type connections
            source_idx = torch.full((num_nodes,), time_node_idx, dtype=torch.long)
            target_idx = torch.arange(num_nodes, dtype=torch.long)
            edge_index = torch.stack([source_idx, target_idx])

            # Assign to the hetero_data object
            hetero_data['time', f'affects', node_type].edge_index = edge_index
            logger.info(f"Added {num_nodes} time-to-{node_type} edges")
        return None
    
    #############################
    # Temporal Snapshots
    #############################

    def build_hetero_temporal_graphs(self) -> Dict[int, HeteroData]:
        """
        Build temporal heterogeneous graph snapshots for each time bucket.
            
        Returns:
            Dictionary mapping time bucket indices to HeteroData graphs
        """
        if not hasattr(self, 'base_hetero_graph'):
            raise ValueError("Base heterogeneous graph not found. Call build_base_hetero_graph() first.")        
        if not hasattr(self, 'time_buckets'):
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")
        
        # Logging
        total_buckets = len(self.time_buckets)
        log_interval = max(1, total_buckets // 20)
        
        # Building temporal graphs
        hetero_temporal_graphs = {}
        for bucket_idx in range(len(self.time_buckets)):
            if bucket_idx % log_interval == 0 or bucket_idx == total_buckets - 1:
                logger.info(f"Building temporal graph {bucket_idx + 1}/{total_buckets}")
            
            # Create a hybrid-copy of the base graph
            time_graph = self._copy_hetero_graph(self.base_hetero_graph)

            # Update property nodes with temporal measurements from the DataFrame
            self._update_property_nodes_temporal(time_graph, bucket_idx)

            # Update outside node with weather data
            self._update_outside_node_temporal(time_graph, bucket_idx)

            # Update time node with temporal features
            self._update_time_node_temporal(time_graph, bucket_idx)

            hetero_temporal_graphs[bucket_idx] = time_graph

        self.hetero_temporal_graphs = hetero_temporal_graphs
        logger.info(f"Built {len(hetero_temporal_graphs)} temporal heterogeneous graphs")
        
        return hetero_temporal_graphs
    
    def _update_property_nodes_temporal(
            self,
            hetero_graph: HeteroData,
            bucket_idx: int,
        ) -> None:
        """Update property nodes with measurements from the DataFrame for this time bucket."""
        if not hasattr(self, 'device_level_df'):
            raise ValueError("Device-level DataFrame not found. Run build_device_level_df first.")
        df = self.device_level_df        
        bucket_df = df[df['bucket_idx'] == bucket_idx]

        # Define the desired columns and the ordered list of node keys
        cols = self.device_level_df_temporal_feature_names
        keys = list(self.node_mappings['property'].keys())
        
        # Vectorized feature alignment
        block = (bucket_df
                .set_index(['device_uri_str', 'property_type'])[cols]
                .astype(np.float32))
        aligned = block.reindex(keys)
        
        # Build the tensor
        prop_feat_mat_temporal = torch.from_numpy(
            aligned.to_numpy(dtype=np.float32, copy=False)
        )
        
        # Concatenate static and temporal property features
        prop_feat_mat_static = hetero_graph['property'].x
        hetero_graph['property'].x = torch.cat([prop_feat_mat_static, prop_feat_mat_temporal], dim=1)

        return None
    
    def _update_outside_node_temporal(
            self, 
            hetero_graph: HeteroData, 
            bucket_idx: int
    ) -> None:
        """Update outside node with weather features for this time bucket."""
        if not hasattr(self, 'weather_df'):
            raise ValueError("Weather DataFrame is not available. Run build_weather_df first")

        if bucket_idx < len(self.weather_df):
            row = self.weather_df.iloc[bucket_idx][self.weather_feature_names]
            features = row.to_numpy(np.float32)[None, :]          # shape (1, D)
            hetero_graph["outside"].x = torch.as_tensor(features, dtype=torch.float32)
        else:
            logger.warning(f"No weather data available for bucket {bucket_idx}.")
            hetero_graph['outside'].x = torch.full(
                (1, len(self.weather_feature_names)), 
                float('nan'), dtype=torch.float32)
        return None
    
    def _update_time_node_temporal(
            self, 
            hetero_graph: HeteroData, 
            bucket_idx: int,
    ) -> None:
        """Update time node with time features for this time bucket."""
        if not hasattr(self, "time_features_df"):
            raise AttributeError("time_features_df not found; run build_time_features_df() first.")
        
        if bucket_idx < len(self.time_features_df):
            row = self.time_features_df.iloc[bucket_idx][self.time_feature_names]
            feats = row.to_numpy(np.float32)[None, :]          # shape (1, 5)
            hetero_graph["time"].x = torch.as_tensor(feats, dtype=torch.float32)
        else:
            logger.warning(f"No time data available for bucket {bucket_idx}.")
            hetero_graph['time'].x = torch.full(
                (1, len(self.time_feature_names)), 
                float('nan'), dtype=torch.float32)
        return None

    def _copy_hetero_graph(
            self, 
            hetero_graph: HeteroData
    ) -> HeteroData:
        """Create a shallow/deep hybrid copy: static nodes share memory, temporal nodes are cloned."""
        new_graph = HeteroData()
        
        # Nodes
        for node_type in hetero_graph.node_types:
            x = hetero_graph[node_type].x
            if node_type in ('room', 'device'):
                # static features -> share the same tensor
                new_graph[node_type].x = x
            else:
                # temporal features -> deep-copy so we can overwrite
                new_graph[node_type].x = x.clone()

        # Edges
        for edge_type in hetero_graph.edge_types:
            data = hetero_graph[edge_type]
            # edge_index and edge_attr never change -> shallow copy
            new_graph[edge_type].edge_index = data.edge_index
            if hasattr(data, 'edge_attr'):
                new_graph[edge_type].edge_attr = data.edge_attr
        
        return new_graph
        
    def prepare_hetero_stgcn_input(self) -> Dict[str, Any]:
        """
        Prepare all necessary inputs for a heterogeneous STGCN model.
                    
        Returns:
            Dictionary containing all hetero STGCN inputs
        """
        if not hasattr(self, 'hetero_temporal_graphs') or not self.hetero_temporal_graphs:
            raise ValueError("Temporal hetero graphs not available. Run build_hetero_temporal_graphs first.")
        
        hetero_stgcn_input = {
            "base_graph": self.base_hetero_graph,
            "temporal_graphs": self.hetero_temporal_graphs,
            "node_mappings": self.node_mappings,
            "reverse_node_mappings": self.reverse_node_mappings,
            "feature_names": self.feature_names,
        }
        
        logger.info("Heterogeneous STGCN input preparation complete.")
        return hetero_stgcn_input
    

    #############################
    # Statistics and Schema
    #############################

    def _log_graph_statistics(self, hetero_graph: HeteroData):
        """Log statistics about the heterogeneous graph."""
        logger.info("Heterogeneous graph statistics:")
        
        # Node statistics
        for node_type in hetero_graph.node_types:
            if hasattr(hetero_graph[node_type], 'x'):
                n_nodes, n_features = hetero_graph[node_type].x.shape
                logger.info(f"  {node_type} nodes: {n_nodes} nodes, {n_features} features")
        
        # Edge statistics
        for edge_type in hetero_graph.edge_types:
            if hasattr(hetero_graph[edge_type], 'edge_index'):
                n_edges = hetero_graph[edge_type].edge_index.shape[1]
                logger.info(f"  {edge_type} edges: {n_edges}")
    
    def visualize_hetero_graph_schema(self, save_path: Optional[str] = None) -> str:
        """
        Create a text representation of the heterogeneous graph schema.
        
        Args:
            save_path: Optional path to save the schema as a text file
            
        Returns:
            String representation of the graph schema
        """
        if self.base_hetero_graph is None:
            raise ValueError("Base heterogeneous graph not built. Run build_base_hetero_graph first.")
        
        schema = []
        schema.append("=" * 60)
        schema.append("HETEROGENEOUS GRAPH SCHEMA")
        schema.append("=" * 60)
        schema.append("")
        
        # Node types
        schema.append("NODE TYPES:")
        schema.append("-" * 20)
        for node_type in sorted(self.base_hetero_graph.node_types):
            if hasattr(self.base_hetero_graph[node_type], 'x'):
                n_nodes, n_features = self.base_hetero_graph[node_type].x.shape
                schema.append(f"  {node_type:12} : {n_nodes:4} nodes, {n_features:2} features")
        schema.append("")
        
        # Edge types
        schema.append("EDGE TYPES:")
        schema.append("-" * 20)
        for edge_type in sorted(self.base_hetero_graph.edge_types):
            src, rel, dst = edge_type
            if hasattr(self.base_hetero_graph[edge_type], 'edge_index'):
                n_edges = self.base_hetero_graph[edge_type].edge_index.shape[1]
                # Special formatting for room adjacencies:
                if rel in ['adjacent_horizontal', 'adjacent_vertical']:
                    rel_display = f"{rel:>18}"
                else:
                    rel_display = f"{rel:>15}"
                schema.append(f"  {src:8} --{rel_display}--> {dst:8} : {n_edges:4} edges")
        schema.append("")
        
        # Feature descriptions
        schema.append("FEATURE DESCRIPTIONS:")
        schema.append("-" * 30)
        schema.append("  room features    : Static room attributes (area, windows, etc.)")
        schema.append("  device features  : Number of properties, device type")
        schema.append("  property features: Property type encoding + temporal measurements")
        schema.append("  outside features : Weather data (normalized)")
        schema.append("  time features : Temporal info (day_of_week, hour, is_workday)")
        schema.append("")
        
        # Adjacency info
        schema.append("ADJACENCY INFORMATION:")
        schema.append("-" * 30)
        schema.append("  horizontal : Same-floor room connections")
        schema.append("  vertical   : Inter-floor room connections (overlapping spaces)")
        schema.append("")
        
        # Additional info
        if hasattr(self, 'hetero_temporal_graphs') and self.hetero_temporal_graphs:
            schema.append(f"TEMPORAL SNAPSHOTS: {len(self.hetero_temporal_graphs)}")
        
        schema_str = "\n".join(schema)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(schema_str)
            logger.info(f"Saved graph schema to {save_path}")
        
        return schema_str