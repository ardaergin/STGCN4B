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
    - 'property':
        - 'prop_Temperature': Temperature measurement nodes (one per device measuring temperature)
        - 'prop_CO2Level': CO2 level measurement nodes (one per device measuring CO2)
        - 'prop_Humidity': Humidity measurement nodes (one per device measuring humidity)
    - 'outside': Single outside node with weather data
    - 'time': Single time information node (time features)
    
    Edge types:
    - ('room', 'adjacent_horizontal', 'room'): Horizontal room-to-room adjacency (same floor, directed)
    - ('room', 'adjacent_vertical', 'room'): Vertical room-to-room adjacency (between floors, directed)
    
    - ('room', 'contains', 'device'): Room-to-device connections
    - ('device', 'contained_in', 'room'): Device-to-room connections
    
    - ('device', 'measures_Temperature', 'prop_Temperature'): Device-to-temperature property connections
    - ('prop_Temperature', 'measured_by', 'device'): Temperature property-to-device connections
    - ('device', 'measures_CO2Level', 'prop_CO2Level'): Device-to-CO2 property connections
    - ('prop_CO2Level', 'measured_by', 'device'): CO2 property-to-device connections
    - ('device', 'measures_Humidity', 'prop_Humidity'): Device-to-humidity property connections
    - ('prop_Humidity', 'measured_by', 'device'): Humidity property-to-device connections
    
    - ('outside', 'influences', 'room'): Outside-to-room connections (for rooms with windows, weighted)
    
    - ('time', 'affects_room', 'room'): Time-to-room connections
    - ('time', 'affects_device', 'device'): Time-to-device connections
    - ('time', 'affects_prop_Temperature', 'prop_Temperature'): Time-to-temperature property connections
    - ('time', 'affects_prop_CO2Level', 'prop_CO2Level'): Time-to-CO2 property connections
    - ('time', 'affects_prop_Humidity', 'prop_Humidity'): Time-to-humidity property connections
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
        self._add_property_nodes_by_type(hetero_data)
        self._add_outside_node(hetero_data)
        self._add_time_node(hetero_data)
        
        # Build edges
        self._add_horizontal_room_edges(hetero_data, horizontal_adj_matrix)
        self._add_vertical_room_edges(hetero_data, vertical_adj_matrix)
        self._add_room_device_edges(hetero_data)
        self._add_device_property_edges_by_type(hetero_data)
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
            'outside': {},
            'time': {}
        }
        self.reverse_node_mappings = {
            'room': {},
            'device': {},
            'outside': {},
            'time': {}
        }
        
        # Initialize mappings for each property type
        for prop_type in self.used_property_types:
            prop_node_type = f'prop_{prop_type}'
            self.node_mappings[prop_node_type] = {}
            self.reverse_node_mappings[prop_node_type] = {}
        
        # Room nodes: Use the same order as adjacency matrix
        for idx, room_uri_str in enumerate(self.room_URIs_str):
            self.node_mappings['room'][room_uri_str] = idx
            self.reverse_node_mappings['room'][idx] = room_uri_str
        
        # Device nodes: Extract unique devices from the DataFrame
        unique_devices = self.device_level_df['device_uri_str'].unique()
        for idx, device_uri_str in enumerate(unique_devices):
            self.node_mappings['device'][device_uri_str] = idx
            self.reverse_node_mappings['device'][idx] = device_uri_str

        # Property nodes: Separate indices for each property type
        unique_combinations = self.device_level_df[['device_uri_str', 'property_type']].drop_duplicates()
        
        # Group by property type and assign indices
        for prop_type in self.used_property_types:
            prop_node_type = f'prop_{prop_type}'
            prop_df = unique_combinations[unique_combinations['property_type'] == prop_type]
            
            for idx, (_, row) in enumerate(prop_df.iterrows()):
                device_uri_str = row['device_uri_str']
                # Use device_uri_str as the key (since one device has at most one node per property type)
                self.node_mappings[prop_node_type][device_uri_str] = idx
                self.reverse_node_mappings[prop_node_type][idx] = device_uri_str
        
        # Outside node (single node)
        self.node_mappings['outside']['outside'] = 0
        self.reverse_node_mappings['outside'][0] = 'outside'
        
        # Time node (single node)
        self.node_mappings['time']['time'] = 0
        self.reverse_node_mappings['time'][0] = 'time'
        
        # Log statistics
        logger.info(f"Created node mappings: {len(self.node_mappings['room'])} rooms, "
                f"{len(self.node_mappings['device'])} devices")
        for prop_type in self.used_property_types:
            prop_node_type = f'prop_{prop_type}'
            logger.info(f"  {prop_node_type}: {len(self.node_mappings[prop_node_type])} nodes")
        logger.info("  1 outside node, 1 time node")
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
        """
        Add device nodes with features including:
        - multi-hot encoding for measured properties
        - one-hot encoding for the device type.
        """
        n_devices = len(self.node_mappings['device'])
        
        # 1. Discover all unique property types to define feature dimensions
        all_property_types = sorted(self.device_level_df['property_type'].unique())
        property_type_to_idx = {ptype: i for i, ptype in enumerate(all_property_types)}
        n_property_types = len(all_property_types)
        
        # 2. Discover all unique device types for one-hot encoding
        device_types = set()
        for device_uri_str in self.node_mappings['device'].keys():
            device = self.office_graph.devices[URIRef(device_uri_str)]
            if hasattr(device, 'device_type') and device.device_type:
                device_types.add(device.device_type)
        
        sorted_device_types = sorted(list(device_types))
        device_type_to_idx = {dtype: i for i, dtype in enumerate(sorted_device_types)}
        n_device_types = len(sorted_device_types)
        
        logger.info(f"Found {n_property_types} unique property types for device features: {all_property_types}")
        logger.info(f"Found {n_device_types} unique device types: {sorted_device_types}")
        
        # 3. Define the new feature structure and names
        prop_feature_names = [f"measures_{ptype}" for ptype in all_property_types]
        type_feature_names = [f"is_type_{dtype}" for dtype in sorted_device_types]
        self.feature_names['device'] = prop_feature_names + type_feature_names
        
        n_features = n_property_types + n_device_types
        device_features = torch.zeros(n_devices, n_features, dtype=torch.float32)
        
        # Helper mapping from property URI to its simple string type
        prop_uri_to_type = {
            str(uri): p_type
            for p_type, uris in self.office_graph.property_type_mappings.items()
            for uri in uris
        }
        
        # 4. Populate the feature tensor for each device
        for device_uri_str, device_idx in self.node_mappings['device'].items():
            device = self.office_graph.devices[URIRef(device_uri_str)]
            
            # Part 1: Multi-hot encode the properties this device measures
            for prop_uri in device.properties:
                prop_type_str = prop_uri_to_type.get(str(prop_uri))
                if prop_type_str in property_type_to_idx:
                    p_idx = property_type_to_idx[prop_type_str]
                    device_features[device_idx, p_idx] = 1.0
            
            # Part 2: One-hot encode the device type
            if hasattr(device, 'device_type') and device.device_type:
                if device.device_type in device_type_to_idx:
                    d_idx = device_type_to_idx[device.device_type]
                    device_features[device_idx, n_property_types + d_idx] = 1.0
        
        hetero_data['device'].x = device_features
        
        logger.info(f"Added {n_devices} device nodes with {n_features} features.")
        logger.info(f"Device features structure: [multi-hot-properties] + [one-hot-type]")
        
        return None
    
    def _add_property_nodes_by_type(self, hetero_data: HeteroData) -> None:
        """Add property nodes as separate node types for each property."""
        # Store temporal feature names for later use
        self.temporal_prop_features = [
            f for f in self.device_level_df_temporal_feature_names
            if f in self.args.hetero_prop_features
        ]
        
        for prop_type in self.used_property_types:
            prop_node_type = f'prop_{prop_type}'
            n_nodes = len(self.node_mappings[prop_node_type])
            
            if n_nodes == 0:
                logger.warning(f"No nodes found for property type {prop_type}")
                continue
            
            # For base graph, we only need placeholders for temporal features
            # No static features needed since the property type is encoded in the node type itself
            n_temporal_features = len(self.temporal_prop_features)
            
            # Initialize with NaN placeholders (will be filled per time bucket)
            property_features = torch.full(
                (n_nodes, n_temporal_features), 
                float('nan'), 
                dtype=torch.float32
            )
            
            hetero_data[prop_node_type].x = property_features
            
            # Store feature names for this property type
            self.feature_names[prop_node_type] = self.temporal_prop_features
            
            logger.info(f"Added {n_nodes} {prop_node_type} nodes with {n_temporal_features} temporal feature placeholders")
        
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
        """
        edge_type = ('room', edge_name, 'room')
        edge_idx = np.argwhere(adj_matrix > 0)
        if edge_idx.size == 0:
            logger.warning("No %s edges found", edge_name)
            return None
        
        edge_weights = adj_matrix[edge_idx[:, 0], edge_idx[:, 1]].astype(np.float32)
        hetero_data[edge_type].edge_index = torch.as_tensor(edge_idx.T, dtype=torch.long)
        hetero_data[edge_type].edge_attr  = torch.from_numpy(edge_weights).unsqueeze(1)
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
            
            for device_uri_ref in room.devices:
                device_uri_str = str(device_uri_ref)
                if device_uri_str in self.node_mappings['device']:
                    device_idx = self.node_mappings['device'][device_uri_str]
                    room_to_device_edges.append([room_idx, device_idx])
        
        if room_to_device_edges:
            edge_index = torch.tensor(room_to_device_edges, dtype=torch.long).t().contiguous()
            hetero_data['room', 'contains', 'device'].edge_index = edge_index
            hetero_data['device', 'contained_in', 'room'].edge_index = edge_index.flip([0])
            logger.info(f"Added {len(room_to_device_edges)} bidirectional room-device edges")
        else:
            logger.warning("No room-device connections were found after processing all rooms.")
        return None
    
    def _add_device_property_edges_by_type(self, hetero_data: HeteroData) -> None:
        """Add bidirectional edges between devices and property nodes (separated by type)."""
        # Get device-property combinations from the DataFrame
        unique_combinations = self.device_level_df[['device_uri_str', 'property_type']].drop_duplicates()
        
        for prop_type in self.used_property_types:
            prop_node_type = f'prop_{prop_type}'
            edge_name_forward = f'measures_{prop_type}'
            edge_name_backward = 'measured_by'
            
            # Filter combinations for this property type
            prop_df = unique_combinations[unique_combinations['property_type'] == prop_type]
            device_to_property_edges = []
            
            for _, row in prop_df.iterrows():
                device_uri_str = row['device_uri_str']
                
                if device_uri_str in self.node_mappings['device']:
                    device_idx = self.node_mappings['device'][device_uri_str]
                    prop_idx = self.node_mappings[prop_node_type][device_uri_str]
                    device_to_property_edges.append([device_idx, prop_idx])
            
            if device_to_property_edges:
                edge_index = torch.tensor(device_to_property_edges, dtype=torch.long).t().contiguous()
                hetero_data['device', edge_name_forward, prop_node_type].edge_index = edge_index
                hetero_data[prop_node_type, edge_name_backward, 'device'].edge_index = edge_index.flip([0])
                logger.info(f"Added {len(device_to_property_edges)} bidirectional device-{prop_node_type} edges")
            else:
                logger.warning(f"No device-{prop_node_type} connections found")
        
        return None

    def _add_outside_to_room_edges(self, hetero_data: HeteroData, outside_adj_vector: np.ndarray) -> None:
        """Add weighted edges from outside node to rooms (no reverse edges)."""        
        edge_type = ('outside', 'influences', 'room')
        outside_idx = 0

        room_idx = np.nonzero(outside_adj_vector > 0)[0]
        if room_idx.size == 0:
            logger.warning("No outside-to-room edges found")
            return None
        
        edge_weights = outside_adj_vector[room_idx].astype(np.float32)
        edge_index = torch.vstack((
            torch.full_like(torch.as_tensor(room_idx), outside_idx, dtype=torch.long),
            torch.as_tensor(room_idx, dtype=torch.long),
        ))
        hetero_data[edge_type].edge_index = edge_index
        hetero_data[edge_type].edge_attr  = torch.from_numpy(edge_weights).unsqueeze(1)
        logger.info(f"Added {len(room_idx)} weighted outside-to-room edges")
        return None

    def _add_time_edges(self, hetero_data: HeteroData) -> None:
        """
        Add edges from time node to all other nodes,
        using a specific relation name for each destination node type.
        e.g., 'affects_room', 'affects_device', 'affects_Temperature'.
        """
        time_node_idx = 0
        
        # Define all node types that the time node should connect to
        target_node_types = ['room', 'device'] + [f'prop_{ptype}' for ptype in self.used_property_types]
        
        for node_type in target_node_types:
            # Check if the destination node type exists and has nodes
            if node_type not in hetero_data.node_types or hetero_data[node_type].num_nodes == 0:
                logger.warning(f"Skipping time edges for empty node type: {node_type}")
                continue
            
            # Create a specific edge name for each destination node type
            edge_name = f'affects_{node_type}'
            
            num_nodes = hetero_data[node_type].num_nodes
            source_idx = torch.full((num_nodes,), time_node_idx, dtype=torch.long)
            target_idx = torch.arange(num_nodes, dtype=torch.long)
            edge_index = torch.stack([source_idx, target_idx])
            
            # Assign edges using the new, specific relation name
            hetero_data['time', edge_name, node_type].edge_index = edge_index
            logger.info(f"Added {num_nodes} time-to-{node_type} edges using relation '{edge_name}'")
        
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

            # Update property nodes with temporal measurements
            self._update_property_nodes_temporal_by_type(time_graph, bucket_idx)

            # Update outside node with weather data
            self._update_outside_node_temporal(time_graph, bucket_idx)

            # Update time node with temporal features
            self._update_time_node_temporal(time_graph, bucket_idx)

            hetero_temporal_graphs[bucket_idx] = time_graph

        self.hetero_temporal_graphs = hetero_temporal_graphs
        logger.info(f"Built {len(hetero_temporal_graphs)} temporal heterogeneous graphs")
        
        return hetero_temporal_graphs
    
    def _update_property_nodes_temporal_by_type(
            self,
            hetero_graph: HeteroData,
            bucket_idx: int,
        ) -> None:
        """Update property nodes (separated by type) with measurements for this time bucket."""
        if not hasattr(self, 'device_level_df'):
            raise ValueError("Device-level DataFrame not found. Run build_device_level_df first.")
        
        df = self.device_level_df        
        bucket_df = df[df['bucket_idx'] == bucket_idx]
        cols = self.temporal_prop_features
        
        # Update each property type separately
        for prop_type in self.used_property_types:
            prop_node_type = f'prop_{prop_type}'
            
            # Filter DataFrame for this property type
            prop_bucket_df = bucket_df[bucket_df['property_type'] == prop_type]
            
            # Get the device URIs in the correct order for this property type
            device_uris_ordered = [
                self.reverse_node_mappings[prop_node_type][idx]
                for idx in range(len(self.node_mappings[prop_node_type]))
            ]
            
            # Create feature matrix for this property type
            if not prop_bucket_df.empty:
                # Vectorized feature alignment
                block = (prop_bucket_df
                        .set_index('device_uri_str')[cols]
                        .astype(np.float32))
                aligned = block.reindex(device_uris_ordered)
                
                # Build the tensor
                prop_feat_mat = torch.from_numpy(
                    aligned.to_numpy(dtype=np.float32, copy=False)
                )
            else:
                # No data for this property type in this bucket
                n_nodes = len(self.node_mappings[prop_node_type])
                prop_feat_mat = torch.full(
                    (n_nodes, len(cols)), 
                    float('nan'), 
                    dtype=torch.float32
                )
            
            # Update the graph
            hetero_graph[prop_node_type].x = prop_feat_mat
        
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
            features = row.to_numpy(np.float32)[None, :]
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
            feats = row.to_numpy(np.float32)[None, :]
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
                # temporal features (property types, outside, time) -> deep-copy so we can overwrite
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
            "property_types": self.used_property_types,  # Added for easy access
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
        for node_type in sorted(hetero_graph.node_types):
            if hasattr(hetero_graph[node_type], 'x'):
                n_nodes, n_features = hetero_graph[node_type].x.shape
                logger.info(f"  {node_type:20} : {n_nodes:4} nodes, {n_features:2} features")
        
        # Edge statistics
        for edge_type in sorted(hetero_graph.edge_types):
            if hasattr(hetero_graph[edge_type], 'edge_index'):
                n_edges = hetero_graph[edge_type].edge_index.shape[1]
                src, rel, dst = edge_type
                logger.info(f"  ({src:15}, {rel:20}, {dst:15}) : {n_edges:4} edges")
    
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
        schema.append("=" * 80)
        schema.append("HETEROGENEOUS GRAPH SCHEMA (WITH SEPARATE PROPERTY NODE TYPES)")
        schema.append("=" * 80)
        schema.append("")
        
        # Node types
        schema.append("NODE TYPES:")
        schema.append("-" * 40)
        for node_type in sorted(self.base_hetero_graph.node_types):
            if hasattr(self.base_hetero_graph[node_type], 'x'):
                n_nodes, n_features = self.base_hetero_graph[node_type].x.shape
                schema.append(f"  {node_type:20} : {n_nodes:4} nodes, {n_features:2} features")
        schema.append("")
        
        # Edge types
        schema.append("EDGE TYPES:")
        schema.append("-" * 40)
        for edge_type in sorted(self.base_hetero_graph.edge_types):
            src, rel, dst = edge_type
            if hasattr(self.base_hetero_graph[edge_type], 'edge_index'):
                n_edges = self.base_hetero_graph[edge_type].edge_index.shape[1]
                schema.append(f"  {src:15} --{rel:>22}--> {dst:15} : {n_edges:4} edges")
        schema.append("")
        
        # Feature descriptions
        schema.append("FEATURE DESCRIPTIONS:")
        schema.append("-" * 40)
        schema.append("  room               : Static room attributes (area, windows, etc.)")
        schema.append("  device             : Multi-hot properties measured + one-hot device type")
        schema.append("  prop_Temperature   : Temperature measurements (temporal)")
        schema.append("  prop_CO2Level      : CO2 level measurements (temporal)")
        schema.append("  prop_Humidity      : Humidity measurements (temporal)")
        schema.append("  outside            : Weather data (normalized)")
        schema.append("  time               : Temporal info (day_of_week, hour, is_workday)")
        schema.append("")
        
        # Adjacency info
        schema.append("ADJACENCY INFORMATION:")
        schema.append("-" * 40)
        schema.append("  adjacent_horizontal : Same-floor room connections")
        schema.append("  adjacent_vertical   : Inter-floor room connections (overlapping spaces)")
        schema.append("  contains/contained  : Room-device bidirectional connections")
        schema.append("  measures/measured   : Device-property bidirectional connections (per property type)")
        schema.append("  influences          : Outside-to-room connections (windows)")
        schema.append("  affects             : Time node connections to all node types")
        schema.append("")
        
        # Additional info
        if hasattr(self, 'hetero_temporal_graphs') and self.hetero_temporal_graphs:
            schema.append(f"TEMPORAL SNAPSHOTS: {len(self.hetero_temporal_graphs)}")
            schema.append("")
        
        # Property type breakdown
        if hasattr(self, 'used_property_types'):
            schema.append("PROPERTY TYPES:")
            schema.append("-" * 40)
            for prop_type in self.used_property_types:
                prop_node_type = f'prop_{prop_type}'
                if prop_node_type in self.node_mappings:
                    n_nodes = len(self.node_mappings[prop_node_type])
                    schema.append(f"  {prop_type:15} : {n_nodes:4} measurement nodes")
        
        schema_str = "\n".join(schema)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(schema_str)
            logger.info(f"Saved graph schema to {save_path}")
        
        return schema_str