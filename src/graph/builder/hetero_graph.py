import sys
import logging
import numpy as np
import torch
from torch_geometric.data import HeteroData
from typing import Dict, List, Any, Optional
from rdflib import URIRef
from datetime import datetime
import math

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class HeteroGraphBuilderMixin:
    """
    Mixin class for building heterogeneous graphs with PyTorch Geometric.
        
    Node types:
    - 'room': Room nodes with static features
    - 'device': Device nodes (intermediary between rooms and properties)
    - 'property': Property nodes with temporal measurements (one per device-property combination)
    - 'outside': Single outside node with weather data
    - 'general': Single general information node (time features)
    
    Edge types:
    - ('room', 'adjacent_horizontal', 'room'): Horizontal room-to-room adjacency (same floor, directed)
    - ('room', 'adjacent_vertical', 'room'): Vertical room-to-room adjacency (between floors, directed)
    - ('room', 'contains', 'device'): Room-to-device connections
    - ('device', 'contained_in', 'room'): Device-to-room connections (reverse)
    - ('device', 'measures', 'property'): Device-to-property connections
    - ('property', 'measured_by', 'device'): Property-to-device connections (reverse)
    - ('outside', 'influences', 'room'): Outside-to-room connections (for rooms with windows, weighted)
    - ('general', 'affects', 'room'): General-to-room connections
    - ('general', 'affects', 'device'): General-to-device connections
    - ('general', 'affects', 'property'): General-to-property connections
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize hetero-specific attributes
        self.base_hetero_graph = None
        self.hetero_temporal_graphs = None
        self.node_mappings = {}
        self.reverse_node_mappings = {}
        
    def build_base_hetero_graph(self) -> HeteroData:
        """
        Build the base heterogeneous graph structure with static features.
        This graph will be duplicated for each time bucket.
        
        Returns:
            HeteroData: Base heterogeneous graph
        """
        # Ensure required components are available
        if not hasattr(self, 'horizontal_adj_matrix') or self.horizontal_adj_matrix is None:
            raise ValueError("Room adjacency matrix not found. Run build_horizontal_adjacency first.")
        if not hasattr(self, 'vertical_adj_matrix') or self.vertical_adj_matrix is None:
            raise ValueError("Vertical adjacency matrix not found. Run build_vertical_adjacency first.")
        if not hasattr(self, 'full_feature_df'):
            raise ValueError("Full feature DataFrame not found. Run build_full_feature_df first.")
        if not hasattr(self, 'static_room_attributes') or not self.static_room_attributes:
            logger.warning("No static room attributes defined. Using default attributes.")
            self.static_room_attributes = ['isProperRoom', 'norm_area_minmax']
        
        # Initialize heterogeneous graph
        hetero_data = HeteroData()
        
        # Build node mappings and features
        self._build_node_mappings()
        self._add_room_nodes(hetero_data)
        self._add_device_nodes(hetero_data)
        self._add_property_nodes(hetero_data)
        self._add_outside_node(hetero_data)
        self._add_general_node(hetero_data)
        
        # Build edges
        self._add_horizontal_room_edges(hetero_data)
        self._add_vertical_room_edges(hetero_data)
        self._add_room_device_edges(hetero_data)
        self._add_device_property_edges(hetero_data)
        self._add_outside_room_edges(hetero_data)
        self._add_general_edges(hetero_data)
        
        self.base_hetero_graph = hetero_data
        logger.info("Built base heterogeneous graph")
        self._log_graph_statistics(hetero_data)
        
        return hetero_data
    
    def _build_node_mappings(self):
        """Build mappings between original URIs/IDs and node indices."""
        self.node_mappings = {
            'room': {},
            'device': {},
            'property': {},
            'outside': {},
            'general': {}
        }
        self.reverse_node_mappings = {
            'room': {},
            'device': {},
            'property': {},
            'outside': {},
            'general': {}
        }
        
        # Room nodes - use the same order as adjacency matrix
        for idx, room_uri in enumerate(self.adj_matrix_room_uris):
            self.node_mappings['room'][room_uri] = idx
            self.reverse_node_mappings['room'][idx] = room_uri
        
        # Device nodes - extract unique devices from the DataFrame
        unique_devices = self.full_feature_df['device_uri'].unique()
        for idx, device_uri in enumerate(unique_devices):
            # Convert string back to URIRef if needed
            if isinstance(device_uri, str) and not isinstance(device_uri, URIRef):
                device_uri_ref = URIRef(device_uri)
            else:
                device_uri_ref = device_uri
            self.node_mappings['device'][device_uri_ref] = idx
            self.reverse_node_mappings['device'][idx] = device_uri_ref
        
        # Property nodes - one node per unique (device, property_type) combination
        property_idx = 0
        self.property_type_lookup = {}
        
        # Get unique device-property combinations from the DataFrame
        unique_combinations = self.full_feature_df[['device_uri', 'property_type']].drop_duplicates()
        
        for _, row in unique_combinations.iterrows():
            device_uri = row['device_uri']
            property_type = row['property_type']
            
            # Convert to URIRef if needed
            if isinstance(device_uri, str) and not isinstance(device_uri, URIRef):
                device_uri = URIRef(device_uri)
            
            # Create key
            device_property_key = (device_uri, property_type)
            self.node_mappings['property'][device_property_key] = property_idx
            self.reverse_node_mappings['property'][property_idx] = device_property_key
            self.property_type_lookup[device_property_key] = property_type
            property_idx += 1
        
        # Outside node (single node)
        self.node_mappings['outside']['outside'] = 0
        self.reverse_node_mappings['outside'][0] = 'outside'
        
        # General node (single node)
        self.node_mappings['general']['general'] = 0
        self.reverse_node_mappings['general'][0] = 'general'
        
        logger.info(f"Created node mappings: {len(self.node_mappings['room'])} rooms, "
                f"{len(self.node_mappings['device'])} devices, "
                f"{len(self.node_mappings['property'])} device-property combinations, "
                f"1 outside, 1 general")
    
    #############################
    # Nodes
    #############################

    def _add_room_nodes(self, hetero_data: HeteroData):
        """Add room nodes with static features."""
        n_rooms = len(self.node_mappings['room'])
        n_features = len(self.static_room_attributes)
        
        # Check if we need normalized areas
        if ('norm_area_minmax' in self.static_room_attributes or 
            'norm_area_prop' in self.static_room_attributes):
            if not hasattr(self, 'norm_areas_minmax') or not self.norm_areas_minmax:
                raise ValueError("Normalized areas not calculated. Run normalize_room_areas first.")
        
        # Initialize room feature matrix
        room_features = torch.zeros(n_rooms, n_features, dtype=torch.float32)
        
        # Fill room features
        for room_uri, room_idx in self.node_mappings['room'].items():
            room = self.office_graph.rooms[room_uri]
            
            for feat_idx, feat_name in enumerate(self.static_room_attributes):
                # Handle special normalized area attributes
                if feat_name == 'norm_area_minmax':
                    # Find the floor and get the normalized area
                    floor_num = self.room_to_floor.get(room_uri)
                    if floor_num is not None and floor_num in self.norm_areas_minmax:
                        if room_uri in self.norm_areas_minmax[floor_num]:
                            room_features[room_idx, feat_idx] = self.norm_areas_minmax[floor_num][room_uri]
                elif feat_name == 'norm_area_prop':
                    # Find the floor and get the proportional area
                    floor_num = self.room_to_floor.get(room_uri)
                    if floor_num is not None and floor_num in self.norm_areas_prop:
                        if room_uri in self.norm_areas_prop[floor_num]:
                            room_features[room_idx, feat_idx] = self.norm_areas_prop[floor_num][room_uri]
                # Handle nested attributes
                elif '.' in feat_name:
                    parts = feat_name.split('.')
                    if len(parts) == 2:
                        container, key = parts
                        if hasattr(room, container) and isinstance(getattr(room, container), dict):
                            container_dict = getattr(room, container)
                            if key in container_dict:
                                room_features[room_idx, feat_idx] = container_dict[key]
                # Handle regular attributes
                elif hasattr(room, feat_name):
                    attr_value = getattr(room, feat_name)
                    if isinstance(attr_value, bool):
                        room_features[room_idx, feat_idx] = float(attr_value)
                    elif isinstance(attr_value, (int, float)):
                        room_features[room_idx, feat_idx] = float(attr_value)
        
        hetero_data['room'].x = room_features
        logger.info(f"Added {n_rooms} room nodes with {n_features} features")
    
    def _add_device_nodes(self, hetero_data: HeteroData):
        """Add device nodes with features including one-hot device type encoding."""
        n_devices = len(self.node_mappings['device'])
        
        # First pass: collect all unique device types
        device_types = set()
        for device_uri in self.node_mappings['device'].keys():
            device = self.office_graph.devices[device_uri]
            if hasattr(device, 'device_type') and device.device_type:
                device_types.add(device.device_type)
        
        device_types = sorted(list(device_types))  # Sort for consistent ordering
        n_device_types = len(device_types)
        
        # Create device type to index mapping
        self.device_type_to_idx = {dtype: idx for idx, dtype in enumerate(device_types)}
        self.idx_to_device_type = {idx: dtype for dtype, idx in self.device_type_to_idx.items()}
        
        logger.info(f"Found {n_device_types} unique device types: {device_types}")
        
        # Features: [num_properties] + [device_type_one_hot_encoding]
        n_features = 1 + n_device_types
        device_features = torch.zeros(n_devices, n_features, dtype=torch.float32)
        
        for device_uri, device_idx in self.node_mappings['device'].items():
            device = self.office_graph.devices[device_uri]
            
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
    
    def _add_property_nodes(self, hetero_data: HeteroData):
        """Add property nodes with one-hot encoded property type features."""
        n_properties = len(self.node_mappings['property'])
        
        # Get unique property types from the DataFrame
        property_types = sorted(self.full_feature_df['property_type'].unique())
        n_property_types = len(property_types)
        
        # Create property type to index mapping
        self.property_type_to_idx = {prop_type: idx for idx, prop_type in enumerate(property_types)}
        self.idx_to_property_type = {idx: prop_type for prop_type, idx in self.property_type_to_idx.items()}
        
        # Store for later use in temporal updates
        self.actual_property_types = property_types
        
        logger.info(f"Found {n_property_types} unique property types: {property_types}")
        
        # Features: [property_type_one_hot_encoding]
        n_features = n_property_types
        property_features = torch.zeros(n_properties, n_features, dtype=torch.float32)
        
        # Fill in the one-hot encoding
        for (device_uri, prop_type), prop_idx in self.node_mappings['property'].items():
            type_idx = self.property_type_to_idx[prop_type]
            property_features[prop_idx, type_idx] = 1.0

        hetero_data['property'].x = property_features
        logger.info(f"Added {n_properties} property nodes with {n_property_types} features")
        logger.info(f"Property features: one-hot[{', '.join(property_types)}]")
    
    def _add_outside_node(self, hetero_data: HeteroData):
        """Add single outside node for weather data."""
        # Initialize with placeholder for weather features
        # Will be filled with actual weather data per time bucket

        if not hasattr(self, 'weather_features_norm_') and self.weather_features_norm_:
            raise ValueError("Weather features unavailable.")
        
        sample = next(iter(self.weather_features_norm_.values()))
        num_weather_feats = len(sample)  # number of keys in this dict

        outside_features = torch.zeros(1, num_weather_feats, dtype=torch.float32)
        hetero_data['outside'].x = outside_features
        logger.info(f"Added 1 outside node with placeholder dim={num_weather_feats} "
                    "(actual features will be filled per time bucket)")
    
    def _add_general_node(self, hetero_data: HeteroData):
        """Add single general information node for temporal features."""
        # Initialize with placeholder features - will be filled per time bucket
        # Features:
        ## Cyclical dims: day of week (7), hour of day (24)
        ## → 2 dims each: sin & cos
        ##
        ## Plus optional flags: is_weekend, is_workday
        D = 2 + 2 + 2  # sin_day, cos_day, sin_hour, cos_hour, is_weekend, is_workday
        general_features = torch.zeros(1, D, dtype=torch.float32)

        hetero_data['general'].x = general_features
        logger.info(f"Added 1 general node with {D} temporal features")
    
    #############################
    # Edges
    #############################

    def _add_horizontal_room_edges(self, hetero_data: HeteroData):
        """Add directed horizontal room-to-room edges based on horizontal adjacency matrix."""
        # Get edges from horizontal adjacency matrix (directed, non-symmetric)
        edge_indices = []
        edge_weights = []
        
        for i in range(self.horizontal_adj_matrix.shape[0]):
            for j in range(self.horizontal_adj_matrix.shape[1]):
                weight = self.horizontal_adj_matrix[i, j]
                if weight > 0:
                    edge_indices.append([i, j])
                    edge_weights.append(weight)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
            
            hetero_data['room', 'adjacent_horizontal', 'room'].edge_index = edge_index
            hetero_data['room', 'adjacent_horizontal', 'room'].edge_attr = edge_attr
            
            logger.info(f"Added {len(edge_indices)} horizontal room-to-room edges")
        else:
            logger.warning("No horizontal room-to-room edges found")
    
    def _add_vertical_room_edges(self, hetero_data: HeteroData):
        """Add directed vertical room-to-room edges based on vertical adjacency matrix."""
        # Get edges from vertical adjacency matrix (directed, non-symmetric)
        edge_indices = []
        edge_weights = []
        
        for i in range(self.vertical_adj_matrix.shape[0]):
            for j in range(self.vertical_adj_matrix.shape[1]):
                weight = self.vertical_adj_matrix[i, j]
                if weight > 0:
                    edge_indices.append([i, j])
                    edge_weights.append(weight)
        
        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
            
            hetero_data['room', 'adjacent_vertical', 'room'].edge_index = edge_index
            hetero_data['room', 'adjacent_vertical', 'room'].edge_attr = edge_attr
            
            logger.info(f"Added {len(edge_indices)} vertical room-to-room edges")
        else:
            logger.warning("No vertical room-to-room edges found")
    
    def _add_room_device_edges(self, hetero_data: HeteroData):
        """Add bidirectional edges between rooms and devices."""
        room_to_device_edges = []
        device_to_room_edges = []
        
        for room_uri, room_idx in self.node_mappings['room'].items():
            room = self.office_graph.rooms[room_uri]
            
            for device_uri in room.devices:
                if device_uri in self.node_mappings['device']:
                    device_idx = self.node_mappings['device'][device_uri]
                    room_to_device_edges.append([room_idx, device_idx])
                    device_to_room_edges.append([device_idx, room_idx])
        
        # Add room -> device edges
        if room_to_device_edges:
            edge_index = torch.tensor(room_to_device_edges, dtype=torch.long).t().contiguous()
            hetero_data['room', 'contains', 'device'].edge_index = edge_index
            logger.info(f"Added {len(room_to_device_edges)} room-to-device edges")
        
        # Add device -> room edges
        if device_to_room_edges:
            edge_index = torch.tensor(device_to_room_edges, dtype=torch.long).t().contiguous()
            hetero_data['device', 'contained_in', 'room'].edge_index = edge_index
            logger.info(f"Added {len(device_to_room_edges)} device-to-room edges")
    
    def _add_device_property_edges(self, hetero_data: HeteroData):
        """Add bidirectional edges between devices and properties."""
        device_to_property_edges = []
        property_to_device_edges = []
        
        # Use the mappings to create edges
        for (device_uri, property_type), prop_idx in self.node_mappings['property'].items():
            if device_uri in self.node_mappings['device']:
                device_idx = self.node_mappings['device'][device_uri]
                device_to_property_edges.append([device_idx, prop_idx])
                property_to_device_edges.append([prop_idx, device_idx])
        
        # Add device -> property edges
        if device_to_property_edges:
            edge_index = torch.tensor(device_to_property_edges, dtype=torch.long).t().contiguous()
            hetero_data['device', 'measures', 'property'].edge_index = edge_index
            logger.info(f"Added {len(device_to_property_edges)} device-to-property edges")
        
        # Add property -> device edges
        if property_to_device_edges:
            edge_index = torch.tensor(property_to_device_edges, dtype=torch.long).t().contiguous()
            hetero_data['property', 'measured_by', 'device'].edge_index = edge_index
            logger.info(f"Added {len(property_to_device_edges)} property-to-device edges")
                
    def _add_outside_room_edges(self, hetero_data: HeteroData):
        """Add weighted edges from outside node to rooms (no reverse edges)."""
        # Check if outside adjacency has been calculated
        if not hasattr(self, 'combined_outside_adj') or self.combined_outside_adj is None:
            raise ValueError("Outside adjacency not found. Run build_outside_adjacency first.")
        
        outside_to_room_edges = []
        edge_weights = []
        
        outside_idx = 0  # Single outside node
        
        # Use the calculated outside adjacency weights
        for room_uri, room_idx in self.node_mappings['room'].items():
            # Find the corresponding index in the adjacency matrix order
            adj_matrix_idx = self.adj_matrix_room_uris.index(room_uri)
            weight = self.combined_outside_adj[adj_matrix_idx]
            
            if weight > 0:  # Only add edges with positive weights
                outside_to_room_edges.append([outside_idx, room_idx])
                edge_weights.append(weight)
        
        # Add outside -> room edges with weights (no reverse edges)
        if outside_to_room_edges:
            edge_index = torch.tensor(outside_to_room_edges, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
            
            hetero_data['outside', 'influences', 'room'].edge_index = edge_index
            hetero_data['outside', 'influences', 'room'].edge_attr = edge_attr
            logger.info(f"Added {len(outside_to_room_edges)} weighted outside-to-room edges")
    
    def _add_general_edges(self, hetero_data: HeteroData):
        """Add edges from general node to all other nodes."""
        general_idx = 0  # Single general node
        
        # General -> room edges
        general_to_room_edges = [[general_idx, room_idx] for room_idx in range(len(self.node_mappings['room']))]
        
        # General -> device edges
        general_to_device_edges = [[general_idx, device_idx] for device_idx in range(len(self.node_mappings['device']))]
        
        # General -> property edges
        general_to_property_edges = [[general_idx, prop_idx] for prop_idx in range(len(self.node_mappings['property']))]
        
        # Add edges
        if general_to_room_edges:
            edge_index = torch.tensor(general_to_room_edges, dtype=torch.long).t().contiguous()
            hetero_data['general', 'affects', 'room'].edge_index = edge_index
            logger.info(f"Added {len(general_to_room_edges)} general-to-room edges")
        
        if general_to_device_edges:
            edge_index = torch.tensor(general_to_device_edges, dtype=torch.long).t().contiguous()
            hetero_data['general', 'affects', 'device'].edge_index = edge_index
            logger.info(f"Added {len(general_to_device_edges)} general-to-device edges")
        
        if general_to_property_edges:
            edge_index = torch.tensor(general_to_property_edges, dtype=torch.long).t().contiguous()
            hetero_data['general', 'affects', 'property'].edge_index = edge_index
            logger.info(f"Added {len(general_to_property_edges)} general-to-property edges")
    
    #############################
    # Graph-building
    #############################

    def build_hetero_temporal_graphs(self) -> Dict[int, HeteroData]:
        """
        Build temporal heterogeneous graphs for each time bucket.
            
        Returns:
            Dictionary mapping time bucket indices to HeteroData graphs
        """
        if self.base_hetero_graph is None:
            logger.info("Building base heterogeneous graph...")
            self.build_base_hetero_graph()
        
        if not hasattr(self, 'time_buckets') or not self.time_buckets:
            raise ValueError("Time buckets not initialized. Call initialize_time_parameters first.")
        
        if not hasattr(self, 'full_feature_df'):
            raise ValueError("Full feature DataFrame not found. Run build_full_feature_df first.")
        
        # Determine feature columns
        feature_cols = [c for c in self.full_feature_df.columns
                        if c not in ('device_uri','property_type','bucket_idx')]
        
        # Make a copy and fill NaNs with 0.0
        df = self.full_feature_df.copy()
        df[feature_cols] = df[feature_cols].fillna(0.0)
        
        hetero_temporal_graphs = {}
        
        total_buckets = len(self.time_buckets)
        log_interval = max(1, total_buckets // 20)

        hetero_temporal_graphs = {}

        for bucket_idx, (bucket_start, bucket_end) in enumerate(self.time_buckets):
            if bucket_idx % log_interval == 0 or bucket_idx == total_buckets - 1:
                logger.info(f"Building temporal graph {bucket_idx + 1}/{total_buckets}")
            
            # Create a deep copy of the base graph
            time_graph = self._copy_hetero_graph(self.base_hetero_graph)

            # Update property nodes with temporal measurements from the DataFrame
            self._update_property_nodes_from_df(time_graph, bucket_idx, df, feature_cols)

            # Update outside node with weather data
            self._update_outside_node_temporal(time_graph, bucket_idx)

            # Update general node with temporal features
            self._update_general_node_temporal(time_graph, bucket_start, bucket_end)

            hetero_temporal_graphs[bucket_idx] = time_graph

        self.hetero_temporal_graphs = hetero_temporal_graphs
        logger.info(f"Built {len(hetero_temporal_graphs)} temporal heterogeneous graphs")
        
        return hetero_temporal_graphs
    
    def _update_property_nodes_from_df(self, hetero_graph: HeteroData, 
                                     bucket_idx: int,
                                     df,
                                     feature_cols: List[str]):
        """Update property nodes with measurements from the DataFrame for this time bucket."""
        n_properties = len(self.node_mappings['property'])
        
        # Get the existing static features
        static_features = hetero_graph['property'].x.clone()
        
        # Get data for this bucket
        bucket_data = df[df['bucket_idx'] == bucket_idx]
        
        # Initialize temporal features
        n_temporal_features = len(feature_cols)
        temporal_features = torch.zeros(n_properties, n_temporal_features, dtype=torch.float32)
        
        # Fill temporal features
        for (device_uri, property_type), prop_idx in self.node_mappings['property'].items():
            # Convert device_uri to string for DataFrame lookup
            device_uri_str = str(device_uri)
            
            # Find the row for this device-property combination
            mask = (bucket_data['device_uri'] == device_uri_str) & (bucket_data['property_type'] == property_type)
            rows = bucket_data[mask]
            
            if len(rows) > 0:
                row = rows.iloc[0]
                
                # Fill feature values
                for feat_idx, feat_name in enumerate(feature_cols):
                    temporal_features[prop_idx, feat_idx] = row[feat_name]
                
        # Concatenate static and temporal features
        hetero_graph['property'].x = torch.cat([static_features, temporal_features], dim=1)
                    
    def _update_outside_node_temporal(self, hetero_graph: HeteroData, bucket_idx: int):
        """Update outside node with weather data for this time bucket."""
        # Must have preloaded & normalized weather
        if not hasattr(self, 'weather_features_norm_') or not self.weather_features_norm_:
            raise ValueError("Weather features unavailable.")

        # Define the base weather features (same order used in _add_outside_node)
        weather_feature_names = [
            'temperature_2m',
            'relative_humidity_2m',
            'precipitation',
            'wind_speed_10m',
            'wind_speed_80m',
            'cloud_cover',
            'wind_direction_10m_sin',
            'wind_direction_10m_cos',
            'wind_direction_80m_sin',
            'wind_direction_80m_cos'
        ]

        # Include any one-hot weather_code columns --> not doing this
        sample = next(iter(self.weather_features_norm_.values()))
        # wc_features = sorted([k for k in sample.keys() if k.startswith('wc_')])
        # all_feature_names = weather_feature_names + wc_features
        all_feature_names = weather_feature_names

        if bucket_idx in self.weather_features_norm_:
            weather_data = self.weather_features_norm_[bucket_idx]
            # Build the feature vector in the consistent order
            weather_features = [weather_data.get(name, 0.0) for name in all_feature_names]
            hetero_graph['outside'].x = torch.tensor([weather_features], dtype=torch.float32)
        else:
            logger.warning(f"No weather data available for bucket {bucket_idx}; using zeros")
            # Fallback: zeros of the correct dimensionality
            hetero_graph['outside'].x = torch.zeros(
                1,
                len(all_feature_names),
                dtype=torch.float32
            )

    def _update_general_node_temporal(self, hetero_graph: HeteroData, 
                                    bucket_start: datetime, bucket_end: datetime):
        """
        Update general node with temporal features for this time bucket,
        using sine‐cosine encoding for cyclic features.
        """
        general_features = torch.zeros_like(hetero_graph['general'].x)

        ts = bucket_start
        
        # Day of week in {0..6}
        dow = ts.weekday()
        general_features[0, 0] = math.sin(2 * math.pi * dow / 7)
        general_features[0, 1] = math.cos(2 * math.pi * dow / 7)
        
        # Hour of day ∈ {0..23}
        hr = ts.hour
        general_features[0, 2] = math.sin(2 * math.pi * hr / 24)
        general_features[0, 3] = math.cos(2 * math.pi * hr / 24)

        # Weekend / workday flags
        general_features[0, 4] = 1.0 if dow >= 5 else 0.0
        general_features[0, 5] = 1.0 if dow < 5 else 0.0

        hetero_graph['general'].x = general_features
        
    def _copy_hetero_graph(self, hetero_graph: HeteroData) -> HeteroData:
        """Create a shallow/deep hybrid copy: static nodes share memory, temporal nodes are cloned."""
        new_graph = HeteroData()
        
        # --- Nodes ---
        for node_type in hetero_graph.node_types:
            x = hetero_graph[node_type].x
            if node_type in ('room', 'device'):
                # static features → share the same tensor
                new_graph[node_type].x = x
            else:
                # temporal features → deep-copy so we can overwrite
                new_graph[node_type].x = x.clone()

        # --- Edges ---
        for edge_type in hetero_graph.edge_types:
            data = hetero_graph[edge_type]
            # edge_index and edge_attr never change → shallow copy
            new_graph[edge_type].edge_index = data.edge_index
            if hasattr(data, 'edge_attr'):
                new_graph[edge_type].edge_attr = data.edge_attr

        return new_graph

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
    
    def prepare_hetero_stgcn_input(self) -> Dict[str, Any]:
        """
        Prepare all necessary inputs for a heterogeneous STGCN model.
                    
        Returns:
            Dictionary containing all hetero STGCN inputs
        """
        if not hasattr(self, 'hetero_temporal_graphs') or not self.hetero_temporal_graphs:
            raise ValueError("Temporal hetero graphs not available. Run build_hetero_temporal_graphs first.")
        
        # Package everything into a dictionary
        hetero_stgcn_input = {
            # For room-to-room adjacencies (separated by type):
            "horizontal_adjacency_matrix": self.horizontal_adj_matrix,
            "vertical_adj_matrix": self.vertical_adj_matrix,
            "room_to_room_adj_matrix": self.room_to_room_adj_matrix,
            "dynamic_adjacencies": self.masked_adjacencies,
            # Hetero graph:
            "base_graph": self.base_hetero_graph,
            "temporal_graphs": self.hetero_temporal_graphs,
            "node_mappings": self.node_mappings,
            "reverse_node_mappings": self.reverse_node_mappings,
            "time_indices": list(range(len(self.time_buckets))),
            "time_buckets": self.time_buckets,
            # Indices for train/val/test splits:
            "train_idx": self.train_indices,
            "val_idx": self.val_indices,
            "test_idx": self.test_indices,
            "blocks": self.blocks,
            # Labels and values (y):
            "workhour_labels": self.get_classification_labels(),
            "consumption_values": self.get_forecasting_values()
        }
        
        logger.info("Heterogeneous STGCN input preparation complete with horizontal and vertical adjacencies")
        return hetero_stgcn_input
    
    def convert_hetero_to_torch_tensors(self, hetero_input: Dict[str, Any], device: str = "cpu") -> Dict[str, Any]:
        """
        Convert heterogeneous data to PyTorch tensors and move to specified device.
        
        Args:
            hetero_input: Dictionary output from prepare_hetero_stgcn_input
            device: PyTorch device to move tensors to
            
        Returns:
            Dictionary with tensors moved to specified device
        """
        torch_input = {}
        
        # Adjacency matrices
        torch_input["horizontal_adjacency_matrix"] = torch.tensor(hetero_input["horizontal_adjacency_matrix"], 
                                                                dtype=torch.float32, 
                                                                device=device)
        
        torch_input["vertical_adj_matrix"] = torch.tensor(hetero_input["vertical_adj_matrix"], 
                                                              dtype=torch.float32, 
                                                              device=device)
        
        torch_input["room_to_room_adj_matrix"] = torch.tensor(hetero_input["room_to_room_adj_matrix"], 
                                                              dtype=torch.float32, 
                                                              device=device)
        
        torch_input["dynamic_adjacencies"] = {}
        for step, masked_adj in hetero_input["dynamic_adjacencies"].items():
            torch_input["dynamic_adjacencies"][step] = torch.tensor(masked_adj, 
                                                                  dtype=torch.float32,
                                                                  device=device)

        # Move base graph to device
        base_graph = hetero_input["base_graph"]
        base_graph_device = self._move_hetero_to_device(base_graph, device)
        torch_input["base_graph"] = base_graph_device
        
        # Move temporal graphs to device
        temporal_graphs_device = {}
        for time_idx, graph in hetero_input["temporal_graphs"].items():
            temporal_graphs_device[time_idx] = self._move_hetero_to_device(graph, device)
        torch_input["temporal_graphs"] = temporal_graphs_device
        
        # Convert labels and values to tensors
        torch_input["workhour_labels"] = torch.tensor(hetero_input["workhour_labels"], 
                                                    dtype=torch.long,
                                                    device=device)
        
        # Convert consumption values
        consumption = np.zeros(len(hetero_input["time_indices"]))
        for idx, value in hetero_input["consumption_values"].items():
            consumption[idx] = value
        torch_input["consumption_values"] = torch.tensor(consumption,
                                                       dtype=torch.float32,
                                                       device=device)
        
        # Copy non-tensor data
        torch_input["node_mappings"] = hetero_input["node_mappings"]
        torch_input["reverse_node_mappings"] = hetero_input["reverse_node_mappings"]
        torch_input["time_indices"] = hetero_input["time_indices"]
        torch_input["time_buckets"] = hetero_input["time_buckets"]
        torch_input["train_idx"] = hetero_input["train_idx"]
        torch_input["val_idx"] = hetero_input["val_idx"]
        torch_input["test_idx"] = hetero_input["test_idx"]
        torch_input["blocks"] = hetero_input["blocks"]
        
        logger.info(f"Converted heterogeneous data to PyTorch tensors on device: {device}")
        return torch_input
    
    def _move_hetero_to_device(self, hetero_graph: HeteroData, device: str) -> HeteroData:
        """Move a heterogeneous graph to the specified device."""
        new_graph = HeteroData()
        
        # Move node features
        for node_type in hetero_graph.node_types:
            if hasattr(hetero_graph[node_type], 'x'):
                new_graph[node_type].x = hetero_graph[node_type].x.to(device)
        
        # Move edge indices and attributes
        for edge_type in hetero_graph.edge_types:
            if hasattr(hetero_graph[edge_type], 'edge_index'):
                new_graph[edge_type].edge_index = hetero_graph[edge_type].edge_index.to(device)
            if hasattr(hetero_graph[edge_type], 'edge_attr'):
                new_graph[edge_type].edge_attr = hetero_graph[edge_type].edge_attr.to(device)
        
        return new_graph
    
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
        schema.append("  general features : Temporal info (day_of_week, hour, is_weekend, is_workday)")
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
    
    def get_adjacency_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the adjacency matrices for analysis and debugging.
        
        Returns:
            Dictionary containing adjacency statistics
        """
        stats = {}
        
        # Horizontal adjacency stats
        if hasattr(self, 'horizontal_adj_matrix') and self.horizontal_adj_matrix is not None:
            h_adj = self.horizontal_adj_matrix
            stats['horizontal'] = {
                'shape': h_adj.shape,
                'total_edges': int((h_adj > 0).sum()),
                'density': float((h_adj > 0).sum()) / (h_adj.shape[0] * h_adj.shape[1]),
                'mean_weight': float(h_adj[h_adj > 0].mean()) if (h_adj > 0).any() else 0.0,
                'max_weight': float(h_adj.max()),
                'min_positive_weight': float(h_adj[h_adj > 0].min()) if (h_adj > 0).any() else 0.0
            }
        
        # Vertical adjacency stats
        if hasattr(self, 'vertical_adj_matrix') and self.vertical_adj_matrix is not None:
            v_adj = self.vertical_adj_matrix
            stats['vertical'] = {
                'shape': v_adj.shape,
                'total_edges': int((v_adj > 0).sum()),
                'density': float((v_adj > 0).sum()) / (v_adj.shape[0] * v_adj.shape[1]),
                'mean_weight': float(v_adj[v_adj > 0].mean()) if (v_adj > 0).any() else 0.0,
                'max_weight': float(v_adj.max()),
                'min_positive_weight': float(v_adj[v_adj > 0].min()) if (v_adj > 0).any() else 0.0
            }
        
        # Combined stats
        if (hasattr(self, 'horizontal_adj_matrix') and self.horizontal_adj_matrix is not None and
            hasattr(self, 'vertical_adj_matrix') and self.vertical_adj_matrix is not None):
            combined = self.horizontal_adj_matrix + self.vertical_adj_matrix
            stats['combined'] = {
                'shape': combined.shape,
                'total_edges': int((combined > 0).sum()),
                'density': float((combined > 0).sum()) / (combined.shape[0] * combined.shape[1]),
                'mean_weight': float(combined[combined > 0].mean()) if (combined > 0).any() else 0.0,
                'max_weight': float(combined.max()),
                'overlap_edges': int(((self.horizontal_adj_matrix > 0) & (self.vertical_adj_matrix > 0)).sum())
            }
        
        # Floor distribution if available
        if hasattr(self, 'room_to_floor') and hasattr(self, 'floor_to_rooms'):
            floor_stats = {}
            for floor_num, room_list in self.floor_to_rooms.items():
                floor_stats[floor_num] = len(room_list)
            stats['floor_distribution'] = floor_stats
        
        return stats

    def log_adjacency_statistics(self):
        """Log adjacency statistics for debugging and analysis."""
        stats = self.get_adjacency_statistics()
        
        logger.info("=" * 50)
        logger.info("ADJACENCY MATRIX STATISTICS")
        logger.info("=" * 50)
        
        for adj_type, adj_stats in stats.items():
            if adj_type == 'floor_distribution':
                logger.info(f"\nFloor Distribution:")
                for floor, count in adj_stats.items():
                    logger.info(f"  Floor {floor}: {count} rooms")
            else:
                logger.info(f"\n{adj_type.upper()} Adjacency:")
                for key, value in adj_stats.items():
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
        
        logger.info("=" * 50)
