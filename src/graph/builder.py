from typing import List, Tuple
import pandas as pd
import numpy as np
import networkx as nx
from rdflib import URIRef
import re

from ..core import Room
from .officegraph import OfficeGraph

class OfficeGraphBuilder:
    """Class to build adjacency matrices and networks from OfficeGraph data."""
    
    def __init__(self, office_graph: OfficeGraph):
        """
        Initialize the builder with the OfficeGraph instance.
        
        Args:
            office_graph: The OfficeGraph instance containing the data.
        """
        self.office_graph = office_graph
        # Flag to track if floor plan data has been loaded
        self.floor_plan_loaded = False
        # Map to track room URIs by room number
        self.room_number_to_uri = {}
        # Map to track floor plan URIs to graph URIs
        self.floorplan_uri_to_graph_uri = {}
        
    def _create_room_mappings(self):
        """Create mappings from room numbers to URIs and from floor plan URIs to graph URIs."""
        # Map room numbers to URIs
        for uri, room in self.office_graph.rooms.items():
            if room.room_number:
                self.room_number_to_uri[room.room_number] = uri
        
        # Also create a mapping based on the URI pattern
        for uri in self.office_graph.rooms.keys():
            uri_str = str(uri)
            # Extract room number from URI string
            room_number_match = re.search(r'_(\d+\.\d+)$', uri_str)
            if room_number_match:
                room_number = room_number_match.group(1)
                self.room_number_to_uri[room_number] = uri
        
        print(f"Created mappings for {len(self.room_number_to_uri)} rooms")
    
    def load_floor_plan(self, csv_path: str = "data/OfficeGraph/floor_plan/floor_7.csv") -> None:
        """
        Load floor plan data from CSV and create new Room instances if they don't exist.
        
        Args:
            csv_path: Path to the floor plan CSV file.
        """
        # First, create mappings from room numbers to URIs
        self._create_room_mappings()
        
        # Load CSV file
        floor_plan_df = pd.read_csv(csv_path)
        
        # Create rooms that don't exist
        rooms_created = 0
        rooms_updated = 0
        
        # Process each row in the CSV
        for _, row in floor_plan_df.iterrows():
            # Extract URI and room number
            uri_str = row['URI']
            room_number = row['room_number']
            
            # Create URI reference
            uri = URIRef(uri_str)
            
            # Try to find existing room by exact URI
            room = None
            if uri in self.office_graph.rooms:
                room = self.office_graph.rooms[uri]
                self.floorplan_uri_to_graph_uri[uri] = uri
            # Try to find by room number
            elif room_number in self.room_number_to_uri:
                graph_uri = self.room_number_to_uri[room_number]
                room = self.office_graph.rooms[graph_uri]
                self.floorplan_uri_to_graph_uri[uri] = graph_uri
            # Create new room if not found
            else:
                print(f"Creating new room for {uri_str} with room number {room_number}")
                room = Room(
                    uri=uri,
                    room_number=room_number,
                    is_support_zone=False
                )
                self.office_graph.rooms[uri] = room
                self.room_number_to_uri[room_number] = uri
                self.floorplan_uri_to_graph_uri[uri] = uri
                rooms_created += 1
            
            # Update room attributes from CSV
            room.x_1 = int(row['X_1']) if not pd.isna(row['X_1']) else None
            room.x_2 = int(row['X_2']) if not pd.isna(row['X_2']) else None
            room.y_1 = row['Y_1'] if not pd.isna(row['Y_1']) else None
            room.y_2 = row['Y_2'] if not pd.isna(row['Y_2']) else None
            room.size_approx = float(row['size_approx']) if not pd.isna(row['size_approx']) else None
            room.isRoom = bool(row['isRoom']) if not pd.isna(row['isRoom']) else None
            
            # Process isFacing (comma-separated string to list)
            if not pd.isna(row['isFacing']) and row['isFacing'] != 'none':
                room.isFacing = [direction.strip() for direction in row['isFacing'].split(',')]
            
            # Process adjacent rooms (comma-separated string to list)
            if not pd.isna(row['adj_list']) and row['adj_list'] != '':
                room.adjacent_rooms = [room_id.strip() for room_id in row['adj_list'].split(',')]
            
            rooms_updated += 1
        
        self.floor_plan_loaded = True
        print(f"Loaded floor plan data for {len(floor_plan_df)} rooms")
        print(f"Created {rooms_created} new rooms, updated {rooms_updated} rooms")

    def _ensure_floor_plan_loaded(self) -> None:
        """Ensure that floor plan data has been loaded."""
        if not self.floor_plan_loaded:
            self.load_floor_plan()
    
    def build_room_adjacency(self) -> Tuple[np.ndarray, List[URIRef]]:
        """
        Build the room adjacency matrix using the floor plan data.
        
        Returns:
            A tuple containing:
                - The room adjacency matrix as a numpy array
                - List of room URIs in the same order as the matrix rows/columns
        """
        self._ensure_floor_plan_loaded()
        
        # Get all rooms that have adjacency information
        rooms_with_adj = [uri for uri, room in self.office_graph.rooms.items() 
                          if hasattr(room, 'adjacent_rooms') and room.adjacent_rooms]
        
        # Create a sorted list of all room URIs to serve as indices
        room_uris = sorted(rooms_with_adj)
        
        if not room_uris:
            print("Warning: No rooms with adjacency information found")
            return np.zeros((0, 0)), []
        
        room_indices = {uri: i for i, uri in enumerate(room_uris)}
        
        # Initialize adjacency matrix with zeros
        n_rooms = len(room_uris)
        adjacency = np.zeros((n_rooms, n_rooms), dtype=int)
        
        # Fill in the adjacency matrix based on the adjacent_rooms lists
        for uri in room_uris:
            room = self.office_graph.rooms[uri]
            from_idx = room_indices[uri]
            
            for adj_room_number in room.adjacent_rooms:
                # Skip if the adjacent room number is not in our mapping
                if adj_room_number not in self.room_number_to_uri:
                    continue
                
                adj_uri = self.room_number_to_uri[adj_room_number]
                # Skip if the adjacent room is not in our list of rooms with adjacency
                if adj_uri not in room_indices:
                    continue
                
                to_idx = room_indices[adj_uri]
                # Set bidirectional adjacency
                adjacency[from_idx, to_idx] = 1
                adjacency[to_idx, from_idx] = 1  # Ensure symmetry
        
        return adjacency, room_uris
    
    def build_device_room_adjacency(self) -> Tuple[np.ndarray, List[URIRef], List[URIRef]]:
        """
        Build the device-room adjacency matrix.
        
        Returns:
            A tuple containing:
                - The device-room adjacency matrix as a numpy array
                - List of device URIs in the same order as the matrix rows
                - List of room URIs in the same order as the matrix columns
        """
        self._ensure_floor_plan_loaded()
        
        # Get all devices and rooms
        device_uris = sorted(self.office_graph.devices.keys())
        room_uris = sorted(self.office_graph.rooms.keys())
        
        # Create indices for devices and rooms
        device_indices = {uri: i for i, uri in enumerate(device_uris)}
        room_indices = {uri: i for i, uri in enumerate(room_uris)}
        
        # Initialize adjacency matrix with zeros
        n_devices = len(device_uris)
        n_rooms = len(room_uris)
        adjacency = np.zeros((n_devices, n_rooms), dtype=int)
        
        # Fill in the adjacency matrix based on the device_room_floor_map
        for device_uri, mappings in self.office_graph.device_room_floor_map.items():
            # Skip if the device is not in our list of devices
            if device_uri not in device_indices:
                continue
            
            device_idx = device_indices[device_uri]
            room_uri = URIRef(mappings["room"])
            
            # Skip if the room is not in our list of rooms
            if room_uri not in room_indices:
                continue
            
            room_idx = room_indices[room_uri]
            adjacency[device_idx, room_idx] = 1
        
        return adjacency, device_uris, room_uris
  
    def build_heterogeneous_graph(self) -> nx.MultiDiGraph:
        """
        Build a heterogeneous graph with rooms and devices as nodes.
        
        Returns:
            A NetworkX MultiDiGraph with nodes for rooms and devices,
            and edges representing adjacency and containment.
        """
        # Get adjacency matrices
        room_adj, room_uris = self.build_room_adjacency()
        device_room_adj, device_uris, room_uris_all = self.build_device_room_adjacency()
        
        # Initialize a multi-directed graph
        G = nx.MultiDiGraph()
        
        # Add room nodes with attributes
        for uri in self.office_graph.rooms.keys():
            room = self.office_graph.rooms[uri]
            node_attrs = {
                'type': 'room',
                'room_number': room.room_number,
                'is_support_zone': room.is_support_zone,
            }
            
            # Add floor plan attributes if they exist
            for attr in ['x_1', 'x_2', 'y_1', 'y_2', 'size_approx', 'isRoom', 'isFacing']:
                if hasattr(room, attr) and getattr(room, attr) is not None:
                    node_attrs[attr] = getattr(room, attr)
            
            G.add_node(uri, **node_attrs)
        
        # Add device nodes with attributes
        for uri in device_uris:
            device = self.office_graph.devices[uri]
            G.add_node(uri, 
                       type='device',
                       manufacturer=device.manufacturer,
                       model=device.model,
                       device_type=device.device_type)
            
            # Add edges between devices and rooms
            if uri in self.office_graph.device_room_floor_map:
                room_uri = URIRef(self.office_graph.device_room_floor_map[uri]["room"])
                if room_uri in G:
                    G.add_edge(uri, room_uri, type='located_in')
                    G.add_edge(room_uri, uri, type='contains')
        
        # Add edges between adjacent rooms
        for i, uri1 in enumerate(room_uris):
            for j, uri2 in enumerate(room_uris):
                if i < j and room_adj[i, j] == 1:
                    G.add_edge(uri1, uri2, type='adjacent_to')
                    G.add_edge(uri2, uri1, type='adjacent_to')
        
        return G

    ### Simplifying the graph building ###
    # Heterogenous with Rooms and Devices -> Homogenous with only Rooms
    # Both static and temporal attributes -> Only temporal attributes (later on to be derived from devices)
    def build_simple_homogeneous_graph(self) -> nx.Graph:
        """
        Build a homogeneous graph with only rooms as nodes, aggregating device measurements.
        
        Returns:
            A NetworkX Graph with nodes for rooms, edges representing adjacency,
            and node attributes containing aggregated device measurements.
        """
        self._ensure_floor_plan_loaded()
        
        # Initialize an undirected graph
        G = nx.Graph()
        
        # Add room nodes with only devices as attributes
        for uri in self.office_graph.rooms.keys():
            room = self.office_graph.rooms[uri]

            G.add_node(uri, devices=list(room.devices))
        
        # Get room adjacency information and add edges between adjacent rooms
        room_adj, room_uris = self.build_room_adjacency()
        for i, uri1 in enumerate(room_uris):
            for j, uri2 in enumerate(room_uris):
                if i < j and room_adj[i, j] == 1:
                    G.add_edge(uri1, uri2, weight=1)
                
        return G
