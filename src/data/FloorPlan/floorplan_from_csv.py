from typing import Dict, List, Tuple
import pandas as pd
from rdflib import URIRef
import numpy as np
import os

from ...core import Room

class FloorPlanFromCSV:
    """Class to handle floor plan data loaded from CSV files."""
    
    def __init__(self, csv_path: str = "data/floor_plan/floor_7.csv"):
        """
        Initialize the FloorPlanFromCSV processor.
        
        Args:
            csv_path: Path to the floor plan CSV file.
        """
        # Data storage
        self.floor_plan_df = None        
        # Map to track room URIs by room number
        self.room_number_to_uri = {}
        self.uri_to_room_number = {}
        # Map to track floor plan URIs to graph URIs
        self.floorplan_uri_to_graph_uri = {}
        
        # Load data
        self._load_floor_plan_csv(csv_path)
        # Create room-URI and URI-room mappings
        self._create_room_mappings()

    def _load_floor_plan_csv(self, csv_path: str = "data/floor_plan/floor_7.csv") -> None:
        """
        Load floor plan data from CSV.
        
        Args:
            csv_path: Path to the floor plan CSV file.
        """
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV file
        self.floor_plan_df = pd.read_csv(csv_path)
        print(f"Loaded floor plan data from {csv_path}: {len(self.floor_plan_df)} rooms")
    
    def _create_room_mappings(self):
        """
        Create mappings from room numbers to URIs and vice versa.
        The CSV file must have the two following columns:
        1. "URI"
        2. "room_number"
        """
        # Process each row in the CSV
        for _, row in self.floor_plan_df.iterrows():
            uri_str = row['URI']
            room_number = row['room_number']
            
            # Create URI reference
            uri = URIRef(uri_str)
            
            # Store mappings
            self.room_number_to_uri[room_number] = uri
            self.uri_to_room_number[uri] = room_number
        
        print(f"Created mappings for {len(self.room_number_to_uri)} rooms")
    
    def get_room_data(self, uri: URIRef) -> Dict:
        """
        Get room data for a specific URI.
        
        Args:
            uri: The room URI.
            
        Returns:
            Dictionary with room data from the CSV.
        """
        # Find the row for this URI
        uri_str = str(uri)
        room_row = self.floor_plan_df[self.floor_plan_df['URI'] == uri_str]

        if len(room_row) == 0:
            return {}

        # Convert the row to a dictionary
        return room_row.iloc[0].to_dict()
    
    def get_adjacency_list(self) -> Dict[URIRef, List[URIRef]]:
        """
        Get adjacency list based on the 'adj_list' column in the CSV.
        
        Returns:
            Dictionary mapping room URIs to lists of adjacent room URIs.
        """        
        adjacency = {}
        
        # Process each row in the CSV
        for _, row in self.floor_plan_df.iterrows():
            uri_str = row['URI']
            uri = URIRef(uri_str)
            
            # Process adjacent rooms (comma-separated string to list)
            if not pd.isna(row['adj_list']) and row['adj_list'] != '':
                adj_room_numbers = [room_id.strip() for room_id in row['adj_list'].split(',')]
                adj_uris = []
                
                for adj_room_number in adj_room_numbers:
                    if adj_room_number in self.room_number_to_uri:
                        adj_uris.append(self.room_number_to_uri[adj_room_number])
                
                adjacency[uri] = adj_uris
            else:
                adjacency[uri] = []
        
        return adjacency
    
    def build_adjacency_matrix(self) -> Tuple[np.ndarray, List[URIRef]]:
        """
        Build the room adjacency matrix using the floor plan data.
        
        Returns:
            A tuple containing:
                - The room adjacency matrix as a numpy array
                - List of room URIs in the same order as the matrix rows/columns
        """        
        # Get adjacency list
        adjacency_list = self.get_adjacency_list()
        
        # Create a sorted list of all room URIs to serve as indices
        room_uris = sorted(adjacency_list.keys())
        
        if not room_uris:
            print("Warning: No rooms with adjacency information found")
            return np.zeros((0, 0)), []
        
        room_indices = {uri: i for i, uri in enumerate(room_uris)}
        
        # Initialize adjacency matrix with zeros
        n_rooms = len(room_uris)
        adjacency_matrix = np.zeros((n_rooms, n_rooms), dtype=int)
        
        # Fill in the adjacency matrix based on the adjacency lists
        for uri, adj_uris in adjacency_list.items():
            from_idx = room_indices[uri]
            
            for adj_uri in adj_uris:
                # Skip if the adjacent room is not in our list of rooms
                if adj_uri not in room_indices:
                    continue
                
                to_idx = room_indices[adj_uri]
                # Set bidirectional adjacency
                adjacency_matrix[from_idx, to_idx] = 1
                adjacency_matrix[to_idx, from_idx] = 1  # Ensure symmetry
        
        return adjacency_matrix, room_uris
    
    def get_pseudo_coordinates_for_rooms(self) -> Dict[URIRef, Tuple[int, int, int, int]]:
        """
        Get room coordinates from the CSV data.
        
        Returns:
            Dictionary mapping room URIs to coordinates (x1, y1, x2, y2).
        """        
        coordinates = {}
        
        # Process each row in the CSV
        for _, row in self.floor_plan_df.iterrows():
            uri_str = row['URI']
            uri = URIRef(uri_str)
            
            # Extract coordinates
            x1 = int(row['X_1']) if not pd.isna(row['X_1']) else None
            x2 = int(row['X_2']) if not pd.isna(row['X_2']) else None
            y1 = row['Y_1'] if not pd.isna(row['Y_1']) else None
            y2 = row['Y_2'] if not pd.isna(row['Y_2']) else None
            
            if x1 is not None and x2 is not None and y1 is not None and y2 is not None:
                coordinates[uri] = (x1, y1, x2, y2)
        
        return coordinates
    
    def get_areas_for_rooms(self) -> Dict[URIRef, float]:
        """
        Get room areas from the CSV data.
        
        Returns:
            Dictionary mapping room URIs to areas.
        """        
        areas = {}
        
        # Process each row in the CSV
        for _, row in self.floor_plan_df.iterrows():
            uri_str = row['URI']
            uri = URIRef(uri_str)
            
            # Extract area
            if not pd.isna(row['size_approx']):
                areas[uri] = float(row['size_approx'])
        
        return areas
    
    def get_window_directions_for_rooms(self) -> Dict[URIRef, List[str]]:
        """
        Get room facing directions from the CSV data.
        
        Returns:
            Dictionary mapping room URIs to lists of directions.
        """        
        directions = {}
        
        # Process each row in the CSV
        for _, row in self.floor_plan_df.iterrows():
            uri_str = row['URI']
            uri = URIRef(uri_str)
            
            # Process isFacing (comma-separated string to list)
            if not pd.isna(row['isFacing']) and row['isFacing'] != 'none':
                directions[uri] = [direction.strip() for direction in row['isFacing'].split(',')]
            else:
                directions[uri] = []
        
        return directions

    def update_create_or_remove_room_objects(self, office_graph) -> None:
        """
        Enrich Room objects in the OfficeGraph with data from the floor plan CSV.
        Only keeps rooms that exist in the CSV (removes others).
        
        Args:
            office_graph: The OfficeGraph instance to update.
        """
        rooms_created = 0
        rooms_updated = 0

        csv_uris = set()
        
        for _, row in self.floor_plan_df.iterrows():
            uri = URIRef(row['URI'])
            csv_uris.add(uri)
            room_number = row['room_number']

        # Find or create room
        if uri in office_graph.rooms:
            room = office_graph.rooms[uri]
            self.floorplan_uri_to_graph_uri[uri] = uri
        elif room_number in self.room_number_to_uri:
            graph_uri = self.room_number_to_uri[room_number]
            room = office_graph.rooms.get(graph_uri)
            if room:
                self.floorplan_uri_to_graph_uri[uri] = graph_uri
            else:
                # Create a new room if not found
                room = Room(uri=uri, room_number=room_number)
                office_graph.rooms[uri] = room
                self.floorplan_uri_to_graph_uri[uri] = uri
                rooms_created += 1
        else:
            room = Room(uri=uri, room_number=room_number)
            office_graph.rooms[uri] = room
            self.room_number_to_uri[room_number] = uri
            self.floorplan_uri_to_graph_uri[uri] = uri
            rooms_created += 1

            # Enrich room attributes
            room.x_1 = int(row['X_1']) if not pd.isna(row['X_1']) else None
            room.x_2 = int(row['X_2']) if not pd.isna(row['X_2']) else None
            room.y_1 = row['Y_1'] if not pd.isna(row['Y_1']) else None
            room.y_2 = row['Y_2'] if not pd.isna(row['Y_2']) else None
            room.size_approx = float(row['size_approx']) if not pd.isna(row['size_approx']) else None
            room.isRoom = bool(row['isRoom']) if not pd.isna(row['isRoom']) else None

            if not pd.isna(row['isFacing']) and row['isFacing'] != 'none':
                room.isFacing = [d.strip() for d in row['isFacing'].split(',')]

            if not pd.isna(row['adj_list']) and row['adj_list'] != '':
                room.adjacent_rooms = [r.strip() for r in row['adj_list'].split(',')]

            rooms_updated += 1

        # Remove rooms not present in CSV
        all_graph_uris = set(office_graph.rooms.keys())
        uris_to_remove = all_graph_uris - csv_uris

        for uri in uris_to_remove:
            office_graph.rooms.pop(uri)

        print(f"Updated {rooms_updated} rooms; created {rooms_created} new rooms")
        print(f"Removed {len(uris_to_remove)} rooms not found in CSV")
