import networkx as nx
from rdflib import URIRef


class OfficeGraphBuilder:
    """Class to build adjacency matrices and networks from OfficeGraph data."""
    
    def __init__(self, office_graph):
        """
        Initialize the builder with the OfficeGraph instance.
        
        Args:
            office_graph: The OfficeGraph instance containing the data.
        """
        self.office_graph = office_graph
        self.room_to_room_adj_matrix = None
        self.room_uris = None
        self.csv_floorplan = None
        self.polygons_floorplan = None
    




    ##### Initialization #####
    def initialize_floorplan_from_CSV(self, csv_path: str = "data/floor_plan/floor_7.csv"):
        """Initialize floor plan from CSV file and update office graph."""
        from ..data.FloorPlan.floorplan_from_csv import FloorPlanFromCSV
        try:
            self.csv_floorplan = FloorPlanFromCSV(csv_path)
            print(f"Successfully initialized floor plan from {csv_path}")
        except Exception as e:
            print(f"Error initializing floor plan from CSV: {e}")
            raise 
      
    def initialize_floorplan_from_Polygons(self, polygons_dict_path: str = "data/floor_plan/room_polygons.py"):
        from ..data.FloorPlan.floorplan_from_polygons import FloorPlanFromPolygons
        try:
            self.polygons_floorplan = FloorPlanFromPolygons(polygons_dict_path)
            print(f"Successfully initialized floor plan from {polygons_dict_path}")
        except Exception as e:
            print(f"Error initializing floor plan from CSV: {e}")
            raise 





    ##### Building Adjacency Matrices #####
    def build_room_to_room_adjacency(self, using="csv", kind=None):
        """
        Build the room-to-room adjacency matrix and store it in class attributes.
        
        This method populates self.room_to_room_adj_matrix and self.room_uris
        with the adjacency matrix and corresponding room URIs.
        
        Args:
            using: Source of adjacency data ('csv' or 'polygons')
            kind: Kind of adjacency when using polygon data. Options:
                - 'binary': Basic binary adjacency based on proximity and boundary length
                - 'distance': Weighted adjacency based on distance between centroids
                - 'boundary': Weighted adjacency based on shared boundary length
        """
        if using == "csv":
            if self.csv_floorplan is None:
                self.initialize_floorplan_from_CSV()
            self.room_to_room_adj_matrix, self.room_uris = self.csv_floorplan.build_adjacency_matrix()
            
        elif using == "polygons":
            if kind == None:
                raise ValueError("If building adjacency from polygon data, provide the kind of adjacency matrix.")

            if self.polygons_floorplan is None:
                self.initialize_floorplan_from_Polygons()
            
            if kind == "binary":
                # Get binary adjacency and convert to matrix format
                self.polygons_floorplan.calculate_adjacency()
                adj_matrix = self.polygons_floorplan.generate_adjacency_matrix()
                
                # Extract matrix values and room URIs
                self.room_to_room_adj_matrix = adj_matrix.values
                self.room_uris = [URIRef(room) for room in adj_matrix.index]
                
            elif kind == "distance":
                # Get distance-based weighted adjacency
                adj_matrix = self.polygons_floorplan.calculate_weighted_adjacency_distance()
                
                # Extract matrix values and room URIs
                self.room_to_room_adj_matrix = adj_matrix.values
                self.room_uris = [URIRef(room) for room in adj_matrix.index]
                
            elif kind == "boundary":
                # Get boundary-based weighted adjacency
                adj_matrix = self.polygons_floorplan.calculate_weighted_adjacency_boundary()
                
                # Extract matrix values and room URIs
                self.room_to_room_adj_matrix = adj_matrix.values
                self.room_uris = [URIRef(room) for room in adj_matrix.index]
                
            else:
                raise ValueError(f"Unknown adjacency kind: {kind}. Use 'binary', 'distance', or 'boundary'.")
  




    ##### Building Graphs #####
    # Simplified the graph building:
    # - Heterogenous with Rooms and Devices -> Homogenous with only Rooms
    # - Both static and temporal attributes -> Only temporal attributes (later on to be derived from devices)
    def build_simple_homogeneous_graph(self, weighted: bool = False) -> nx.Graph:
        """
        Build a homogeneous graph with only rooms as nodes, aggregating device measurements.
        
        Args:
            weighted: if True, use the actual values in `self.room_to_room_adj_matrix`
                      as edge weights; otherwise every edge gets weight=1.
        Returns:
            A NetworkX Graph where any non-zero adjacency entry becomes an edge,
            with `weight` either 1 or the original matrix value.
        """
        # Check if adjacency matrix is available
        if self.room_to_room_adj_matrix is None or self.room_uris is None:
            raise ValueError("Room adjacency matrix or Room URIs not found. Run build_room_to_room_adjacency first.")

        # Initialize an undirected graph
        G = nx.Graph()
        
        # Add room nodes with only devices as attributes
        for uri in self.office_graph.rooms.keys():
            room = self.office_graph.rooms[uri]

            G.add_node(uri, devices=list(room.devices))
        
        # Get room adjacency information and add edges between adjacent rooms
        mat = self.room_to_room_adj_matrix
        for i in range(len(self.room_uris)):
            for j in range(i+1, len(self.room_uris)):
                w = mat[i, j]
                if w != 0:
                    G.add_edge(self.room_uris[i],
                               self.room_uris[j],
                               weight=(w if weighted else 1))
                
        return G
