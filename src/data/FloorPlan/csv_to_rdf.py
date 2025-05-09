from rdflib import Graph, Namespace, URIRef, Literal, RDF
from rdflib.namespace import RDFS, XSD
import pandas as pd
import math

from ...core import Building

class RoomEnrichmentViaCSV:
    """
    Read a CSV with columns 'URI', 'isFacing', 'isRoom' and add corresponding triples
    into an existing RDF Graph using a custom namespace for the new predicates.
    """
    def __init__(self,
                 building: Building,
                 csv_path: str = "data/topology/VideoLab_floor7.csv"):
        self.csv_path = csv_path
        self.building = building
        self.graph = Graph()
        
        # Define namespaces
        self.IC = Namespace("https://interconnectproject.eu/example/")
        self.S4BLDG = Namespace("https://saref.etsi.org/saref4bldg/")
        self.EX = Namespace("https://example.org/")
        self.BOT = Namespace("https://w3id.org/bot#")
        self.EXONT = Namespace("https://example.org/ontology#")
        
        # Bind namespaces to prefixes
        self.graph.bind("ic", self.IC)
        self.graph.bind("s4bldg", self.S4BLDG)
        self.graph.bind("ex", self.EX)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
        self.graph.bind("bot", self.BOT)
        self.graph.bind("ex-ont", self.EXONT)
        
        # Mapping building-relative directions to actual headings
        self.direction_heading_map = {
            "front": self.building.heading,
            "right": (self.building.heading + 90) % 360,
            "back": (self.building.heading + 180) % 360,
            "left": (self.building.heading + 270) % 360,
        }
    
    def read_csv_to_dataframe(self):
        """
        Read the CSV file and return a pandas DataFrame with the required columns.
        """
        # Read the CSV file
        df = pd.read_csv(self.csv_path)
        
        # Check required columns
        required_columns = ['URI', 'isRoom', 'isFacing']
        if all(col in df.columns for col in required_columns):
            # Return all columns in case we need additional data
            return df
        else:
            missing_cols = [col for col in required_columns if col not in df.columns]
            raise ValueError(f"Required columns {missing_cols} not found in CSV file.")
    
    def enrich_rdf(self):
        """
        Read the CSV file and add triples to the graph for isRoom and window orientations.
        """
        # Read the CSV file
        df = self.read_csv_to_dataframe()
        
        # Add triples to the graph
        for _, row in df.iterrows():
            room_uri = URIRef(row['URI'])
            
            # Removed the room number triple that you don't want
            # (The code that was here adding ex-ont:roomNumber has been removed)
            
            # Add isRoom property - handle both string and boolean types
            if isinstance(row['isRoom'], str):
                is_room_value = row['isRoom'].strip().upper() == 'TRUE'
            else:
                # If pandas already converted to boolean
                is_room_value = bool(row['isRoom'])
                
            self.graph.add((room_uri, self.EX.isRoom, Literal(is_room_value, datatype=XSD.boolean)))
            
            # Parse the facing directions and create window instances
            if pd.notna(row['isFacing']) and isinstance(row['isFacing'], str):
                directions_text = row['isFacing'].strip().lower()
                if directions_text != 'none' and directions_text != '':
                    # If isFacing contains multiple directions (comma-separated)
                    directions = [d.strip() for d in directions_text.split(',')]
                    
                    # Create window instances for each direction
                    for direction in directions:
                        if direction in self.direction_heading_map:
                            # Extract room_name from URI for creating window URIs
                            room_name = row['URI'].split('/')[-1]
                            
                            # Create a window URI using ex: namespace and the facing direction
                            window_uri = self.EX[f"{room_name}_window_{direction}"]
                            
                            # Add window instance
                            self.graph.add((window_uri, RDF.type, self.BOT.Element))
                            
                            # Create more descriptive label
                            if 'room_number' in row and pd.notna(row['room_number']):
                                window_label = f"{direction.capitalize()}-facing Window of Room {row['room_number']}"
                            else:
                                window_label = f"{direction.capitalize()}-facing Window of {room_name}"
                                
                            self.graph.add((window_uri, RDFS.label, Literal(window_label)))
                            
                            # Link window to room
                            self.graph.add((room_uri, self.EXONT.hasWindow, window_uri))
                            
                            # Add both the building-relative direction and actual heading
                            self.graph.add((window_uri, self.EXONT.facingRelativeDirection, Literal(direction, datatype=XSD.string)))
                            heading = self.direction_heading_map[direction]
                            self.graph.add((window_uri, self.EXONT.hasFacingDirection, Literal(heading, datatype=XSD.integer)))
        
        return self.graph
    
    def save_rdf(self, output_path):
        """
        Save the graph to a file in Turtle format.
        """
        self.graph.serialize(destination=output_path, format='turtle')
    
    def run(self, output_path="data/topology/VideoLab_floor7_csv_enrichment.ttl"):
        """
        Run the full process: read CSV, enrich RDF, and save to file.
        """
        self.enrich_rdf()
        self.save_rdf(output_path)
        return self.graph

if __name__ == "__main__":
    # Initialize the VideoLab building
    videolab = Building("VideoLab")

    # Create an instance of the RoomEnrichmentViaCSV class
    enrichment = RoomEnrichmentViaCSV(building=videolab)
    graph = enrichment.run()
    
    # Print some statistics
    print(f"Enrichment complete. Created RDF file with {len(graph)} triples.")
