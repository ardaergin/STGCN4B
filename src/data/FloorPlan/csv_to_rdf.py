from rdflib import URIRef, Literal, RDF
from rdflib.namespace import RDFS, XSD
import pandas as pd
import os

from ...config.namespaces import NamespaceMixin
from ...core.building import Building

class RoomEnrichmentViaCSV(NamespaceMixin):
    """
    Read a CSV with columns 'URI', 'isFacing', 'isRoom' and add corresponding triples
    into an existing RDF Graph using a custom namespace for the new predicates.
    """
    def __init__(self,
                 building: Building,
                 floor_number: int,
                 csv_dir: str = "data/topology/CSVs",
                 output_dir: str = "data/topology/TTLs"):
        self.floor_number = floor_number
        self.csv_dir = csv_dir
        self.output_dir = output_dir
        self.csv_path = os.path.join(csv_dir, f"floor_{floor_number}.csv")
        self.building = building
        self.graph = self.create_empty_graph_with_namespace_bindings()
        
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
        # Check if CSV file exists
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
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
                        
            # Add isRoom property - handle both string and boolean types
            if isinstance(row['isRoom'], str):
                is_room_value = row['isRoom'].strip().upper() == 'TRUE'
            else:
                # If pandas already converted to boolean
                is_room_value = bool(row['isRoom'])
                
            self.graph.add((room_uri, self.EX_ONT.isProperRoom, Literal(is_room_value, datatype=XSD.boolean)))
            
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
                            self.graph.add((room_uri, self.EX_ONT.hasWindow, window_uri))
                            
                            # Add both the building-relative direction and actual heading
                            self.graph.add((window_uri, self.EX_ONT.facingRelativeDirection, Literal(direction, datatype=XSD.string)))
                            heading = self.direction_heading_map[direction]
                            self.graph.add((window_uri, self.EX_ONT.hasFacingDirection, Literal(heading, datatype=XSD.integer)))
        
        return self.graph
    
    def save_rdf(self, output_path):
        """
        Save the graph to a file in Turtle format.
        """
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        self.graph.serialize(destination=output_path, format='turtle')
    
    def run(self):
        """
        Run the full process: read CSV, enrich RDF, and save to file.
        """
        # Generate output path
        output_filename = f"floor_{self.floor_number}_enrichment.ttl"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Process the data
        self.enrich_rdf()
        self.save_rdf(output_path)
        
        print(f"Floor {self.floor_number}: Created RDF file with {len(self.graph)} triples at {output_path}")
        return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enrich RDF with room and window data from CSV files")
    
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="data/topology/CSVs",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/topology/TTLs",
        help="Output directory for RDF files"
    )
    parser.add_argument(
        "--all_floors",
        action="store_true",
        help="If set, generate RDF for all floors 1 through 7"
    )
    parser.add_argument(
        "--floor",
        type=int,
        help="Single floor number to extract (required if --all_floors is not set)"
    )
    
    args = parser.parse_args()
    
    if not args.all_floors and args.floor is None:
        parser.error("--floor is required if --all_floors is not set")
    
    # Initialize the VideoLab building
    videolab = Building("VideoLab")
    
    if args.all_floors:
        for floor in range(1, 8):
            print(f"Generating RDF for floor {floor}...")
            enrichment = RoomEnrichmentViaCSV(
                building=videolab,
                floor_number=floor,
                csv_dir=args.csv_dir,
                output_dir=args.output_dir
            )
            ttl_path = enrichment.run()
    else:
        enrichment = RoomEnrichmentViaCSV(
            building=videolab,
            floor_number=args.floor,
            csv_dir=args.csv_dir,
            output_dir=args.output_dir
        )
        ttl_path = enrichment.run()