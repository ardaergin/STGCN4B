import os, sys, pickle, argparse
from typing import Dict, List, Tuple, Optional, Set
from rdflib import Graph, Namespace, URIRef, RDF, RDFS, XSD, OWL
import logging
logger = logging.getLogger(__name__)

from ..core import Device, Measurement, Room, Floor, Building
from ..data.OfficeGraph.device_loader import FloorDeviceRetriever
from ..data.OfficeGraph.ttl_loader import load_multiple_ttl_files, load_VideoLab_topology, load_csv_enrichment

class OfficeGraph:
    """Class to represent and manipulate the IoT Office Graph."""
   
    def __init__(self, 
                 base_dir = 'data', 
                 floors_to_load: Optional[List[int]] = None):
        """
        Initialize the OfficeGraph object.

        Args:
            base_dir (str, optional): Base directory for data files. Defaults to 'data'.
            floors_to_load (Optional[List[int]], optional): List of floor numbers to load. 
                                                          Defaults to [7] if None.
        """
        # Default to loading only the 7th floor
        if floors_to_load is None:
            floors_to_load = [7]
        if floors_to_load == [7]:
            only_floor7 = True
        else: 
            only_floor7 = False

        # Main RDF graph
        self.graph = Graph()
        self.base_dir = base_dir
       
        # Define and bind namespaces
        self._setup_namespaces()

        # Collections to store entities
        self._init_collections()

        # Load VideoLab topology
        VideoLab_topology_graph = load_VideoLab_topology(self.base_dir, load_only_floor7=only_floor7)
        self.graph += VideoLab_topology_graph
        print(f"Loaded VideoLab topology ({len(VideoLab_topology_graph)} triples).")

        # Load CSV enrichment
        csv_enrichment_graph = load_csv_enrichment(self.base_dir)
        self.graph += csv_enrichment_graph
        print(f"Loaded CSV enrichment ({len(csv_enrichment_graph)} triples).")

        # Initialize retriever and load OfficeGraph
        self.retriever = FloorDeviceRetriever()
        self.load_devices_on_floors(floors_to_load)

    def _setup_namespaces(self):
        """Set up and bind namespaces."""
        # Define namespaces
        self.IC = Namespace("https://interconnectproject.eu/example/")
        self.SAREF = Namespace("https://saref.etsi.org/core/")
        self.S4ENER = Namespace("https://saref.etsi.org/saref4ener/")
        self.S4BLDG = Namespace("https://saref.etsi.org/saref4bldg/")
        self.OM = Namespace("http://www.wurvoc.org/vocabularies/om-1.8/")
        self.BOT = Namespace("https://w3id.org/bot#")
        self.GEO = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
        self.GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")
        self.EX = Namespace("https://example.org/")
        self.EXONT = Namespace("https://example.org/ontology#")
        
        # Bind namespaces to prefixes for serialization
        self.graph.bind("ic", self.IC)
        self.graph.bind("saref", self.SAREF)
        self.graph.bind("s4ener", self.S4ENER)
        self.graph.bind("s4bldg", self.S4BLDG)
        self.graph.bind("om", self.OM)
        self.graph.bind("bot", self.BOT)
        self.graph.bind("geo", self.GEO)
        self.graph.bind("geosparql", self.GEOSPARQL)
        self.graph.bind("ex", self.EX)
        self.graph.bind("ex-ont", self.EXONT)

        # built-in RDF namespaces (not necessary, but just in case)
        self.graph.bind("rdfs", RDFS)
        self.graph.bind("xsd", XSD)
        self.graph.bind("owl", OWL)
    
    def _init_collections(self):
        """Initialize collections to store entities."""
        self.devices: Dict[URIRef, Device] = {}
        self.measurements: Dict[URIRef, Measurement] = {}
        self.rooms: Dict[URIRef, Room] = {}
        self.floors: Dict[URIRef, Floor] = {}
        self.building: Optional[Building] = None

        # Mapping property types to lists of specific property URIs
        self.property_type_mappings: Dict[str, List[URIRef]] = {}
        
        # Measurement sequence links
        self.measurement_sequences: Dict[Tuple[URIRef, URIRef], List[Measurement]] = {}
    
    def load_devices_on_floors(self, floor_numbers: List[int]) -> None:
        """
        Load data for specified floor(s).
        
        Args:
            floor_numbers (List[int]): List of floor numbers to load
        """
        print(f"Loading data for floors: {floor_numbers}")
        
        # Create a combined relationship graph for all floors
        all_floors_graph = Graph()
        all_file_paths = []
        
        for floor_num in floor_numbers:
            # Get the relationship graph for this floor
            floor_graph = self.retriever.get_device_room_floor_graph(floor_num)
            print(f"Retrieved relationship graph for floor {floor_num} with {len(floor_graph)} triples")
            
            # Add it to the combined graph
            all_floors_graph += floor_graph
            
            # Get the device filenames for this floor
            file_paths = self.retriever.get_device_filenames_from_graph(floor_graph)
            all_file_paths.extend(file_paths)
            print(f"Found {len(file_paths)} device files for floor {floor_num}")
        
        # Remove duplicate file paths
        all_file_paths = list(set(all_file_paths))
        
        # Load TTL files
        graph_with_devices = load_multiple_ttl_files(all_file_paths)
        print(f"Loaded {len(all_file_paths)} device files.")
        graph_with_devices_cleaned = self._remove_duplicate_room_triples(graph_with_devices)
        
        # Add both graphs to the main graph
        self.graph += graph_with_devices_cleaned
        print(f"Successfully added the data of {len(all_file_paths)} device(s) to the graph.")
        self.graph += all_floors_graph
        print(f"Successfully added the data of {len(floor_numbers)} floor(s) to the graph.")

    def _remove_duplicate_room_triples(self, input_graph: Graph) -> Graph:
        """
        Remove unnecessary triples related to duplicate rooms.
        We already have derived these from the devices_in_rooms enrichment file.
        So, we should remove these redundant entries regarding location from the TTL files of devices. 

        Args:
            input_graph (Graph): Input RDF graph with potentially duplicate room triples
            
        Returns:
            Graph: Cleaned graph with duplicates removed
        """
        # Identify triples to remove
        triples_to_remove = []
        for (s, p, o) in input_graph:
            # 1) Remove anything typed as s4bldg:BuildingSpace
            if p == RDF.type and o == self.S4BLDG.BuildingSpace:
                triples_to_remove.append((s, p, o))
            # 2) Also remove any 'contains', 'isSpaceOf', or 'isContainedIn' statements
            elif p in (self.S4BLDG.contains, self.S4BLDG.isContainedIn, self.S4BLDG.isSpaceOf):
                triples_to_remove.append((s, p, o))

        # Remove them in bulk
        for triple in triples_to_remove:
            input_graph.remove(triple)

        print(f"Removed {len(triples_to_remove)} unnecessary triples related to duplicate rooms.")
        return input_graph
    
    def get_devices_in_room(self, room_uri: URIRef) -> List[Device]:
        """Get all devices in a specific room."""
        if room_uri in self.rooms:
            return [self.devices[device_uri] for device_uri in self.rooms[room_uri].devices
                    if device_uri in self.devices]
        return []
    
    def get_room_for_device(self, device_uri: URIRef) -> Optional[Room]:
        """Get the room containing a specific device."""
        if device_uri in self.devices:
            room_uri = self.devices[device_uri].room
            if room_uri and room_uri in self.rooms:
                return self.rooms[room_uri]
        return None
    
    def get_property_types_in_room(self, room_uri: URIRef) -> Set[str]:
        """Get all property types measured in a specific room."""
        property_types = set()
        devices_in_room = self.get_devices_in_room(room_uri)
        
        for device in devices_in_room:
            # For each property type URI in the device
            for property_uri in device.properties:
                # Find its property type name
                for prop_type, uris in self.property_type_mappings.items():
                    if property_uri in uris:
                        property_types.add(prop_type)
                        break
        
        return property_types
    
    def get_measurements_by_property_in_room(self, room_uri: URIRef, property_type: str) -> List[Measurement]:
        """Get all measurements of a specific property type in a room."""
        if property_type not in self.property_type_mappings:
            return []
        
        property_uris = self.property_type_mappings[property_type]
        measurements = []
        
        for device in self.get_devices_in_room(room_uri):
            for property_uri in property_uris:
                if property_uri in device.measurements_by_property:
                    measurements.extend(device.measurements_by_property[property_uri])
        
        return sorted(measurements, key=lambda m: m.timestamp)






if __name__ == "__main__":
    
    ### Arguments ###
    parser = argparse.ArgumentParser(description="Load and inspect the IoT Office Graph")
    parser.add_argument(
        "--base-dir", 
        type=str,
        default="data",
        help="Base directory for TTL data files (default: data)"
    )
    parser.add_argument(
        "--floors", 
        type=int,
        nargs="+",
        default=[7],
        help="List of floor numbers to load (default: 7)"
    )
    args = parser.parse_args()

    ### Running ###
    print(f"Starting OfficeGraph with base directory '{args.base_dir}' and floors {args.floors}")
    office_graph = OfficeGraph(base_dir=args.base_dir, floors_to_load=args.floors)
    total_triples = len(office_graph.graph)
    print(f"Graph loaded successfully with {total_triples} triples.")

    ### Saving ###
    output_path = os.path.join(args.base_dir, "processed", "officegraph_base.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, "wb") as f:
            pickle.dump(office_graph, f)
        print(f"Successfully saved office graph to {output_path}")
    except (OSError, pickle.PicklingError) as e:
        logger.error("Failed to save office graph: %s", e)
        sys.exit(1)
