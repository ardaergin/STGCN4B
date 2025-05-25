import os, sys, pickle, argparse
from typing import Dict, List, Tuple, Optional, Set
from rdflib import Graph, URIRef, RDF

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from ..config.namespaces import NamespaceMixin
from ..core import Device, Measurement, Room, Floor, Building
from ..data.OfficeGraph.device_loader import FloorDeviceRetriever
from ..data.OfficeGraph.ttl_loader import load_multiple_ttl_files, load_building_topology, load_csv_enrichment

from .extraction import OfficeGraphExtractor

class OfficeGraph(NamespaceMixin, OfficeGraphExtractor):
    """Class to represent and manipulate the IoT Office Graph."""
   
    def __init__(self, 
                 base_dir: str = 'data', 
                 floors_to_load: Optional[List[int]] = None,
                 auto_extract: bool = True,
                 parallel_extraction: bool = False,
                 load_from_pickle: bool = False,
                 pickle_path: Optional[str] = None):
        """
        Initialize the OfficeGraph object.

        Args:
            base_dir (str, optional): Base directory for data files. Defaults to 'data'.
            floors_to_load (Optional[List[int]], optional): List of floor numbers to load. 
                                                          Defaults to [7] if None.
            auto_extract (bool, optional): Whether to automatically extract entities after loading.
                                         Defaults to True.
            parallel_extraction (bool, optional): Whether to use parallel processing for extraction.
                                                 Defaults to False.
            load_from_pickle (bool, optional): Whether to load from a saved pickle file.
                                             Defaults to False.
            pickle_path (Optional[str], optional): Path to the pickle file. If None, uses default path.
        """
        # Default to loading only the 7th floor
        if floors_to_load is None:
            floors_to_load = [7]

        self.base_dir = base_dir
        self.floors_to_load = floors_to_load
        
        # Initialize collections
        self._init_collections()

        if load_from_pickle:
            # Load from pickle file
            if pickle_path is None:
                pickle_path = os.path.join(base_dir, "processed", "officegraph_entities.pkl")
            
            self.load_from_pickle(pickle_path)
            logger.info("Loaded OfficeGraph entities from %s", pickle_path)
        else:
            # Load from RDF data
            self._load_from_rdf(floors_to_load, auto_extract, parallel_extraction)

    def _load_from_rdf(self, floors_to_load: List[int], auto_extract: bool, parallel_extraction: bool):
        """Load and process RDF data."""
        # Main RDF graph - only created when loading from RDF
        self.graph = self.create_empty_graph_with_namespace_bindings()

        # Load building topology for specified floors
        building_topology_graph = load_building_topology(self.base_dir, floors=floors_to_load)
        self.graph += building_topology_graph
        logger.info("Loaded building topology for floors %s (%d triples)", floors_to_load, len(building_topology_graph))

        # Load CSV enrichment
        csv_enrichment_graph = load_csv_enrichment(self.base_dir, floors=floors_to_load)
        self.graph += csv_enrichment_graph
        logger.info("Loaded CSV enrichment for floors %s (%d triples)", floors_to_load, len(csv_enrichment_graph))

        # Initialize retriever and load OfficeGraph
        self.retriever = FloorDeviceRetriever()
        self.load_devices_on_floors(floors_to_load)
        
        # Automatically extract entities if requested
        if auto_extract:
            if parallel_extraction:
                import multiprocessing
                num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
                logger.info("Running in parallel mode with %d workers", num_workers)
                self.extract_all_parallel(num_workers=num_workers)
            else:
                logger.info("Extracting entities from the graph...")
                self.extract_all()
            logger.info("Entity extraction complete")
    
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
        
        # Store metadata about what was loaded
        self.metadata = {
            'floors_loaded': [],
            'total_devices': 0,
            'total_measurements': 0,
            'total_rooms': 0,
            'total_floors': 0,
            'extraction_timestamp': None
        }
    
    def save_entities_to_pickle(self, pickle_path: Optional[str] = None) -> str:
        """
        Save all extracted entities (excluding the RDF graph) to a pickle file.
        
        Args:
            pickle_path (Optional[str]): Path where to save the pickle file.
                                       If None, uses default path.
        
        Returns:
            str: Path where the file was saved.
        """
        if pickle_path is None:
            pickle_path = os.path.join(self.base_dir, "processed", "officegraph_entities.pkl")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
        
        # Prepare data to save (everything except the RDF graph)
        entities_data = {
            'devices': self.devices,
            'measurements': self.measurements,
            'rooms': self.rooms,
            'floors': self.floors,
            'building': self.building,
            'property_type_mappings': self.property_type_mappings,
            'measurement_sequences': self.measurement_sequences,
            'base_dir': self.base_dir,
            'floors_to_load': self.floors_to_load,
            'metadata': self._update_metadata()
        }
        
        try:
            with open(pickle_path, "wb") as f:
                pickle.dump(entities_data, f)
            logger.info("Successfully saved OfficeGraph entities to %s", pickle_path)
            logger.info("Saved data: %d devices, %d measurements, %d rooms, %d floors", 
                       len(self.devices), len(self.measurements), len(self.rooms), len(self.floors))
            return pickle_path
        except (OSError, pickle.PicklingError) as e:
            logger.error("Failed to save OfficeGraph entities: %s", e)
            raise
    
    def load_from_pickle(self, pickle_path: str) -> None:
        """
        Load all extracted entities from a pickle file.
        
        Args:
            pickle_path (str): Path to the pickle file to load.
        """
        try:
            with open(pickle_path, "rb") as f:
                entities_data = pickle.load(f)
            
            # Restore all the entity collections
            self.devices = entities_data['devices']
            self.measurements = entities_data['measurements']
            self.rooms = entities_data['rooms']
            self.floors = entities_data['floors']
            self.building = entities_data['building']
            self.property_type_mappings = entities_data['property_type_mappings']
            self.measurement_sequences = entities_data['measurement_sequences']
            self.base_dir = entities_data['base_dir']
            self.floors_to_load = entities_data['floors_to_load']
            self.metadata = entities_data.get('metadata', {})
            
            logger.info("Loaded data: %d devices, %d measurements, %d rooms, %d floors", 
                       len(self.devices), len(self.measurements), len(self.rooms), len(self.floors))
            
        except (OSError, pickle.UnpicklingError) as e:
            logger.error("Failed to load OfficeGraph entities from %s: %s", pickle_path, e)
            raise
        except FileNotFoundError:
            logger.error("Pickle file not found: %s", pickle_path)
            raise
    
    def _update_metadata(self) -> Dict:
        """Update metadata with current state."""
        import datetime
        self.metadata.update({
            'floors_loaded': self.floors_to_load,
            'total_devices': len(self.devices),
            'total_measurements': len(self.measurements),
            'total_rooms': len(self.rooms),
            'total_floors': len(self.floors),
            'extraction_timestamp': datetime.datetime.now().isoformat()
        })
        return self.metadata
    
    @classmethod
    def load_entities_from_pickle(cls, pickle_path: str) -> 'OfficeGraph':
        """
        Class method to create an OfficeGraph instance by loading from a pickle file.
        
        Args:
            pickle_path (str): Path to the pickle file.
            
        Returns:
            OfficeGraph: A new OfficeGraph instance with loaded entities.
        """
        return cls(load_from_pickle=True, pickle_path=pickle_path)
    
    def has_graph(self) -> bool:
        """Check if the instance has an RDF graph loaded."""
        return hasattr(self, 'graph') and self.graph is not None
    
    # All your existing methods remain the same...
    def load_devices_on_floors(self, floor_numbers: List[int]) -> None:
        """
        Load data for specified floor(s).
        
        Args:
            floor_numbers (List[int]): List of floor numbers to load
        """
        logger.info("Loading data for floors: %s", floor_numbers)
        
        # Create a combined relationship graph for all floors
        all_floors_graph = Graph()
        all_file_paths = []
        
        for floor_num in floor_numbers:
            # Get the relationship graph for this floor
            floor_graph = self.retriever.get_device_room_floor_graph(floor_num)
            logger.info("Retrieved relationship graph for floor %d with %d triples", floor_num, len(floor_graph))
            
            # Add it to the combined graph
            all_floors_graph += floor_graph
            
            # Get the device filenames for this floor
            file_paths = self.retriever.get_device_filenames_from_graph(floor_graph)
            all_file_paths.extend(file_paths)
            logger.info("Found %d device files for floor %d", len(file_paths), floor_num)
        
        # Remove duplicate file paths
        all_file_paths = list(set(all_file_paths))
        
        # Load TTL files
        graph_with_devices = load_multiple_ttl_files(all_file_paths)
        logger.info("Loaded %d device files", len(all_file_paths))
        graph_with_devices_cleaned = self._remove_duplicate_room_triples(graph_with_devices)
        
        # Add both graphs to the main graph
        self.graph += graph_with_devices_cleaned
        logger.info("Successfully added the data of %d device(s) to the graph", len(all_file_paths))
        self.graph += all_floors_graph
        logger.info("Successfully added the data of %d floor(s) to the graph", len(floor_numbers))

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

        logger.info("Removed %d unnecessary triples related to duplicate rooms", len(triples_to_remove))
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


def main():
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
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip automatic entity extraction"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Utilize parallel processing for entity extraction (default: False)"
    )
    parser.add_argument(
        "--load-from-pickle",
        action="store_true",
        help="Load from pickle file instead of processing RDF data"
    )
    parser.add_argument(
        "--entities-pickle-path",
        type=str,
        default="data/processed/officegraph_entities.pkl",
        help="Path for saving/loading entities pickle file"
    )

    args = parser.parse_args()

    ### Running ###
    if args.load_from_pickle:
        logger.info("Loading OfficeGraph entities from %s", args.entities_pickle_path)
        office_graph = OfficeGraph.load_entities_from_pickle(args.entities_pickle_path)
        logger.info("OfficeGraph loaded successfully from pickle")
    else:
        # Process from RDF data
        logger.info("Starting OfficeGraph with base directory '%s' and floors %s", args.base_dir, args.floors)
        office_graph = OfficeGraph(
            base_dir=args.base_dir, 
            floors_to_load=args.floors,
            auto_extract=not args.no_extract,
            parallel_extraction=args.parallel,
            load_from_pickle = False,
            pickle_path = None
        )
        
        if office_graph.has_graph():
            total_triples = len(office_graph.graph)
            logger.info("Graph loaded successfully with %d triples", total_triples)
        
        # Save entities to pickle if requested
        office_graph.save_entities_to_pickle(pickle_path=args.entities_pickle_path)

if __name__ == "__main__":
    main()