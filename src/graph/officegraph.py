from typing import Dict, List, Tuple
import networkx as nx
import numpy as np
from rdflib import Graph, Namespace, URIRef, RDF
import logging
logger = logging.getLogger(__name__)

from ..core import Device, Measurement, Room, Floor
from ..utils.get_devices_on_floor import FloorDeviceRetriever
from ..utils.ttl_loader import (
    load_multiple_ttl_files,
    load_device_files,
    load_devices_in_rooms_enrichment,
    load_wikidata_days_enrichment,
    load_floor7_graph_learning_enrichments,
)

class OfficeGraph:
    """Class to represent and manipulate the IoT Office Graph."""
   
    def __init__(self, base_dir = 'data/OfficeGraph', 
                 load_only_7th_floor: bool = True, add_enrichments: bool = False):
        """Initialize the OfficeGraph object."""
        # Main RDF graph
        self.graph = Graph()
        self.base_dir = base_dir
       
        # Define namespaces
        self.IC = Namespace("https://interconnectproject.eu/example/")
        self.SAREF = Namespace("https://saref.etsi.org/core/")
        self.S4ENER = Namespace("https://saref.etsi.org/saref4ener/")
        self.S4BLDG = Namespace("https://saref.etsi.org/saref4bldg/")
        self.OM = Namespace("http://www.wurvoc.org/vocabularies/om-1.8/")
       
        # Bind namespaces to prefixes for serialization
        self.graph.bind("ic", self.IC)
        self.graph.bind("saref", self.SAREF)
        self.graph.bind("s4ener", self.S4ENER)
        self.graph.bind("s4bldg", self.S4BLDG)
        self.graph.bind("om", self.OM)
       
        # Collections to store entities
        self.devices: Dict[URIRef, Device] = {}
        self.measurements: Dict[URIRef, Measurement] = {}
        self.rooms: Dict[URIRef, Room] = {}
        self.floors: Dict[URIRef, Floor] = {}
        
        # Mapping property types (e.g., "Temperature") to lists of specific property URIs from devices:
        self.property_type_mappings: Dict[str, List[URIRef]] = {}
    
        # Measurement sequence links
        self.measurement_sequences: Dict[Tuple[URIRef, URIRef], List[Measurement]] = {}

        # Additional mapping dictionaries for downstream tasks
        self.room_to_device: Dict[URIRef, List[URIRef]] = {}
        self.device_to_room: Dict[URIRef, URIRef] = {}
        self.room_to_property_type: Dict[URIRef, List[str]] = {}
        self.room_to_property_measurements: Dict[URIRef, Dict[str, List[Measurement]]] = {}

        # Loading OfficeGraph
        if load_only_7th_floor:
            self.retriever = FloorDeviceRetriever()
            devices_on_floor_7 = self.retriever.get_devices_on_floor(7)

            file_names = set(record["filename"] for record in devices_on_floor_7)
            file_paths = self.retriever.get_full_paths_for_filenames(file_names)

            graph_with_devices_of_a_single_floor = load_multiple_ttl_files(file_paths)
            print("Loaded all device files.")

            ### Solving issue with duplicate rooms ###
            # The ttl files of samsung devices contain weird entries regarding location.
            # I just wanted to get rid of all of them.
            triples_to_remove = []
            for (s, p, o) in graph_with_devices_of_a_single_floor:
                # 1) Remove anything typed as s4bldg:BuildingSpace
                if p == RDF.type and o == self.S4BLDG.BuildingSpace:
                    triples_to_remove.append((s, p, o))
                # 2) Also remove any 'contains', 'isSpaceOf', or 'isContainedIn' statements
                elif p in (self.S4BLDG.contains, self.S4BLDG.isContainedIn, self.S4BLDG.isSpaceOf):
                    triples_to_remove.append((s, p, o))

            # Now remove them in bulk:
            for triple in triples_to_remove:
                graph_with_devices_of_a_single_floor.remove(triple)

            print("Removed unnecessary triples.")
            ### ###

            self.graph += graph_with_devices_of_a_single_floor

            self.device_room_floor_map: Dict[URIRef, Dict[str, str]] = {
                URIRef(rec["deviceURI"]): {
                    "room": rec["buildingSpace"],
                    "floor": rec["floor"],
                }
                for rec in devices_on_floor_7
            }

        else:
            self.load_full_OfficeGraph(self.base_dir)


        # Initializing the extractor
        from .extraction import OfficeGraphExtractor
        self.extractor = OfficeGraphExtractor(self)
        self.extract_all()
        
        # Initializing the builder
        from .builder import OfficeGraphBuilder
        self.builder = OfficeGraphBuilder(self)
        # Load floor plan data
        self.builder.load_floor_plan()
        print("Loaded floor plan data.")

        # Creating Device-Room mappings to class objects
        print("Creating Device-Room mappings to class objects...")
        for dev_uri, mapping in self.device_room_floor_map.items():
            # If the device was extracted
            if dev_uri in self.devices:
                device = self.devices[dev_uri]
                # Grab the room URI from the map
                room_uri = URIRef(mapping["room"])
                device.room = room_uri  # So device_obj.room is set

                # Also add the device to the room object
                if room_uri in self.rooms:
                    self.rooms[room_uri].add_device(dev_uri)
        
        if add_enrichments and load_only_7th_floor:            
            # Load wikidata days enrichment
            wikidata_days_enrichment_graph = load_wikidata_days_enrichment(self.base_dir)
            print("Loaded wikidata days enrichment")

            # Load wikidata days enrichment
            floor7_graph_learning_enrichments_graph = load_floor7_graph_learning_enrichments(self.base_dir)
            print("Loaded devices in rooms enrichment")
                                
            # Add the loaded graphs to self.graph
            self.graph += wikidata_days_enrichment_graph
            self.graph += floor7_graph_learning_enrichments_graph

        # Build additional mapping dictionaries for downstream tasks
        self.build_mappings()

    def load_full_OfficeGraph(self) -> None:
        """
        Load and merge RDFLib graphs from .ttl files in the specified OfficeGraph data directory.

        Args:
            base_dir (str): The base directory containing the .ttl files.
        """
        # Load device files
        device_graph = load_device_files(self.base_dir)
        print("Loaded device files")
        
        # Load devices in rooms enrichment
        devices_in_rooms_enrichment_graph = load_devices_in_rooms_enrichment(self.base_dir)
        print("Loaded devices in rooms enrichment")

        # Load wikidata days enrichment
        wikidata_days_enrichment_graph = load_wikidata_days_enrichment(self.base_dir)
        print("Loaded wikidata days enrichment enrichment")
                               
        # Add the loaded graphs to self.graph
        self.graph += device_graph
        self.graph += devices_in_rooms_enrichment_graph
        self.graph += wikidata_days_enrichment_graph

    def extract_all(self) -> None:
        """
        Extract all the relevant data from `self.graph` into Python objects.
        We delegate to the OfficeGraphExtractor to do the heavy lifting.
        """
        print("Extracting the OfficeGraph data...")

        # 1) Rooms & Floors
        self.extractor.extract_rooms()
        print("- Rooms and Floors extracted")
        # 2) Devices
        self.extractor.extract_devices()
        print("- Devices extracted")
        # 3) Measurements
        self.extractor.extract_measurements()
        print("- Measurements extracted")
        # 4) Measurement sequences
        self.extractor.build_measurement_sequences()
        print("- Measurement sequences extracted")
        # 5) Property type mappings
        self.extractor.extract_property_type_mappings()
        print("- Property type mappings extracted")

        logger.info("Extraction done: devices=%d, rooms=%d, measurements=%d, property_types=%d",
            len(self.devices), len(self.rooms), len(self.measurements), 
            len(self.property_type_mappings))
   
    def get_room_adjacency(self) -> Tuple[np.ndarray, List[URIRef]]:
        """
        Get the room adjacency matrix using the builder.
       
        Returns:
            The room adjacency matrix and room URIs
        """
        return self.builder.build_room_adjacency()
   
    def get_device_room_adjacency(self) -> Tuple[np.ndarray, List[URIRef], List[URIRef]]:
        """
        Get the device-room adjacency matrix using the builder.
        
        Returns:
            The device-room adjacency matrix, device URIs, and room URIs
        """
        return self.builder.build_device_room_adjacency()
   
    def get_heterogeneous_graph(self) -> nx.MultiDiGraph:
        """
        Get the heterogeneous graph using the builder.
       
        Returns:
            A NetworkX MultiDiGraph representing the heterogeneous graph
        """
        return self.builder.build_heterogeneous_graph()

    def get_simple_homogeneous_graph(self) -> nx.MultiDiGraph:
        """
        Get the simple homogenous graph using the builder.
       
        Returns:
            A NetworkX MultiDiGraph representing the homogenous graph
        """
        return self.builder.build_simple_homogeneous_graph()

    def build_mappings(self) -> None:
        """
        Build useful mapping dictionaries for downstream tasks:
        - room_to_device: Maps room URIs to lists of device URIs
        - device_to_room: Maps device URIs to their containing room URIs
        - room_to_property_type: Maps room URIs to lists of property types measured within them
        - room_to_property_measurements: Nested dictionary mapping rooms to property types to lists of measurements
        """
        print("Building mapping dictionaries...")
        
        # Initialize dictionaries
        self.room_to_device = {}
        self.device_to_room = {}
        self.room_to_property_type = {}
        self.room_to_property_measurements = {}
        
        # Build room_to_device and device_to_room mappings
        for device_uri, mapping in self.device_room_floor_map.items():
            room_uri = URIRef(mapping["room"])
            
            # room_to_device mapping
            if room_uri not in self.room_to_device:
                self.room_to_device[room_uri] = []
            self.room_to_device[room_uri].append(device_uri)
            
            # device_to_room mapping
            self.device_to_room[device_uri] = room_uri
        
        # Build room_to_property_type mapping
        # First, get the properties measured by each device
        device_to_property_types = {}
        for device_uri, device in self.devices.items():
            property_types = set()
            for meas in device.measurements:
                if meas.property_type:
                    # Extract property type name from URI
                    property_type_str = str(meas.property_type)
                    # Check if it's in our property_type_mappings
                    for prop_name, prop_uris in self.property_type_mappings.items():
                        if meas.property_type in prop_uris:
                            property_types.add(prop_name)
                            break
            
            device_to_property_types[device_uri] = property_types
        
        # Now build room_to_property_type using device_to_room and device_to_property_types
        for device_uri, property_types in device_to_property_types.items():
            if device_uri in self.device_to_room:
                room_uri = self.device_to_room[device_uri]
                
                if room_uri not in self.room_to_property_type:
                    self.room_to_property_type[room_uri] = set()
                
                self.room_to_property_type[room_uri].update(property_types)
        
        # Convert sets to lists for easier serialization
        for room_uri in self.room_to_property_type:
            self.room_to_property_type[room_uri] = list(self.room_to_property_type[room_uri])
        
        # Build room_to_property_measurements
        # This complex nested dictionary maps: room -> property_type -> list of measurements
        for room_uri, property_types in self.room_to_property_type.items():
            self.room_to_property_measurements[room_uri] = {}
            
            for prop_type in property_types:
                # Initialize empty list for this property type
                self.room_to_property_measurements[room_uri][prop_type] = []
                
                # Get devices in this room
                if room_uri in self.room_to_device:
                    for device_uri in self.room_to_device[room_uri]:
                        if device_uri in self.devices:
                            device = self.devices[device_uri]
                            
                            # For each measurement in the device
                            for meas in device.measurements:
                                if meas.property_type:
                                    # Check if this measurement's property belongs to current property type
                                    for p_uri in self.property_type_mappings.get(prop_type, []):
                                        if meas.property_type == p_uri:
                                            self.room_to_property_measurements[room_uri][prop_type].append(meas)
                                            break
        
        print(f"Built mappings for {len(self.room_to_device)} rooms, {len(self.device_to_room)} devices")
        print(f"Rooms with property types: {len(self.room_to_property_type)}")
        print(f"Rooms with property measurements: {len(self.room_to_property_measurements)}")
