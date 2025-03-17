from typing import Dict, List, Tuple, Set, Optional, Union, Any
from collections import defaultdict
import networkx as nx
import numpy as np
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS
import logging
logger = logging.getLogger(__name__)

from ..core import Device, Measurement, Room, Floor
from ..utils.ttl_loader import categorize_ttl_files, load_multiple_ttl_files


class OfficeGraph:
    """Class to represent and manipulate the IoT Knowledge Graph."""
    
    def __init__(self, testing: bool = False):
        """
        Initialize the KnowledgeGraph object.
        Args:
            testing (bool): A flag indicating whether the object is being initialized for testing purposes. Defaults to False.
        """
        self.testing = testing
        # Main RDF graph
        self.graph = Graph()
        
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
        self.buildings: Dict[URIRef, Set[URIRef]] = {}
        
        # Property type mappings
        self.property_types: Dict[URIRef, str] = {}
        
        # Measurement sequence links
        self.measurement_sequences: Dict[Tuple[URIRef, URIRef], List[Measurement]] = {}

        # Loading OfficeGraph
        print(f"Loading OfficeGraph... (Testing Mode: {self.testing})")
        self.load_OfficeGraph(base_dir='data/OfficeGraph')

        # Initializing the extractor
        from .extraction import OfficeGraphExtractor
        self.extractor = OfficeGraphExtractor(self)

        # Initializing the builder
        from .builder import OfficeGraphBuilder
        self.builder = OfficeGraphBuilder(self)
        
    def load_OfficeGraph(self, base_dir: str) -> None:
        """
        Load and merge RDFLib graphs from .ttl files in the specified directory.
        This method categorizes .ttl files in the given base directory, flattens
        the device and enrichment lists, and merges them into a single RDFLib Graph.
        The combined graph is then added to the instance's graph attribute.
        Args:
            base_dir (str): The base directory containing the .ttl files.
        Returns:
            None
        Notes:
            - If the instance is in testing mode, only a subset of the device files
              will be loaded (limited to 5 files).
            - The method prints the number of TTL files loaded and whether it is in
              testing mode.
        """
        # 1) Categorize all .ttl files
        paths_dict = categorize_ttl_files(base_dir)
        
        # Flatten device and enrichment lists
        device_files = paths_dict['devices']
        enrichments = paths_dict['enrichments']
        enrichment_files = (enrichments['devices_in_rooms'])
                            # + enrichments['wikidata_days']
                            # + enrichments['graph_learning']
                            
        # If testing, only take a subset of the files
        if self.testing:
            device_files = device_files[:5]  # Limit to 5 device files
        
        # 2) Merge them into a single big RDFLib Graph
        all_files = device_files + enrichment_files
        combined_graph = load_multiple_ttl_files(all_files)
        
        # 3) Add the combined graph to self.graph
        self.graph += combined_graph  # Merges in-place
        print(f"Loaded {len(all_files)} TTL files")
    
    def extract_all(self) -> None:
        """
        Extract all the relevant data from `self.graph` into Python objects.
        We delegate to the OfficeGraphExtractor to do the heavy lifting.
        """
        print("Extracting the OfficeGraph data...")

        # 1) Rooms & Floors
        self.extractor.extract_rooms_and_floors()
        print("- Rooms and Floors extracted")
        # 2) Devices
        self.extractor.extract_devices()
        print("- Devices extracted")
        # 3) Measurements
        self.extractor.extract_measurements()
        print("- Measurements extracted")
        # 4) Property types
        self.extractor.extract_property_types()
        print("- Property types extracted")
        # 5) Measurement sequences
        self.extractor.build_measurement_sequences()
        print("- dMeasurement sequences extracted")

        logger.info("Extraction done: devices=%d, rooms=%d, floors=%d, measurements=%d",
            len(self.devices), len(self.rooms), len(self.floors), len(self.measurements))

    def get_device_adjacency(self) -> Tuple[np.ndarray, List[URIRef]]:
        """
        Get the device adjacency matrix using the builder.
        
        Returns:
            The device adjacency matrix and device URIs
        """
        return self.builder.build_device_adjacency()
    
    def get_room_adjacency(self) -> Tuple[np.ndarray, List[URIRef]]:
        """
        Get the room adjacency matrix using the builder.
        
        Returns:
            The room adjacency matrix and room URIs
        """
        return self.builder.build_room_adjacency()
    
    def get_heterogeneous_graph(self) -> nx.MultiDiGraph:
        """
        Get the heterogeneous graph using the builder.
        
        Returns:
            A NetworkX MultiDiGraph representing the heterogeneous graph
        """
        return self.builder.build_heterogeneous_graph()
