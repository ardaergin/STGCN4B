import datetime
import functools
import numpy as np
import os, pickle
from typing import Any, Dict, List, Tuple, Optional
from rdflib import Graph, URIRef, RDF

import logging, sys
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
   
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize the OfficeGraph object.

        Args:
            base_dir (str, optional): Base directory for data files. Defaults to 'data'.
        """
        self.base_dir = data_dir
        self._init_collections()

    def _init_collections(self):
        """Initialize collections to store entities."""
        self.building: Optional[Building] = Building("VideoLab")
        self.floors: Dict[URIRef, Floor] = {}
        self.rooms: Dict[URIRef, Room] = {}
        self.devices: Dict[URIRef, Device] = {}
        self.measurements: Dict[URIRef, Measurement] = {}

        # Mapping property types to lists of specific property URIs
        self.property_type_mappings: Dict[str, List[URIRef]] = {}
        
        # Measurement sequence links
        self.measurement_sequences: Dict[Tuple[URIRef, URIRef], List[Measurement]] = {}
        
        # Store metadata about what was loaded
        self.metadata = {
            'floors_to_load': [],
            'total_floors': 0,
            'total_rooms': 0,
            'total_devices': 0,
            'total_measurements': 0,
            'total_properties': 0,
            'extraction_timestamp': None
        }

    def _update_metadata(self) -> Dict:
        """Update metadata with current state."""
        total_properties = sum(len(uris) for uris in self.property_type_mappings.values())

        self.metadata.update({
            'building_name': self.building.name if self.building else None,
            'floors_to_load': self.floors_to_load,
            'total_floors': len(self.floors),
            'total_rooms': len(self.rooms),
            'total_devices': len(self.devices),
            'total_measurements': len(self.measurements),
            'total_properties': total_properties,
            'extraction_timestamp': datetime.datetime.now().isoformat()
        })

        logger.info("Metadata updated: floors_to_load=%s, total_rooms=%d, total_devices=%d, total_measurements=%d, total_properties=%d", 
            self.metadata['floors_to_load'], 
            self.metadata['total_rooms'], 
            self.metadata['total_devices'], 
            self.metadata['total_measurements'],
            self.metadata['total_properties'])

        return self.metadata
    


    ##############################
    # Loading from scratch from RDF data
    ##############################
    

    @classmethod
    def from_rdf(cls, floors_to_load: List[int], data_dir: str = 'data', auto_extract: bool = True):
        """
        Create an OfficeGraph instance by loading RDF data and extracting entities.
        For now, only one floor can be loaded at a time due to memory constraints.
        """
        if len(floors_to_load) > 1:
            raise ValueError("Currently, due to memory constraints, just a single floor can be loaded.")

        instance = cls(data_dir=data_dir)
        instance.floors_to_load = floors_to_load
        instance.graph = instance.create_empty_graph_with_namespace_bindings()

        # Load building topology for specified floors
        building_topology_graph = load_building_topology(data_dir, floors=floors_to_load)
        instance.graph += building_topology_graph
        logger.info("Loaded building topology for floors %s (%d triples)", floors_to_load, len(building_topology_graph))

        # Load CSV enrichment
        csv_enrichment_graph = load_csv_enrichment(data_dir, floors=floors_to_load)
        instance.graph += csv_enrichment_graph
        logger.info("Loaded CSV enrichment for floors %s (%d triples)", floors_to_load, len(csv_enrichment_graph))

        # Initialize retriever and load OfficeGraph
        instance.retriever = FloorDeviceRetriever()
        instance.load_devices_on_floors(floors_to_load)
        
        # Automatically extract entities if requested
        if auto_extract:
            logger.info("Extracting entities from the graph...")
            instance.extract_all()
            logger.info("Entity extraction complete")
        
        # Update metadata
        instance._update_metadata()
        return instance

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



    ##############################
    # Loading from pickle(s)
    ##############################


    @classmethod
    def from_pickles(cls, floors_to_load: List[int], data_dir: str = 'data'):
        """
        Create an OfficeGraph instance by loading extracted entity data from one or more pickle files.
        Each floor must have a corresponding file called "officegraph_entities_floor_{floor}.pkl" 
        under the {data_dir}/interim/.
        """

        instance = cls(data_dir=data_dir)
        instance.floors_to_load = floors_to_load
        instance.graph = instance.create_empty_graph_with_namespace_bindings()

        for floor in floors_to_load:
            fname = f"officegraph_entities_floor_{floor}.pkl"
            pickle_path = os.path.join(data_dir, "interim", fname)
            try:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
            except FileNotFoundError:
                logger.error("Pickle file not found for floor %d: %s", floor, pickle_path)
                raise
            except (OSError, pickle.UnpicklingError) as e:
                logger.error("Error loading pickle for floor %d: %s", floor, e)
                raise

            instance.devices.update(data.get('devices', {}))
            instance.measurements.update(data.get('measurements', {}))
            instance.rooms.update(data.get('rooms', {}))
            instance.floors.update(data.get('floors', {}))
            if data.get('building') is not None:
                instance.building = data['building']

            for prop_type, new_uris in data.get('property_type_mappings', {}).items():
                existing_uris = instance.property_type_mappings.setdefault(prop_type, [])
                for uri in new_uris:
                    if uri not in existing_uris:
                        existing_uris.append(uri)

            for k, seq in data.get('measurement_sequences', {}).items():
                instance.measurement_sequences.setdefault(k, []).extend(seq)

            logger.info("Loaded floor %d: %d devices, %d measurements, %d rooms",
                        floor,
                        len(data.get('devices', {})),
                        len(data.get('measurements', {})),
                        len(data.get('rooms', {}))
            )

        instance._update_metadata()

        logger.info("Finished loading from pickle(s).")
        
        return instance
    


    ##############################
    # Saving to a pickle
    ##############################


    def save_entities_to_pickle(self) -> str:
        """
        Save all extracted entities (excluding the RDF graph, if exists) to a pickle file.
        The filename will include the floor number(s) that were loaded.
        """
        if sorted(self.floors_to_load) == list(range(1, 8)):
            filename = "officegraph_entities.pkl"
        else:
            floors_str = "_".join(str(f) for f in self.floors_to_load) # turn [7] into "7", or [1,2] into "1_2", etc.
            filename = f"officegraph_entities_floor_{floors_str}.pkl"
        
        pickle_path = os.path.join(self.base_dir, "interim", filename)

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
            logger.info(
                "Saved data: %d devices, %d measurements, %d rooms, %d floors",
                len(self.devices), len(self.measurements),
                len(self.rooms), len(self.floors)
            )
            return pickle_path
        except (OSError, pickle.PicklingError) as e:
            logger.error("Failed to save OfficeGraph entities: %s", e)
            raise

    ##############################
    # Convenience helpers
    ##############################

    def _map_DeviceURIRef_to_RoomURIRef(self, dev_uriref: URIRef) -> URIRef:
        device_obj = self.devices.get(dev_uriref)
        if device_obj is None:
            raise KeyError(f"Device with URI <{dev_uriref}> not found in self.devices.")
        
        room_uriref = device_obj.room
        if room_uriref is None:
            raise ValueError(f"Device <{dev_uriref}> does not have an associated room.")
            
        return room_uriref

    def _map_RoomURIRef_to_FloorURIRef(self, room_uriref: URIRef) -> URIRef:
        room_obj = self.rooms.get(room_uriref)
        if room_obj is None:
            raise KeyError(f"Room with URI <{room_uriref}> not found in self.rooms.")

        floor_uriref = room_obj.floor
        if floor_uriref is None:
            raise ValueError(f"Room <{room_uriref}> does not have an associated floor.")

        return floor_uriref

    def _map_DeviceURIRef_to_FloorNumber(self, dev_uriref: URIRef) -> URIRef:
        room_uriref = self._map_DeviceURIRef_to_RoomURIRef(dev_uriref)
        floor_uriref = self._map_RoomURIRef_to_FloorURIRef(room_uriref)
        floor_obj = self.floors.get(floor_uriref, None)
        if floor_obj is None:
            raise KeyError(f"Floor object with URI <{floor_uriref}> not found in self.floors.")
        else: 
            return floor_obj.floor_number
    
    def _map_RoomURIRef_to_RoomNumber(self, room_uriref: URIRef) -> str:
        room_obj = self.rooms.get(room_uriref)
        if room_obj is None:
            raise KeyError(f"Room with URI <{room_uriref}> not found in self.rooms.")
        
        if not hasattr(room_obj, 'room_number') or room_obj.room_number is None:
            raise AttributeError(f"Room <{room_uriref}> does not have a 'room_number' attribute.")
            
        return room_obj.room_number

    def _map_FloorURIRef_to_FloorNumber(self, floor_uriref: URIRef) -> int:
        floor_obj = self.floors.get(floor_uriref)
        if floor_obj is None:
            raise KeyError(f"Floor with URI <{floor_uriref}> not found in self.floors.")
            
        if not hasattr(floor_obj, 'floor_number') or floor_obj.floor_number is None:
            raise AttributeError(f"Floor <{floor_uriref}> does not have a 'floor_number' attribute.")

        return floor_obj.floor_number

    @staticmethod
    def _get_nested_attr(obj: Any, attr_string: str, default: Any = np.nan) -> Any:
        """
        Private helper to safely access nested attributes and dictionary keys.

        Used especially to access the nested attributes for the Room class instances.
        """
        try:
            attributes = attr_string.split('.')
            def _reducer(current_obj, part):
                if isinstance(current_obj, dict):
                    return current_obj.get(part)
                else:
                    return getattr(current_obj, part)
            final_value = functools.reduce(_reducer, attributes, obj)
            return final_value if final_value is not None else default
        except (AttributeError, TypeError):
            return default

    def __str__(self):
        floors = ", ".join(str(f) for f in self.metadata.get('floors_to_load', []))
        
        summary = (
            f"OfficeGraph Summary\n"
            f"--------------------\n"
            f"Building            : {self.metadata.get('building_name', '?')}\n"
            f"Loaded floors       : {floors}\n"
            f"Floors              : {self.metadata.get('total_floors', 0)}\n"
            f"Rooms               : {self.metadata.get('total_rooms', 0)}\n"
            f"Devices             : {self.metadata.get('total_devices', 0)}\n"
            f"Measurements        : {self.metadata.get('total_measurements', 0)}\n"
            f"Properties          : {self.metadata.get('total_properties', 0)}\n"
            f"Extraction timestamp: {self.metadata.get('extraction_timestamp', 'N/A')}"
        )

        if hasattr(self, 'graph'):
            summary += f"\nRDF triples         : {len(self.graph)}"

        return summary

def main():
    ### Arguments ###
    from ..config.args import parse_args
    args = parse_args()

    ### Running ###
    logger.info("Starting OfficeGraph with base directory '%s' and floors %s", args.data_dir, args.floors)
    
    # Create from RDF
    office_graph = OfficeGraph.from_rdf(
        floors_to_load=args.floors,
        data_dir=args.data_dir,
        auto_extract=not args.no_extract
    )
    
    # Print summary
    print(office_graph)
    
    # Save entities to pickle by default
    if not args.no_save_pickle:
        pickle_path = office_graph.save_entities_to_pickle()
        logger.info("Entities saved to: %s", pickle_path)

if __name__ == "__main__":
    main()