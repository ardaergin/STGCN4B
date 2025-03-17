from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
from rdflib import Graph, Namespace, URIRef, Literal
from rdflib.namespace import RDF, RDFS

import logging
logger = logging.getLogger(__name__)

from ..core import Device, Measurement, Room, Floor
from .officegraph import OfficeGraph


class OfficeGraphExtractor:
    """
    A class that extracts all relevant objects (rooms, floors, devices, measurements, etc.)
    from an OfficeGraph instance.
    """
    
    def __init__(self, office_graph: OfficeGraph):
        """
        Args:
            office_graph: The OfficeGraph instance containing the RDF graph.
        """
        self.office_graph = office_graph
        
    def extract_rooms_and_floors(self) -> None:
        """
        Extract all rooms and floors from the graph.
        Also derive which buildings those floors belong to.
        Updates the office_graph's rooms, floors, and buildings collections.
        """
        # Clear existing data
        self.office_graph.rooms.clear()
        self.office_graph.floors.clear()
        self.office_graph.buildings.clear()
        
        # 1) Extract rooms
        room_query = """
        SELECT ?room ?comment ?floor
        WHERE {
            ?room a <https://saref.etsi.org/saref4bldg/BuildingSpace> .
            OPTIONAL { ?room <http://www.w3.org/2000/01/rdf-schema#comment> ?comment . }
            OPTIONAL { ?room <https://saref.etsi.org/saref4bldg/isSpaceOf> ?floor . }
        }
        """
        
        for row in self.office_graph.graph.query(room_query):
            room_uri = row.room
            comment = str(row.comment) if row.comment else None
            floor_uri = row.floor if row.floor else None
            
            # Attempt to parse out "room_number" from the URI (if it has "roomname_")
            room_number = None
            room_type = None
            
            if comment:
                room_type = comment  # e.g. "room" or "support_zone"
                
            room_str = str(room_uri)
            if "roomname_" in room_str:
                try:
                    room_number = room_str.split("roomname_")[-1]
                except:
                    pass
                    
            # Build the Room dataclass
            room_obj = Room(
                uri=room_uri,
                name=None,
                floor=floor_uri,
                room_number=room_number,
                room_type=room_type
            )
            
            self.office_graph.rooms[room_uri] = room_obj
            
            # 2) Add room to the associated Floor (if any)
            if floor_uri not in (None, ""):
                # Attempt to parse a floor number if possible
                if floor_uri not in self.office_graph.floors:
                    possible_num = None
                    floor_str = str(floor_uri)
                    if "floor_" in floor_str:
                        try:
                            possible_num = int(floor_str.split("floor_")[-1])
                        except ValueError:
                            pass
                            
                    self.office_graph.floors[floor_uri] = Floor(
                        uri=floor_uri,
                        floor_number=possible_num
                    )
                    
                # Add room to floor
                self.office_graph.floors[floor_uri].add_room(room_uri)
                
        # 3) Extract floor-building relationships
        building_query = """
        SELECT ?floor ?building
        WHERE {
            ?floor <https://saref.etsi.org/saref4bldg/isSpaceOf> ?building .
            ?building a <https://saref.etsi.org/saref4bldg/Building> .
        }
        """
        
        # Use a temporary defaultdict to collect floors by building
        buildings = defaultdict(set)
        
        for row in self.office_graph.graph.query(building_query):
            floor_uri = row.floor
            building_uri = row.building
            
            if floor_uri in self.office_graph.floors:
                # Update Floor object
                self.office_graph.floors[floor_uri].building = building_uri
                # Add floor to building
                buildings[building_uri].add(floor_uri)
                
        # Convert defaultdict to regular dict and store in office_graph
        self.office_graph.buildings = dict(buildings)
        
        logger.info("Extracted %d rooms, %d floors, %d building URIs",
                   len(self.office_graph.rooms), 
                   len(self.office_graph.floors), 
                   len(self.office_graph.buildings))
    
    def extract_devices(self) -> None:
        """
        Extract device info from the graph, link them to rooms/floors/buildings.
        Updates the office_graph's devices collection.
        """
        # Clear existing devices
        self.office_graph.devices.clear()
        
        # Query for devices
        query = """
        SELECT DISTINCT ?device ?manufacturer ?model ?deviceType
        WHERE {
            ?device a ?deviceClass .
            ?deviceClass rdfs:subClassOf* <https://saref.etsi.org/core/Device> .
            OPTIONAL { ?device <https://saref.etsi.org/core/hasManufacturer> ?manufacturer . }
            OPTIONAL { ?device <https://saref.etsi.org/core/hasModel> ?model . }
            OPTIONAL { ?device <https://interconnectproject.eu/example/hasDeviceType> ?deviceType . }
        }
        """
        
        for row in self.office_graph.graph.query(query):
            dev_uri = row.device
            manufacturer = str(row.manufacturer) if row.manufacturer else None
            model = str(row.model) if row.model else None
            device_type = str(row.deviceType) if row.deviceType else None
            
            dev_obj = Device(
                uri=dev_uri,
                manufacturer=manufacturer,
                model=model,
                device_type=device_type
            )
            self.office_graph.devices[dev_uri] = dev_obj
            
        # Query for device-room-floor-building relationships
        room_query = """
        SELECT ?device ?room ?floor ?building
        WHERE {
            ?device <https://saref.etsi.org/saref4bldg/isContainedIn> ?room .
            OPTIONAL { ?room <https://saref.etsi.org/saref4bldg/isSpaceOf> ?floor . }
            OPTIONAL { ?floor <https://saref.etsi.org/saref4bldg/isSpaceOf> ?building . }
        }
        """
        
        for row in self.office_graph.graph.query(room_query):
            dev_uri = row.device
            room_uri = row.room
            floor_uri = row.floor if row.floor else None
            building_uri = row.building if row.building else None
            
            if dev_uri in self.office_graph.devices:
                device = self.office_graph.devices[dev_uri]
                device.room = room_uri
                device.floor = floor_uri
                device.building = building_uri
                
                # Also link the device to the Room object itself if it exists
                if room_uri in self.office_graph.rooms:
                    # Add device to room
                    self.office_graph.rooms[room_uri].add_device(dev_uri)
                    
        logger.info("Extracted %d devices", len(self.office_graph.devices))
    
    def extract_measurements(self) -> None:
        """
        Extract measurement information and attach to devices.
        Updates the office_graph's measurements collection.
        """
        # Clear existing measurements
        self.office_graph.measurements.clear()
        
        # Query for measurements
        query = """
        SELECT ?device ?meas ?timestamp ?value ?unit ?property
        WHERE {
            ?device <https://saref.etsi.org/core/makesMeasurement> ?meas .
            ?meas <https://saref.etsi.org/core/hasTimestamp> ?timestamp .
            ?meas <https://saref.etsi.org/core/hasValue> ?value .
            OPTIONAL { ?meas <https://saref.etsi.org/core/isMeasuredIn> ?unit . }
            OPTIONAL { ?meas <https://saref.etsi.org/core/relatesToProperty> ?property . }
        }
        """
        
        for row in self.office_graph.graph.query(query):
            device_uri = row.device
            meas_uri = row.meas
            # Convert timestamp to Python datetime if needed
            timestamp = row.timestamp.toPython() if hasattr(row.timestamp, 'toPython') else row.timestamp
            value = float(row.value)
            unit = row.unit if row.unit else None
            property_type = row.property if row.property else None
            
            meas_obj = Measurement(
                meas_uri=meas_uri,
                device_uri=device_uri,
                timestamp=timestamp,
                value=value,
                unit=unit,
                property_type=property_type
            )
            
            self.office_graph.measurements[meas_uri] = meas_obj
            
            # Attach to device
            if device_uri in self.office_graph.devices:
                self.office_graph.devices[device_uri].add_measurement(meas_obj)
                
        # Query for measurement sequences
        seq_query = """
        SELECT ?meas ?next
        WHERE {
            ?meas <https://interconnectproject.eu/example/next_node> ?next .
        }
        """
        
        for row in self.office_graph.graph.query(seq_query):
            meas_uri = row.meas
            next_uri = row.next
            
            # Link measurements
            if meas_uri in self.office_graph.measurements and next_uri in self.office_graph.measurements:
                self.office_graph.measurements[meas_uri].next_meas_uri = next_uri
                self.office_graph.measurements[next_uri].prev_meas_uri = meas_uri
                
        logger.info("Extracted %d measurements", len(self.office_graph.measurements))
    
    def extract_property_types(self) -> None:
        """
        Extract property type resources and their labels.
        Updates the office_graph's property_types collection.
        """
        # Clear existing property types
        self.office_graph.property_types.clear()
        
        # Query for property types
        query = """
        SELECT ?property ?label
        WHERE {
            ?property a ?type .
            OPTIONAL { ?property rdfs:label ?label . }
            FILTER(
                STRSTARTS(STR(?type), "https://saref.etsi.org/core/") ||
                STRSTARTS(STR(?type), "https://interconnectproject.eu/example/")
            )
        }
        """
        
        for row in self.office_graph.graph.query(query):
            prop_uri = row.property
            # Use label if available, otherwise extract from URI
            label = str(row.label) if row.label else str(prop_uri).split("/")[-1]
            self.office_graph.property_types[prop_uri] = label
            
        logger.info("Extracted %d property types", len(self.office_graph.property_types))
        
    def build_measurement_sequences(self) -> None:
        """
        Build time-ordered measurement sequences by device and property type.
        Updates the office_graph's measurement_sequences collection.
        """
        # Clear existing sequences
        self.office_graph.measurement_sequences.clear()
        
        # Group measurements by (device, property_type)
        sequences = defaultdict(list)
        
        for meas_obj in self.office_graph.measurements.values():
            if meas_obj.property_type:  # Only include measurements with property type
                key = (meas_obj.device_uri, meas_obj.property_type)
                sequences[key].append(meas_obj)
                
        # Sort each sequence by timestamp
        for key, meas_list in sequences.items():
            meas_list.sort(key=lambda m: m.timestamp)
            self.office_graph.measurement_sequences[key] = meas_list
            
        logger.info("Built %d measurement sequences", len(self.office_graph.measurement_sequences))
