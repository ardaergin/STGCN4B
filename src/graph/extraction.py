import logging
import sys
from collections import defaultdict
from rdflib import URIRef
from typing import Dict, Tuple, List

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from ..core import Device, Measurement, Room, Floor

class OfficeGraphExtractor:
    """A mixin class that provides extraction methods for OfficeGraph instances."""
    
    def extract_rooms_and_floors(self) -> None:
        """
        Extract rooms and floors from the graph and establish their relationships.
        Updates the office_graph's rooms and floors collections.
        Processes spatial data including WKT polygons (both geo and document), altitude.
        Also extracts CSV enrichment data: isProperRoom, hasWindows, and hasWindowsFacingDirection.
        """
        # 0) Clear existing data
        self.rooms.clear()
        self.floors.clear()

        # 1) Extract all floors
        floor_query = """
        PREFIX s4bldg: <https://saref.etsi.org/saref4bldg/>
        PREFIX ic:     <https://interconnectproject.eu/example/>

        SELECT DISTINCT ?floor
        WHERE {
            ?room s4bldg:isSpaceOf ?floor .
            FILTER(STRSTARTS(STR(?floor), STR(ic:VL_floor_)))
        }
        """
        for row in self.graph.query(floor_query):
            floor_uri = row.floor
            floor_number = None
            try:
                floor_number = int(str(floor_uri).split("VL_floor_")[-1])
            except:
                pass

            floor_obj = Floor(uri=floor_uri, floor_number=floor_number)
            self.floors[floor_uri] = floor_obj

        logger.info("Extracted %d floors", len(self.floors))

        # 2) Extract basic room data (without instantiating Room yet)
        room_query = """
        PREFIX s4bldg:   <https://saref.etsi.org/saref4bldg/>
        PREFIX rdfs:     <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX geo1:     <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX ex:       <https://example.org/>
        PREFIX ex-ont:   <https://example.org/ontology#>

        SELECT ?room ?floor ?comment ?label ?altitude ?isProperRoom
        WHERE {
            ?room a s4bldg:BuildingSpace .
            ?room s4bldg:isSpaceOf ?floor .
            OPTIONAL { ?room rdfs:comment ?comment. }
            OPTIONAL { ?room rdfs:label   ?label.   }
            OPTIONAL { ?room geo1:alt     ?altitude.}
            OPTIONAL { ?room ex-ont:isProperRoom    ?isProperRoom.  }
            FILTER NOT EXISTS { ?other s4bldg:isSpaceOf ?room. }
        }
        """
        room_data: Dict[URIRef, Dict] = {}
        for row in self.graph.query(room_query):
            room_uri     = row.room
            floor_uri    = row.floor
            comment      = str(row.comment)  if row.comment else None
            room_number  = str(row.label)    if row.label   else None
            altitude     = float(row.altitude) if row.altitude else None
            is_room_val  = bool(row.isProperRoom)  if row.isProperRoom is not None else None

            if not room_number and "roomname_" in str(room_uri):
                try:
                    room_number = str(room_uri).split("roomname_")[-1]
                except:
                    pass

            room_data[room_uri] = {
                'floor_uri':           floor_uri,
                'is_support_zone':     (comment == "support_zone"),
                'room_number':         room_number,
                'altitude':            altitude,
                'isProperRoom':        is_room_val,
                'geo_wkt_polygon':     None,
                'doc_wkt_polygon':     None,
                'window_headings':     [],
                'relative_directions': []
            }

        # 3) Extract geo/document polygons
        geo_query = """
        PREFIX geosparql: <http://www.opengis.net/ont/geosparql#>
        SELECT ?room ?wkt WHERE {
        ?room geosparql:hasGeometry ?g .
        ?g    geosparql:asWKT      ?wkt .
        }
        """
        for row in self.graph.query(geo_query):
            if row.room in room_data:
                room_data[row.room]['geo_wkt_polygon'] = str(row.wkt)

        doc_query = """
        PREFIX geosparql: <http://www.opengis.net/ont/geosparql#>
        PREFIX ex-ont:   <https://example.org/ontology#>
        SELECT ?room ?wkt WHERE {
        ?room ex-ont:hasDocumentGeometry ?g .
        ?g    geosparql:asWKT            ?wkt .
        }
        """
        for row in self.graph.query(doc_query):
            if row.room in room_data:
                room_data[row.room]['doc_wkt_polygon'] = str(row.wkt)

        # 4) Extract window headings & relative directions
        window_query = """
        PREFIX ex-ont: <https://example.org/ontology#>
        SELECT ?room ?hasFacingDirection ?facingRelativeDirection WHERE {
            ?room   ex-ont:hasWindow            ?win .
            ?win    ex-ont:hasFacingDirection   ?hasFacingDirection .
            ?win    ex-ont:facingRelativeDirection ?facingRelativeDirection .
        }
        """
        room_windows = defaultdict(list)
        room_relative_directions = defaultdict(list)
        for row in self.graph.query(window_query):
            room = row.room
            room_windows[room].append(int(row.hasFacingDirection))
            room_relative_directions[room].append(str(row.facingRelativeDirection))

        # 5) Merge those two separately
        for uri, headings in room_windows.items():
            if uri in room_data:
                room_data[uri]['window_headings'] = headings

        for uri, rel_dirs in room_relative_directions.items():
            if uri in room_data:
                room_data[uri]['relative_directions'] = rel_dirs

        # 6) Instantiate Room objects
        for room_uri, data in room_data.items():
            headings = data['window_headings']
            rel_dirs = data['relative_directions']

            room_obj = Room(
                uri                         = room_uri,
                room_number                 = data['room_number'],
                is_support_zone             = data['is_support_zone'],
                floor                       = data['floor_uri'],
                altitude                    = data['altitude'],
                isProperRoom                = data['isProperRoom'],
                geo_wkt_polygon             = data['geo_wkt_polygon'],
                doc_wkt_polygon             = data['doc_wkt_polygon'],
                hasWindows                  = bool(headings),
                hasWindowsFacingDirection   = headings,
                hasWindowsRelativeDirection = rel_dirs
            )

            self.rooms[room_uri] = room_obj
            if data['floor_uri'] in self.floors:
                self.floors[data['floor_uri']].add_room(room_uri)

        logger.info("Extracted %d rooms", len(self.rooms))
    
    def extract_devices(self) -> None:
        """
        Extract device info from the graph and link them to rooms.
        Updates the office_graph's devices collection.
        """
        # Clear existing devices
        self.devices.clear()
        
        # Query for devices and their rooms
        query = """
        PREFIX s4ener: <https://saref.etsi.org/saref4ener/>
        PREFIX saref: <https://saref.etsi.org/core/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX s4bldg: <https://saref.etsi.org/saref4bldg/>
        PREFIX ic: <https://interconnectproject.eu/example/>

        SELECT DISTINCT ?device ?room ?manufacturer ?model ?deviceType
        WHERE {
            ?device a s4ener:Device .
            OPTIONAL { ?device s4bldg:isContainedIn ?room . }
            OPTIONAL { ?device saref:hasManufacturer ?manufacturer . }
            OPTIONAL { ?device saref:hasModel ?model . }
            OPTIONAL { ?device ic:hasDeviceType ?deviceType . }
        }
        """
        
        for row in self.graph.query(query):
            dev_uri = row.device
            room_uri = row.room if row.room else None
            manufacturer = str(row.manufacturer) if row.manufacturer else None
            model = str(row.model) if row.model else None
            device_type = str(row.deviceType) if row.deviceType else None
            
            # Create the Device object with direct reference to its room
            dev_obj = Device(
                uri=dev_uri,
                manufacturer=manufacturer,
                model=model,
                device_type=device_type,
                room=room_uri
            )
            
            self.devices[dev_uri] = dev_obj
            
            # Link the room to this device
            if room_uri and room_uri in self.rooms:
                self.rooms[room_uri].add_device(dev_uri)
                            
        logger.info("Extracted %d devices", len(self.devices))
    
    def extract_measurements(self) -> None:
        """
        Extract measurement information and attach to devices.
        Updates the office_graph's measurements collection.
        """
        # Clear existing measurements
        self.measurements.clear()
        
        # Query for measurements
        query = """
        PREFIX saref: <https://saref.etsi.org/core/>

        SELECT ?device ?meas ?timestamp ?value ?unit ?property
        WHERE {
            ?device saref:makesMeasurement ?meas .
            ?meas saref:hasTimestamp ?timestamp .
            ?meas saref:hasValue ?value .
            OPTIONAL { ?meas saref:isMeasuredIn ?unit . }
            OPTIONAL { ?meas saref:relatesToProperty ?property . }
        }
        """
        
        for row in self.graph.query(query):
            device_uri = row.device
            meas_uri = row.meas
            timestamp = row.timestamp.toPython() if hasattr(row.timestamp, 'toPython') else row.timestamp
            value = float(row.value)
            unit = row.unit if row.unit else None
            property_type = row.property if row.property else None

            # Create the Measurement object
            meas_obj = Measurement(
                meas_uri=meas_uri,
                device_uri=device_uri,
                timestamp=timestamp,
                value=value,
                unit=unit,
                property_type=property_type
            )

            # Add to the measurements collection
            self.measurements[meas_uri] = meas_obj

            # Add to the appropriate device (this also adds to device.properties and device.measurements_by_property)
            if device_uri in self.devices:
                self.devices[device_uri].add_measurement(meas_obj)
                                
        logger.info("Extracted %d measurements", len(self.measurements))
        
    def build_measurement_sequences(self) -> None:
        """
        Build time-ordered measurement sequences by device and property type.
        Updates the office_graph's measurement_sequences collection.
        """
        # Clear existing sequences
        self.measurement_sequences.clear()
        
        # Group measurements by (device, property_type)
        sequences: Dict[Tuple[URIRef, URIRef], list] = defaultdict(list)
        
        for meas_obj in self.measurements.values():
            if meas_obj.property_type:  # Only include measurements with property type
                key = (meas_obj.device_uri, meas_obj.property_type)
                sequences[key].append(meas_obj)
                
        # Sort each sequence by timestamp
        for key, meas_list in sequences.items():
            meas_list.sort(key=lambda m: m.timestamp)            
            self.measurement_sequences[key] = meas_list
            
        logger.info("Built %d measurement sequences", len(self.measurement_sequences))

    def extract_property_type_mappings(self) -> None:
        """
        Extract mappings from specific property URIs to their general property types.
        Updates the office_graph's property_type_mappings collection.
        """
        # Clear existing mappings
        self.property_type_mappings.clear()
        
        # Query for properties and their types
        query = """
        PREFIX ic: <https://interconnectproject.eu/example/>

        SELECT ?property ?propertyType
        WHERE {
            ?property a ?propertyType .
            FILTER(STRSTARTS(STR(?property), STR(ic:property_)))
        }
        """
        
        # Dictionary to track all property URIs for each property type
        type_to_properties = defaultdict(list)
        
        for row in self.graph.query(query):
            property_uri = row.property
            property_type_uri = row.propertyType
            
            # Extract the short name of the property type (e.g., "Temperature" from "saref:Temperature")
            property_type_str = str(property_type_uri)
            # Get the part after the last '/' or '#' (if it exists)
            type_name = property_type_str.split('/')[-1].split('#')[-1]
            
            # Add the property URI to the list for this property type
            type_to_properties[type_name].append(property_uri)
        
        # Update the office_graph's property_type_mappings
        self.property_type_mappings = dict(type_to_properties)
        
        logger.info("Extracted mappings for %d property types", len(self.property_type_mappings))
        
        # Optionally, log some stats about the mappings
        for type_name, properties in self.property_type_mappings.items():
            logger.info("  - %s: %d properties", type_name, len(properties))
    
    def extract_all(self) -> None:
        """
        Convenience method to run all extraction methods in the correct order.
        """
        self.extract_rooms_and_floors()
        self.extract_devices()
        self.extract_measurements()
        self.build_measurement_sequences()
        self.extract_property_type_mappings()