import os, sys, pickle, argparse
from collections import defaultdict
import logging
from rdflib import URIRef
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

from ..core import Device, Measurement, Room, Floor
from .officegraph import OfficeGraph

class OfficeGraphExtractor:
    """A class that extracts all relevant objects from an OfficeGraph instance."""
    
    def __init__(self, office_graph: OfficeGraph):
        """
        Args:
            office_graph: The OfficeGraph instance containing the RDF graph.
        """
        self.office_graph = office_graph
    
    def extract_rooms_and_floors(self) -> None:
        """
        Extract rooms and floors from the graph and establish their relationships.
        Updates the office_graph's rooms and floors collections.
        Processes spatial data including WKT polygons, metric area, and altitude.
        Also extracts CSV enrichment data: isRoom, hasWindows, and hasWindowsFacingDirection.
        """
        # Clear existing data
        self.office_graph.rooms.clear()
        self.office_graph.floors.clear()

        # First, extract all floors
        floor_query = """
        PREFIX s4bldg: <https://saref.etsi.org/saref4bldg/>
        PREFIX ic: <https://interconnectproject.eu/example/>

        SELECT DISTINCT ?floor
        WHERE {
            ?room s4bldg:isSpaceOf ?floor .
            FILTER(STRSTARTS(STR(?floor), STR(ic:VL_floor_)))
        }
        """
        
        for row in self.office_graph.graph.query(floor_query):
            floor_uri = row.floor
            
            # Extract the floor number from the URI
            floor_number = None
            floor_str = str(floor_uri)
            if "VL_floor_" in floor_str:
                try:
                    floor_number = int(floor_str.split("VL_floor_")[-1])
                except Exception:
                    pass
            
            # Create the Floor object
            floor_obj = Floor(
                uri=floor_uri,
                floor_number=floor_number
            )
            
            self.office_graph.floors[floor_uri] = floor_obj
        
        logger.info("Extracted %d floors", len(self.office_graph.floors))
        
        # Now, extract all rooms with their spatial data and link them to floors
        room_query = """
        PREFIX s4bldg: <https://saref.etsi.org/saref4bldg/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX geosparql: <http://www.opengis.net/ont/geosparql#>
        PREFIX bot: <https://w3id.org/bot#>
        PREFIX geo1: <http://www.w3.org/2003/01/geo/wgs84_pos#>
        PREFIX ex: <https://example.org/>
        PREFIX ex-ont: <https://example.org/ontology#>

        SELECT ?room ?floor ?comment ?label ?wkt_polygon ?metric_area ?altitude ?isRoom 
        WHERE {
            ?room a s4bldg:BuildingSpace .
            ?room s4bldg:isSpaceOf ?floor .
            OPTIONAL { ?room rdfs:comment ?comment . }
            OPTIONAL { ?room rdfs:label ?label . }
            
            # Spatial data from VideoLab topology
            OPTIONAL { 
                ?room geosparql:hasGeometry ?geometry .
                ?geometry geosparql:asWKT ?wkt_polygon .
            }
            OPTIONAL { ?room geosparql:hasMetricArea ?metric_area . }
            OPTIONAL { ?room geo1:alt ?altitude . }
            
            # CSV enrichment data
            OPTIONAL { ?room ex:isRoom ?isRoom . }
            
            # Exclude floors themselves
            FILTER NOT EXISTS { ?otherRoom s4bldg:isSpaceOf ?room }
        }
        """

        # Create room objects first
        for row in self.office_graph.graph.query(room_query):
            room_uri = row.room
            floor_uri = row.floor
            comment = str(row.comment) if row.comment else None
            room_number = str(row.label) if row.label else None
            wkt_polygon = str(row.wkt_polygon) if row.wkt_polygon else None
            metric_area = float(row.metric_area) if row.metric_area else None
            altitude = float(row.altitude) if row.altitude else None
            is_room_value = bool(row.isRoom) if row.isRoom is not None else None

            # If room_number is not in the label, try to extract from URI
            if not room_number:
                room_str = str(room_uri)
                if "roomname_" in room_str:
                    try:
                        room_number = room_str.split("roomname_")[-1]
                    except Exception:
                        pass

            # Determine if it's a support zone
            is_support_zone = (comment == "support_zone")

            # Create the Room object with direct reference to its floor
            room_obj = Room(
                uri=room_uri,
                room_number=room_number,
                is_support_zone=is_support_zone,
                floor=floor_uri,
                wkt_polygon=wkt_polygon,
                metric_area=metric_area,
                altitude=altitude,
                isRoom=is_room_value  # Add the isRoom value from CSV enrichment
            )

            # The Room.__post_init__ method will process the WKT polygon data automatically
            # and calculate all derived spatial properties (centroid, perimeter, etc.)

            self.office_graph.rooms[room_uri] = room_obj
            
            # Link the floor to this room
            if floor_uri in self.office_graph.floors:
                self.office_graph.floors[floor_uri].add_room(room_uri)

        logger.info("Extracted %d rooms", len(self.office_graph.rooms))
        
        # Now extract window information for the rooms
        window_query = """
        PREFIX ex: <https://example.org/>
        PREFIX ex-ont: <https://example.org/ontology#>
        
        SELECT ?room ?window ?facingDirection ?hasFacingDirection
        WHERE {
            ?room ex-ont:hasWindow ?window .
            ?window ex-ont:facingRelativeDirection ?facingDirection .
            ?window ex-ont:hasFacingDirection ?hasFacingDirection .
        }
        """
        
        # Dictionary to track windows by room
        room_windows = {}
        
        for row in self.office_graph.graph.query(window_query):
            room_uri = row.room
            window_uri = row.window
            facing_direction = str(row.facingDirection)
            heading = int(row.hasFacingDirection)
            
            # Initialize entry for this room if not already done
            if room_uri not in room_windows:
                room_windows[room_uri] = []
                
            # Add window direction to the room's list
            room_windows[room_uri].append(heading)
        
        # Update room objects with window information
        for room_uri, headings in room_windows.items():
            if room_uri in self.office_graph.rooms:
                room_obj = self.office_graph.rooms[room_uri]
                room_obj.hasWindows = True
                room_obj.hasWindowsFacingDirection = headings
        
        # Set hasWindows=False for rooms with no windows found
        for room_uri, room_obj in self.office_graph.rooms.items():
            if room_obj.hasWindows is None:
                room_obj.hasWindows = False
    
    def extract_devices(self) -> None:
        """
        Extract device info from the graph and link them to rooms.
        Updates the office_graph's devices collection.
        """
        # Clear existing devices
        self.office_graph.devices.clear()
        
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
        
        for row in self.office_graph.graph.query(query):
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
            
            self.office_graph.devices[dev_uri] = dev_obj
            
            # Link the room to this device
            if room_uri and room_uri in self.office_graph.rooms:
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
        
        for row in self.office_graph.graph.query(query):
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
            self.office_graph.measurements[meas_uri] = meas_obj

            # Add to the appropriate device (this also adds to device.properties and device.measurements_by_property)
            if device_uri in self.office_graph.devices:
                self.office_graph.devices[device_uri].add_measurement(meas_obj)
                                
        logger.info("Extracted %d measurements", len(self.office_graph.measurements))
        
    def build_measurement_sequences(self) -> None:
        """
        Build time-ordered measurement sequences by device and property type.
        Updates the office_graph's measurement_sequences collection.
        """
        # Clear existing sequences
        self.office_graph.measurement_sequences.clear()
        
        # Group measurements by (device, property_type)
        sequences: Dict[Tuple[URIRef, URIRef], list] = defaultdict(list)
        
        for meas_obj in self.office_graph.measurements.values():
            if meas_obj.property_type:  # Only include measurements with property type
                key = (meas_obj.device_uri, meas_obj.property_type)
                sequences[key].append(meas_obj)
                
        # Sort each sequence by timestamp
        for key, meas_list in sequences.items():
            meas_list.sort(key=lambda m: m.timestamp)            
            self.office_graph.measurement_sequences[key] = meas_list
            
        logger.info("Built %d measurement sequences", len(self.office_graph.measurement_sequences))

    def extract_property_type_mappings(self) -> None:
        """
        Extract mappings from specific property URIs to their general property types.
        Updates the office_graph's property_type_mappings collection.
        """
        # Clear existing mappings
        self.office_graph.property_type_mappings.clear()
        
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
        
        for row in self.office_graph.graph.query(query):
            property_uri = row.property
            property_type_uri = row.propertyType
            
            # Extract the short name of the property type (e.g., "Temperature" from "saref:Temperature")
            property_type_str = str(property_type_uri)
            # Get the part after the last '/' or '#' (if it exists)
            type_name = property_type_str.split('/')[-1].split('#')[-1]
            
            # Add the property URI to the list for this property type
            type_to_properties[type_name].append(property_uri)
        
        # Update the office_graph's property_type_mappings
        self.office_graph.property_type_mappings = dict(type_to_properties)
        
        logger.info("Extracted mappings for %d property types", len(self.office_graph.property_type_mappings))
        
        # Optionally, print some stats about the mappings
        for type_name, properties in self.office_graph.property_type_mappings.items():
            logger.info("  - %s: %d properties", type_name, len(properties))

if __name__ == "__main__":
    
    ### Arguments ###
    parser = argparse.ArgumentParser(description="Load and inspect the IoT Office Graph")
    parser.add_argument(
        "--pkl-path", 
        type=str,
        default="data/processed/officegraph_base.pkl",
        help="File path for OfficeGraph pickle (default: data/processed/officegraph_base.pkl)"
    )
    parser.add_argument(
        "--output-path", 
        type=str,
        default="data/processed/officegraph_extracted.pkl",
        help="File path for saving the updated OfficeGraph pickle (default: data/processed/officegraph_extracted.pkl)"
    )
    args = parser.parse_args()

    ### Running ###
    
    # Load OfficeGraph pickle
    print(f"Loading OfficeGraph with the file path: '{args.pkl_path}'")
    with open(args.pkl_path, 'rb') as f:
        office_graph = pickle.load(f)
    print(f"OfficeGraph loaded successfully.")

    # Initialize the extractor
    office_graph.extractor = OfficeGraphExtractor(office_graph)

    # 1) Rooms & Floors
    office_graph.extractor.extract_rooms_and_floors()
    # 2) Devices
    office_graph.extractor.extract_devices()
    # 3) Measurements
    office_graph.extractor.extract_measurements()
    # 4) Measurement sequences
    office_graph.extractor.build_measurement_sequences()
    # 5) Property type mappings
    office_graph.extractor.extract_property_type_mappings()

    ### Saving ###
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    try:
        with open(args.output_path, "wb") as f:
            pickle.dump(office_graph, f)
        print(f"Successfully saved office graph to {args.output_path}")
    except (OSError, pickle.PicklingError) as e:
        logger.error("Failed to save office graph: %s", e)
        sys.exit(1)
