from collections import defaultdict

import logging
logger = logging.getLogger(__name__)

from ..core import Device, Measurement, Room
from .officegraph import OfficeGraph


class OfficeGraphExtractor:
    """A class that extracts all relevant objects from an OfficeGraph instance."""
    
    def __init__(self, office_graph: OfficeGraph):
        """
        Args:
            office_graph: The OfficeGraph instance containing the RDF graph.
        """
        self.office_graph = office_graph
        
    def extract_rooms(self) -> None:
        """
        Extract rooms from the graph (without linking them to floors or buildings).
        Updates the office_graph's rooms collection.
        """
        # Clear existing data
        self.office_graph.rooms.clear()

        # Query to extract all building spaces with optional comment (e.g., "room" or "support_zone")
        room_query = """
        PREFIX s4bldg: <https://saref.etsi.org/saref4bldg/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?room ?comment
        WHERE {
            ?room a s4bldg:BuildingSpace .
            OPTIONAL { ?room rdfs:comment ?comment . }
        }
        """

        for row in self.office_graph.graph.query(room_query):
            room_uri = row.room
            comment = str(row.comment) if row.comment else None

            # Extract room number from URI, if present
            room_number = None
            room_str = str(room_uri)
            if "roomname_" in room_str:
                try:
                    room_number = room_str.split("roomname_")[-1]
                except Exception:
                    pass

            # Determine if it's a support zone
            is_support_zone = (comment == "support_zone")

            room_obj = Room(
                uri=room_uri,
                room_number=room_number,
                is_support_zone=is_support_zone
            )

            self.office_graph.rooms[room_uri] = room_obj

        logger.info("Extracted %d rooms", len(self.office_graph.rooms))
            
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
