import os
import re
from typing import List, Set
from rdflib import Graph, Namespace, RDF

from ...config.namespaces import NamespaceMixin


class FloorDeviceRetriever(NamespaceMixin):
    """
    A helper class to load the device map, parse an enrichment TTL,
    and query for devices exclusively on a specified floor (excluding support_zone).
    It then maps the query results back to the local TTL file names.
    """

    def __init__(self,
                 devices_root_dir: str = "data/devices",
                 enrichment_ttl_path: str = "data/enrichments/devices_in_rooms_enrichment.ttl"):
        """
        :param devices_root_dir: Root path where device TTL files live.
        :param enrichment_ttl_path: Path to the 'devices_in_rooms_enrichment.ttl' file.
        """
        self.devices_root_dir = devices_root_dir
        self.enrichment_ttl_path = enrichment_ttl_path

        # Step 1: Build the file_device_map from all TTL files
        self.file_device_map = self._build_file_device_map()

        # Step 2: Parse the enrichment graph
        self.graph = Graph()
        self.graph.parse(self.enrichment_ttl_path, format="turtle")

        # Step 3: Build the reverse lookup from expanded URI -> (filename, short_id)
        self.reverse_lookup = self._build_reverse_lookup()

    def _build_file_device_map(self) -> dict:
        """
        Crawls 'self.devices_root_dir' for .ttl files,
        extracts the first device URI (ic:...) and the optional parent serial,
        storing them in {filename: {identifier1, identifier2, ...}}.
        """
        file_device_map = {}

        for subdir, _, files in os.walk(self.devices_root_dir):
            for file in files:
                if file.endswith(".ttl"):
                    filepath = os.path.join(subdir, file)
                    device_uri = None
                    parent_serial = None
                    # Read top ~20 lines to find device definition
                    with open(filepath, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i >= 20:
                                break
                            # Match something like 'ic:SmartSense_Motion_Sensor_6 a s4ener:Device'
                            if device_uri is None:
                                m = re.match(r"^(ic:[^\s]+)\s+a\s+s4ener:Device", line)
                                if m:
                                    device_uri = m.group(1)
                            # Match 'ic:hasParentSerialNumber "..."'
                            if parent_serial is None:
                                m = re.search(r'ic:hasParentSerialNumber\s+"([^"]+)"', line)
                                if m:
                                    parent_serial = m.group(1)
                            if device_uri and parent_serial:
                                break
                    identifiers = set()
                    if device_uri:
                        identifiers.add(device_uri)
                    if parent_serial:
                        identifiers.add(parent_serial)
                    if identifiers:
                        file_device_map[file] = identifiers

        return file_device_map

    @staticmethod
    def _to_full_uri(short_id: str) -> str:
        """
        Converts a short ID like:
           ic:Zigbee_Thermostat_9    => https://interconnectproject.eu/example/Zigbee_Thermostat_9
           ic:device_C2FA1A         => https://interconnectproject.eu/example/device_C2FA1A
           urn:Device:SmartThings:... => https://interconnectproject.eu/example/device_urn:Device:SmartThings:...
        """
        if short_id.startswith("ic:"):
            return "https://interconnectproject.eu/example/" + short_id[len("ic:"):]
        elif short_id.startswith("urn:Device:SmartThings:"):
            return "https://interconnectproject.eu/example/device_" + short_id
        else:
            # fallback if pattern changes
            return short_id

    def _build_reverse_lookup(self) -> dict:
        """
        Builds a dict from expandedURI -> list of (filename, short_id).
        For instance:
          "https://interconnectproject.eu/example/Zigbee_Thermostat_9" -> [
              ("urn_Device_SmartThings_9405019c...", "ic:Zigbee_Thermostat_9"),
              ...
          ]
        """
        reverse_lookup = {}
        for ttl_filename, short_ids in self.file_device_map.items():
            for sid in short_ids:
                expanded = self._to_full_uri(sid)
                if expanded not in reverse_lookup:
                    reverse_lookup[expanded] = []
                reverse_lookup[expanded].append((ttl_filename, sid))
        return reverse_lookup

    def get_device_room_floor_graph(self, floor_number: int = 7) -> Graph:
        """
        Returns an RDF graph containing device-room-floor relationship triples for the specified floor.
        
        :param floor_number: The floor number to query (e.g., 1, 2, 7)
        :return: RDFLib Graph containing device-room-floor relationship triples
        """
        # Build the floor URI
        floor_uri = f"https://interconnectproject.eu/example/VL_floor_{floor_number}"
        
        # CONSTRUCT query to directly create the relationship triples
        construct_query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX ic: <https://interconnectproject.eu/example/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX s4bldg: <https://saref.etsi.org/saref4bldg/>
        
        CONSTRUCT {{
            ?device s4bldg:isContainedIn ?buildingSpace .
            ?buildingSpace s4bldg:contains ?device .
            ?buildingSpace s4bldg:isSpaceOf ?floor .
            ?buildingSpace rdf:type s4bldg:BuildingSpace .
            ?buildingSpace rdfs:comment ?comment .
        }}
        WHERE {{
            ?buildingSpace a s4bldg:BuildingSpace ;
                        s4bldg:contains ?device ;
                        s4bldg:isSpaceOf ?floor .
            OPTIONAL {{ ?buildingSpace rdfs:comment ?comment }}

            # Must be on desired floor
            FILTER (?floor = <{floor_uri}>)

            # Exclude "support_zone"
            FILTER (!BOUND(?comment) || ?comment != "support_zone")

            # Ensure device is not contained in buildingSpaces on other floors
            FILTER NOT EXISTS {{
                ?otherBS s4bldg:contains ?device ;
                        s4bldg:isSpaceOf ?otherFloor .
                FILTER (?otherFloor != <{floor_uri}>)
            }}
        }}
        """
        
        # Execute the CONSTRUCT query
        relationship_graph = self.graph.query(construct_query).graph
        
        # Bind namespaces for nicer serialization
        relationship_graph.bind("ic", self.IC)
        relationship_graph.bind("rdfs", self.RDFS)
        relationship_graph.bind("s4bldg", self.S4BLDG)
        relationship_graph.bind("rdf", RDF)
        
        return relationship_graph

    def get_device_filenames_from_graph(self, relationship_graph: Graph) -> List[str]:
        """
        Extracts device URIs from the relationship graph and finds the corresponding
        TTL filenames that need to be loaded.
        
        :param relationship_graph: Graph containing device-room-floor relationships
        :return: List of full file paths to the TTL files that need to be loaded
        """
        # Extract all device URIs from the graph
        device_uris = set()
        
        # Find all subjects of s4bldg:isContainedIn triples
        for s, p, o in relationship_graph.triples((None, self.S4BLDG.isContainedIn, None)):
            device_uris.add(str(s))
        
        # Find all objects of s4bldg:contains triples
        for s, p, o in relationship_graph.triples((None, self.S4BLDG.contains, None)):
            device_uris.add(str(o))
        
        # Map these URIs to filenames using the reverse lookup
        filenames = set()
        for device_uri in device_uris:
            if device_uri in self.reverse_lookup:
                for filename, _ in self.reverse_lookup[device_uri]:
                    filenames.add(filename)
        
        # Get the full paths for these filenames
        full_paths = []
        for subdir, _, files in os.walk(self.devices_root_dir):
            for file in files:
                if file in filenames:
                    full_paths.append(os.path.join(subdir, file))
                    
        return full_paths


def main():
    retriever = FloorDeviceRetriever()

    # Floor 7
    floor_7_relationship_graph = retriever.get_device_room_floor_graph(7)
    floor_7_filenames = retriever.get_device_filenames_from_graph(floor_7_relationship_graph)
    print("***** Device URIs (Floor 7) *****")
    for filename in floor_7_filenames:
        print(filename)

    # All floors
    print("***** Device counts (all floors) *****")
    for floor_num in range(8):
        relationship_graph = retriever.get_device_room_floor_graph(floor_num)
        filenames = retriever.get_device_filenames_from_graph(relationship_graph)
        print(f"== Floor {floor_num} ==")
        print("Device count:", len(filenames))

if __name__ == "__main__":
    main()

