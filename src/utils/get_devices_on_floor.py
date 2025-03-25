import os
import re
from typing import List, Set
from rdflib import Graph, Namespace

class FloorDeviceRetriever:
    """
    A helper class to load the device map, parse an enrichment TTL,
    and query for devices exclusively on a specified floor (excluding support_zone).
    It then maps the query results back to the local TTL file names.
    """

    def __init__(self,
                 devices_root_dir: str = "data/OfficeGraph/devices",
                 enrichment_ttl_path: str = "data/OfficeGraph/enrichments/devices_in_rooms_enrichment.ttl"):
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

        # Namespaces
        self.IC     = Namespace("https://interconnectproject.eu/example/")
        self.RDFS   = Namespace("http://www.w3.org/2000/01/rdf-schema#")
        self.S4BLDG = Namespace("https://saref.etsi.org/saref4bldg/")

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

    def get_devices_on_floor(self, floor_number: int = 7):
        """
        Runs a SPARQL query to find devices strictly on the given floor_number
        (and not in a 'support_zone'). Then cross-references them with the local TTL filenames.

        :param floor_number: e.g. 1, 2, 7, etc.
        :return: A list of records, each record is a dict with:
                 {
                    "deviceURI": <string>,
                    "short_id": <string from TTL>,
                    "filename": <the .ttl file where we found it>,
                    "buildingSpace": <URI of the building space>,
                    "floor": <URI of the floor>,
                    "roomComment": <the rdfs:comment if any, else None>
                 }
        """

        # Build the floor URI, e.g. ic:VL_floor_7 => "https://interconnectproject.eu/example/VL_floor_7"
        floor_uri = f"https://interconnectproject.eu/example/VL_floor_{floor_number}"

        # Parameterized SPARQL query
        sparql_query = f"""
        PREFIX ic: <https://interconnectproject.eu/example/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX s4bldg: <https://saref.etsi.org/saref4bldg/>
        SELECT DISTINCT ?device ?buildingSpace ?floor (SAMPLE(?comment) AS ?roomComment)
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
        GROUP BY ?device ?buildingSpace ?floor
        """

        results = self.graph.query(sparql_query)

        final_records = []
        # For each row, see if it's in our local dictionary
        for row in results:
            device_uri = str(row.device)
            bspace_uri = str(row.buildingSpace)
            floor_uri  = str(row.floor)
            comment    = str(row.roomComment) if row.roomComment else None

            if device_uri in self.reverse_lookup:
                # match found in local TTL dictionary
                for (filename, short_id) in self.reverse_lookup[device_uri]:
                    rec = {
                        "deviceURI": device_uri,
                        "short_id": short_id,
                        "filename": filename,
                        "buildingSpace": bspace_uri,
                        "floor": floor_uri,
                        "roomComment": comment
                    }
                    final_records.append(rec)

        return final_records

    def get_full_paths_for_filenames(self, filenames: Set[str]) -> List[str]:
        """
        Given a set of TTL filenames, return their full paths under devices_root_dir.

        :param filenames: e.g. {"C2E886.ttl", "urn_Device_SmartThings_xxx.ttl"}
        :return: List of full paths like ["data/OfficeGraph/devices/thermostats/C2E886.ttl", ...]
        """
        matches = []
        for subdir, _, files in os.walk(self.devices_root_dir):
            for file in files:
                if file in filenames:
                    matches.append(os.path.join(subdir, file))
        return matches
