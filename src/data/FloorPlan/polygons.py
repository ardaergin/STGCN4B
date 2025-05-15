import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from rdflib import Graph, Namespace, Literal, RDF, RDFS, XSD
import logging
from pyproj import Transformer, CRS

logger = logging.getLogger(__name__)

from ...config.namespaces import NamespaceMixin
from ...core.building import Building
from .svg_parser import SVGParser

@dataclass
class PolygonData:
    """Class to hold polygon data for a room."""
    room_id: str
    floor_number: int
    points: List[Tuple[float, float]]  # SVG coordinates
    geo_points: List[Tuple[float, float]] = None  # Real-world coordinates (lon, lat)
    utm_points: List[Tuple[float, float]] = None  # UTM coordinates (meters)


class SvgToRdf(NamespaceMixin):
    """Class to convert SVG floor plans to RDF spatial topology using PyProj."""
    
    def __init__(self,
                 svg_path: str,
                 building: Building,
                 output_dir: str = 'data/topology',
                 floor_numbers: List[int] = None):
        """Initialize the converter.
        
        Args:
            svg_path: Path to the SVG file
            building: Building object with geographic coordinates
            output_dir: Directory to save the output files
            floor_numbers: List of floor numbers to extract from the SVG file
        """
        self.svg_path = svg_path
        self.output_dir = output_dir
        self.floor_numbers = floor_numbers or [7]  # Default to 7th floor if not specified
        self.building = building

        # Create SVG parser instance
        self.svg_parser = SVGParser(svg_path)
        
        # SVG dimensions (will be extracted from the SVG by the parser)
        self.svg_width = None
        self.svg_height = None
        
        # Store room polygons per floor
        self.rooms_by_floor: Dict[int, Dict[str, PolygonData]] = {floor: {} for floor in self.floor_numbers}
        
        # Store building polygon
        self.building_polygon = None
        
        # The RDF graph for output
        self.graph = self.create_empty_graph_with_namespace_bindings()
        
        # Set up coordinate transformers
        self._setup_transformers()
    def _setup_transformers(self):
        """Set up PyProj transformers for coordinate conversion."""
        # Define coordinate reference systems
        self.wgs84_crs = CRS.from_epsg(4326)  # WGS84 (latitude/longitude)
        
        # For the Netherlands (Eindhoven), UTM zone 31N is appropriate
        # Can also use the Dutch national grid (Rijksdriehoekstelsel/RD New) with EPSG:28992
        self.utm_crs = CRS.from_epsg(32631)  # UTM zone 31N
        
        # Create transformers
        # always_xy=True ensures that coordinates are handled as (x,y) = (lon,lat) pairs
        self.wgs84_to_utm = Transformer.from_crs(self.wgs84_crs, self.utm_crs, always_xy=True)
        self.utm_to_wgs84 = Transformer.from_crs(self.utm_crs, self.wgs84_crs, always_xy=True)
    
    def parse_svg(self):
        """Parse the SVG file to extract room polygons for each floor layer."""
        # Get the SVG dimensions
        self.svg_width, self.svg_height = self.svg_parser.get_svg_dimensions()
        
        # Get the building polygon
        self.building_polygon = self.svg_parser.get_building_polygon()
        
        # Get room polygons for each floor
        rooms_by_floor = self.svg_parser.get_room_polygons(self.floor_numbers)
        
        # Convert raw points to PolygonData objects for rooms
        for floor_number, rooms in rooms_by_floor.items():
            for room_id, points in rooms.items():
                self.rooms_by_floor[floor_number][room_id] = PolygonData(
                    room_id=room_id,
                    floor_number=floor_number,
                    points=points
                )
        
        # Log summary
        if self.building_polygon:
            logger.info(f"Building polygon extracted with {len(self.building_polygon)} points")
        else:
            logger.warning("No building polygon found in SVG. Will use bounding box as fallback.")
            
    def convert_to_geo_coordinates(self):
        """Convert SVG coordinates to geographic coordinates using PyProj for all floors."""
        # Geo bounds (min_lon, min_lat, max_lon, max_lat)
        geo_bounds = (
            self.building.top_left_lon,             # min_lon
            self.building.bottom_right_lat,         # min_lat
            self.building.bottom_right_lon,         # max_lon
            self.building.top_left_lat              # max_lat
        )
        
        logger.info(f"Geographic bounds: {geo_bounds}")
        
        # Convert building polygon if available
        if self.building_polygon:
            building_geo_points = self._svg_to_geo(self.building_polygon, (self.svg_width, self.svg_height), geo_bounds)
            building_utm_points = [self.wgs84_to_utm.transform(lon, lat) for lon, lat in building_geo_points]
            self.building_geo_points = building_geo_points
            self.building_utm_points = building_utm_points
        
        # Convert each room's points for all floors
        for floor_number, rooms in self.rooms_by_floor.items():
            for room_id, room_data in rooms.items():
                # Convert SVG points to geographic points
                geo_points = self._svg_to_geo(room_data.points, (self.svg_width, self.svg_height), geo_bounds)
                room_data.geo_points = geo_points
                
                # Also convert to UTM for accurate area calculations
                utm_points = [self.wgs84_to_utm.transform(lon, lat) for lon, lat in geo_points]
                room_data.utm_points = utm_points
    
    def _svg_to_geo(self, svg_points, svg_dimensions, geo_bounds):
        """
        Transform SVG coordinates to geographic coordinates.
        
        Args:
            svg_points: List of (x, y) points in SVG coordinates
            svg_dimensions: (width, height) of the SVG
            geo_bounds: (min_lon, min_lat, max_lon, max_lat) of the area
            
        Returns:
            List of (lon, lat) points in WGS84 coordinates
        """
        svg_width, svg_height = svg_dimensions
        min_lon, min_lat, max_lon, max_lat = geo_bounds
        
        # First normalize SVG coordinates (0 to 1)
        normalized_points = []
        for x, y in svg_points:
            # Flip y-axis as SVG has origin at top-left
            norm_x = x / svg_width
            norm_y = 1 - (y / svg_height)  # Flip y-axis
            normalized_points.append((norm_x, norm_y))
        
        # Map to geographic coordinates (WGS84)
        geo_points = []
        for norm_x, norm_y in normalized_points:
            lon = min_lon + norm_x * (max_lon - min_lon)
            lat = min_lat + norm_y * (max_lat - min_lat)
            geo_points.append((lon, lat))
        
        return geo_points
    
    def create_rdf_graph(self):
        """Create the RDF graph with spatial topology for all floors."""
        # Add building
        building_uri = self.EX.VideoLab
        self.graph.add((building_uri, RDF.type, self.BOT.Building))
        
        # Add building geometry
        self._add_building_geometry(building_uri)
        
        # Add floors and rooms for each floor
        for floor_number in self.floor_numbers:
            if floor_number in self.rooms_by_floor and self.rooms_by_floor[floor_number]:
                self._add_floor_and_rooms(building_uri, floor_number)
    
    def _add_floor_and_rooms(self, building_uri, floor_number):
        """Add floor and its rooms to the RDF graph."""
        # Add floor
        floor_uri = self.IC[f"VL_floor_{floor_number}"]
        self.graph.add((floor_uri, RDF.type, self.BOT.Storey))
        self.graph.add((floor_uri, RDFS.label, Literal(f"floor {floor_number}")))
        self.graph.add((building_uri, self.BOT.hasStorey, floor_uri))
        
        # Get the floor height from the building class if available
        # Use building's floor_heights_accounting_for_sea_level if available
        if hasattr(self.building, 'floor_heights_accounting_for_sea_level') and \
        self.building.floor_heights_accounting_for_sea_level and \
        floor_number < len(self.building.floor_heights_accounting_for_sea_level):
            # floor_number in the array is 0-indexed 
            # (0 = ground floor, 1 = first floor, etc.)
            floor_height = self.building.floor_heights_accounting_for_sea_level[floor_number]
            logger.info(f"Using floor height from building data for floor {floor_number}: {floor_height}m")
        else:
            # Calculate floor height in meters (assuming 3 meters per floor, ground floor at 0)
            # This is the fallback calculation if building data is not available
            floor_height = self.building.height_from_sealevel + (floor_number * 3.0) if hasattr(self.building, 'height_from_sealevel') and self.building.height_from_sealevel else (floor_number * 3.0)
            logger.warning(f"Floor height data not available for floor {floor_number}, using calculated value: {floor_height}m")
        
        # Add rooms for this floor
        for room_id, room_data in self.rooms_by_floor[floor_number].items():
            if not room_data.geo_points:
                logger.warning(f"No geographic points for room {room_id} on floor {floor_number}")
                continue
                
            # URIs for the room and its geometry
            room_uri = self.IC[f"roomname_{room_id}"]
            geometry_uri = self.IC[f"floor{floor_number}_{room_id.replace('.', '')}geo"]
            
            # Add room properties
            self.graph.add((room_uri, RDF.type, self.BOT.Space))
            self.graph.add((room_uri, RDF.type, self.S4BLDG.BuildingSpace))
            self.graph.add((room_uri, RDF.type, self.GEOSPARQL.Feature))
            self.graph.add((room_uri, RDFS.comment, Literal("room")))
            self.graph.add((room_uri, RDFS.label, Literal(f"{room_id}")))
            self.graph.add((room_uri, self.S4BLDG.isSpaceOf, floor_uri))
            self.graph.add((room_uri, self.GEOSPARQL.hasGeometry, geometry_uri))
            
            # Add floor height to the room data as metadata
            self.graph.add((room_uri, self.GEO.alt, Literal(floor_height, datatype=XSD.float)))
            
            # Add room to floor relationship
            self.graph.add((floor_uri, self.BOT.hasSpace, room_uri))
            
            # Calculate and add room area using UTM coordinates
            if room_data.utm_points:
                area = self._calculate_polygon_area(room_data.utm_points)
                self.graph.add((room_uri, self.GEOSPARQL.hasMetricArea, Literal(area, datatype=XSD.float)))
            
            # Create WKT for the polygon and add it to the graph
            # Pass the actual floor height value for the z-coordinate
            wkt = self._create_wkt_polygon(room_data.geo_points, floor_height)
            self.graph.add((geometry_uri, self.GEOSPARQL.asWKT, Literal(wkt)))

    def _add_building_geometry(self, building_uri):
        """Add geometry information for the building."""
        # Check if we have valid geographic coordinates for the building
        valid_coordinates = (
            self.building.top_left_lat is not None and 
            self.building.top_left_lon is not None and
            self.building.bottom_right_lat is not None and
            self.building.bottom_right_lon is not None
        )
        
        # Add geographic center point if we have valid coordinates
        if valid_coordinates:
            center_lat = (self.building.top_left_lat + self.building.bottom_right_lat) / 2
            center_lon = (self.building.top_left_lon + self.building.bottom_right_lon) / 2
            
            self.graph.add((building_uri, self.GEO.lat, Literal(center_lat, datatype=XSD.float)))
            self.graph.add((building_uri, self.GEO.long, Literal(center_lon, datatype=XSD.float)))
            
            # Add height from sea level if available
            if hasattr(self.building, 'height_from_sealevel') and self.building.height_from_sealevel is not None:
                self.graph.add((building_uri, self.GEO.alt, Literal(self.building.height_from_sealevel, datatype=XSD.float)))
        
        # Get ground floor height (for z-coordinate of building footprint)
        ground_floor_height = 0.0
        if hasattr(self.building, 'height_from_sealevel') and self.building.height_from_sealevel is not None:
            ground_floor_height = self.building.height_from_sealevel
        
        # Add building geometry using the actual polygon extracted from SVG
        if hasattr(self, 'building_geo_points') and self.building_geo_points:
            geometry_uri = self.EX.VideoLabBuildingGeo
            self.graph.add((building_uri, self.GEOSPARQL.hasGeometry, geometry_uri))
            
            # Create WKT for the building polygon (at ground floor level)
            wkt = self._create_wkt_polygon(self.building_geo_points, ground_floor_height)
            self.graph.add((geometry_uri, self.GEOSPARQL.asWKT, Literal(wkt)))
            
            # Calculate and add building area using UTM coordinates
            if hasattr(self, 'building_utm_points') and self.building_utm_points:
                area = self._calculate_polygon_area(self.building_utm_points)
                self.graph.add((building_uri, self.GEOSPARQL.hasMetricArea, Literal(area, datatype=XSD.float)))
        elif valid_coordinates:
            # Fallback to bounding box if no building polygon was extracted
            logger.warning("No building polygon found in SVG. Using bounding box instead.")
            # Add bounding box as a polygon
            geo_points = [
                (self.building.top_left_lon,      self.building.top_left_lat),
                (self.building.bottom_right_lon,  self.building.top_left_lat),
                (self.building.bottom_right_lon,  self.building.bottom_right_lat),
                (self.building.top_left_lon,      self.building.bottom_right_lat),
                (self.building.top_left_lon,      self.building.top_left_lat),
            ]
            
            geometry_uri = self.EX.VideoLabBuildingGeo
            self.graph.add((building_uri, self.GEOSPARQL.hasGeometry, geometry_uri))
            
            # Create WKT for the building polygon (at ground floor level)
            wkt = self._create_wkt_polygon(geo_points, ground_floor_height)
            self.graph.add((geometry_uri, self.GEOSPARQL.asWKT, Literal(wkt)))
            
            # Calculate and add building area using UTM coordinates
            # First convert to UTM
            utm_points = [self.wgs84_to_utm.transform(lon, lat) for lon, lat in geo_points]
            area = self._calculate_polygon_area(utm_points)
            self.graph.add((building_uri, self.GEOSPARQL.hasMetricArea, Literal(area, datatype=XSD.float)))
        else:
            logger.warning("No building geometry available and no valid coordinates for bounding box.")    
    def _create_wkt_polygon(self, points, z_coord=0.0):
        """Create WKT POLYGON representation from points with Z-coordinate.
        
        Args:
            points: List of (lon, lat) tuples
            z_coord: Z-coordinate for 3D polygon (height in meters)
            
        Returns:
            WKT string for the polygon
        """
        # Format: POLYGON Z((lon1 lat1 z1, lon2 lat2 z2, ...))
        point_strings = [f"{lon} {lat} {z_coord}" for lon, lat in points]
        return f"POLYGON Z(({', '.join(point_strings)}))"
    
    def _calculate_polygon_area(self, points):
        """Calculate the area of a polygon in square meters using UTM coordinates.
        
        This uses the Shoelace formula (Gauss's area formula).
        
        Args:
            points: List of (x, y) points in UTM coordinates (meters)
            
        Returns:
            Area in square meters
        """
        # Apply Shoelace formula
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    
    def save_to_file(self):
        """Save the RDF graph to a TTL file."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{self.building.name}.ttl")
        
        self.graph.serialize(destination=output_path, format="turtle")
        logger.info(f"Saved {len(self.graph)} triples to {output_path}")
        
        return output_path

    def run(self):
        """Run the full conversion process."""
        self.parse_svg()
        self.convert_to_geo_coordinates()
        self.create_rdf_graph()
        return self.save_to_file()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert SVG floor plans to RDF spatial topology")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data",
        help="Base directory for data files"
    )
    parser.add_argument(
        "--floors",
        type=int,
        nargs="+",
        default=[7],
        help="List of floor numbers to extract (default: 7)"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the topology even if it already exists"
    )
    args = parser.parse_args()

    # Initialize paths
    svg_path = os.path.join(args.base_dir, "topology", "videolab.svg")
    output_dir = os.path.join(args.base_dir, "topology")

    # Initialize the Building class instance for VideoLab building
    videolab = Building(name="VideoLab")

    # Initialize the converter and run
    converter = SvgToRdf(
        svg_path=svg_path,
        output_dir=output_dir,
        floor_numbers=args.floors,
        building=videolab
    )
    ttl_path = converter.run()
