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
from . import polygon_utils


@dataclass
class PolygonData:
    """Class to hold polygon data for a room."""
    room_id: str
    points: List[Tuple[float, float]]  # SVG coordinates
    geo_points: List[Tuple[float, float]] = None  # Real-world coordinates (lon, lat)
    utm_points: List[Tuple[float, float]] = None  # UTM coordinates (meters)


class SvgToRdf(NamespaceMixin):
    """Class to convert SVG floor plans to RDF spatial topology using PyProj."""
    
    def __init__(self,
                 building: Building,
                 floor_number: int,
                 output_dir: str = 'data/topology/TTLs'):
        """Initialize the converter.
        
        Args:
            building: Building object with geographic coordinates
            floor_number: Floor number to extract from the SVG file
            output_dir: Directory to save the output files
        """
        self.floor_number = floor_number
        self.output_dir = output_dir
        self.building = building
        
        # Construct the SVG path based on floor number
        self.svg_path = os.path.join("data", "topology", "SVGs", f"floor_{self.floor_number}.svg")
        
        # Create SVG parser instance
        self.svg_parser = SVGParser(self.svg_path)
        
        # SVG dimensions (will be extracted from the SVG by the parser)
        self.svg_width = None
        self.svg_height = None
        
        # Store room polygons
        self.rooms: Dict[str, PolygonData] = {}
        
        # Store building and building ground polygons
        self.building_polygon = None
        self.building_ground_polygon = None
        
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
        """Parse the SVG file to extract room polygons, building polygon, and building ground polygon."""
        # Get the SVG dimensions
        self.svg_width, self.svg_height = self.svg_parser.get_svg_dimensions()
        
        # Get the building polygon
        self.building_polygon = self.svg_parser.get_building_polygon()
        
        # Get the building ground polygon
        self.building_ground_polygon = self.svg_parser.get_building_ground_polygon()
        if not self.building_ground_polygon:
            raise ValueError("No building ground polygon found in SVG. Building ground polygon with inkscape:label='building_ground' is required.")
        
        # Get room polygons
        rooms_dict = self.svg_parser.get_room_polygons()
        
        # Convert raw points to PolygonData objects for rooms
        for room_id, points in rooms_dict.items():
            self.rooms[room_id] = PolygonData(
                room_id=room_id,
                points=points
            )
        
        # Log summary
        logger.info(f"Extracted {len(rooms_dict)} room polygons for floor {self.floor_number}")
        
        if self.building_polygon:
            logger.info(f"Building polygon extracted with {len(self.building_polygon)} points")
        else:
            logger.warning("No building polygon found in SVG, but continuing as this is optional.")
            
        logger.info(f"Building ground polygon extracted with {len(self.building_ground_polygon)} points")
            
    def convert_to_geo_coordinates(self):
        """Convert SVG coordinates to geographic coordinates using the building ground polygon."""
        # Verify we have a valid building ground polygon
        if len(self.building_ground_polygon) < 4:
            raise ValueError("Building ground polygon has fewer than 4 points. A valid rectangular polygon is required.")
        
        # For a rectangular polygon, find the min/max coordinates
        svg_xs = [p[0] for p in self.building_ground_polygon]
        svg_ys = [p[1] for p in self.building_ground_polygon]
        
        svg_min_x = min(svg_xs)
        svg_max_x = max(svg_xs)
        svg_min_y = min(svg_ys)
        svg_max_y = max(svg_ys)
        
        # Map these to geo coordinates
        geo_bounds = (
            self.building.top_left_lon,     # min_lon maps to min_x
            self.building.bottom_right_lat, # min_lat maps to max_y (SVG y is top-down)
            self.building.bottom_right_lon, # max_lon maps to max_x
            self.building.top_left_lat      # max_lat maps to min_y (SVG y is top-down)
        )
        
        logger.info(f"SVG building ground bounds: ({svg_min_x}, {svg_min_y}, {svg_max_x}, {svg_max_y})")
        logger.info(f"Geographic bounds: {geo_bounds}")
        
        # Convert building polygon if available
        if self.building_polygon:
            building_geo_points = self._svg_to_geo_using_ground(
                self.building_polygon, 
                (svg_min_x, svg_min_y, svg_max_x, svg_max_y), 
                geo_bounds
            )
            building_utm_points = [self.wgs84_to_utm.transform(lon, lat) for lon, lat in building_geo_points]
            self.building_geo_points = building_geo_points
            self.building_utm_points = building_utm_points
        
        # Convert building ground polygon
        building_ground_geo_points = self._svg_to_geo_using_ground(
            self.building_ground_polygon,
            (svg_min_x, svg_min_y, svg_max_x, svg_max_y),
            geo_bounds
        )
        building_ground_utm_points = [self.wgs84_to_utm.transform(lon, lat) for lon, lat in building_ground_geo_points]
        self.building_ground_geo_points = building_ground_geo_points
        self.building_ground_utm_points = building_ground_utm_points
        
        # Convert each room's points
        for room_id, room_data in self.rooms.items():
            # Convert SVG points to geographic points
            geo_points = self._svg_to_geo_using_ground(
                room_data.points,
                (svg_min_x, svg_min_y, svg_max_x, svg_max_y),
                geo_bounds
            )
            room_data.geo_points = geo_points
            
            # Also convert to UTM for accurate area calculations
            utm_points = [self.wgs84_to_utm.transform(lon, lat) for lon, lat in geo_points]
            room_data.utm_points = utm_points
    
    def _svg_to_geo_using_ground(self, svg_points, svg_bounds, geo_bounds):
        """
        Transform SVG coordinates to geographic coordinates using building ground bounds
        while preserving aspect ratio.
        """
        svg_min_x, svg_min_y, svg_max_x, svg_max_y = svg_bounds
        min_lon, min_lat, max_lon, max_lat = geo_bounds
        
        # Calculate SVG bounds dimensions
        svg_width = svg_max_x - svg_min_x
        svg_height = svg_max_y - svg_min_y
        svg_aspect_ratio = svg_width / svg_height if svg_height > 0 else 1
        
        # Calculate geo bounds dimensions in meters using UTM
        # Convert corners to UTM
        min_x_utm, min_y_utm = self.wgs84_to_utm.transform(min_lon, min_lat)
        max_x_utm, max_y_utm = self.wgs84_to_utm.transform(max_lon, max_lat)
        
        # Calculate UTM dimensions
        utm_width = max_x_utm - min_x_utm
        utm_height = max_y_utm - min_y_utm
        utm_aspect_ratio = utm_width / utm_height if utm_height > 0 else 1
        
        # Calculate scaling factor to preserve aspect ratio
        scale_factor = utm_aspect_ratio / svg_aspect_ratio
        
        # Transform each point to UTM first, then to WGS84
        geo_points = []
        for x, y in svg_points:
            # Normalize the point within the building ground bounds
            norm_x = (x - svg_min_x) / svg_width if svg_width > 0 else 0
            norm_y = 1 - ((y - svg_min_y) / svg_height if svg_height > 0 else 0)  # Invert y
            
            # Apply aspect ratio correction
            if scale_factor > 1:
                # UTM wider than SVG relative to height - adjust x
                norm_x = 0.5 + (norm_x - 0.5) * scale_factor
            else:
                # UTM taller than SVG relative to width - adjust y
                norm_y = 0.5 + (norm_y - 0.5) / scale_factor
            
            # Map to UTM coordinates
            utm_x = min_x_utm + norm_x * utm_width
            utm_y = min_y_utm + norm_y * utm_height
            
            # Convert to WGS84 (lon, lat)
            lon, lat = self.utm_to_wgs84.transform(utm_x, utm_y)
            geo_points.append((lon, lat))
        
        return geo_points
    
    def create_rdf_graph(self):
        """Create the RDF graph with spatial topology for the floor."""
        # Add building
        building_uri = self.EX.VideoLab
        self.graph.add((building_uri, RDF.type, self.BOT.Building))
        
        # Add building geometry
        self._add_building_geometry(building_uri)
        
        # Add floor and rooms
        self._add_floor_and_rooms(building_uri)
    
    def _add_floor_and_rooms(self, building_uri):
        """Add floor and its rooms to the RDF graph."""
        # Add floor
        floor_uri = self.IC[f"VL_floor_{self.floor_number}"]
        self.graph.add((floor_uri, RDF.type, self.BOT.Storey))
        self.graph.add((floor_uri, RDFS.label, Literal(f"floor {self.floor_number}")))
        self.graph.add((building_uri, self.BOT.hasStorey, floor_uri))
        
        # Get the floor height from the building class if available
        # Use building's floor_heights_accounting_for_sea_level if available
        if hasattr(self.building, 'floor_heights_accounting_for_sea_level') and \
        self.building.floor_heights_accounting_for_sea_level and \
        self.floor_number < len(self.building.floor_heights_accounting_for_sea_level):
            # floor_number in the array is 0-indexed 
            # (0 = ground floor, 1 = first floor, etc.)
            floor_height = self.building.floor_heights_accounting_for_sea_level[self.floor_number]
            logger.info(f"Using floor height from building data for floor {self.floor_number}: {floor_height}m")
        else:
            # Calculate floor height in meters (assuming 3 meters per floor, ground floor at 0)
            # This is the fallback calculation if building data is not available
            floor_height = self.building.height_from_sealevel + (self.floor_number * 3.0) if hasattr(self.building, 'height_from_sealevel') and self.building.height_from_sealevel else (self.floor_number * 3.0)
            logger.warning(f"Floor height data not available for floor {self.floor_number}, using calculated value: {floor_height}m")
        
        # Add rooms
        for room_id, room_data in self.rooms.items():
            if not room_data.geo_points:
                logger.warning(f"No geographic points for room {room_id} on floor {self.floor_number}")
                continue
                
            # URIs for the room and its geometry
            room_uri = self.IC[f"roomname_{room_id}"]
            geo_geometry_uri = self.EX[f"floor{self.floor_number}_{room_id.replace('.', '')}_geo"]
            doc_geometry_uri = self.EX[f"floor{self.floor_number}_{room_id.replace('.', '')}_doc"]
            
            # Add room properties
            self.graph.add((room_uri, RDF.type, self.BOT.Space))
            self.graph.add((room_uri, RDF.type, self.S4BLDG.BuildingSpace))
            self.graph.add((room_uri, RDF.type, self.GEOSPARQL.Feature))
            self.graph.add((room_uri, self.S4BLDG.isSpaceOf, floor_uri))
            
            # Add both real-world and document geometries
            self.graph.add((room_uri, self.GEOSPARQL.hasGeometry, geo_geometry_uri))
            self.graph.add((room_uri, self.EX_ONT.hasDocumentGeometry, doc_geometry_uri))
            
            # Add floor height to the room data as metadata
            self.graph.add((room_uri, self.GEO.alt, Literal(floor_height, datatype=XSD.float)))
            
            # Add room to floor relationship
            self.graph.add((floor_uri, self.BOT.hasSpace, room_uri))
            
            # Calculate and add room area using UTM coordinates
            if room_data.utm_points:
                # Use centralized utility for area calculation
                area = polygon_utils.calculate_polygon_area(room_data.utm_points)
                self.graph.add((room_uri, self.GEOSPARQL.hasMetricArea, Literal(area, datatype=XSD.float)))
            
            # Create WKT for the real-world polygon and add it to the graph
            geo_wkt = polygon_utils.create_wkt_polygon(room_data.geo_points, floor_height)
            self.graph.add((geo_geometry_uri, self.GEOSPARQL.asWKT, Literal(geo_wkt)))
            
            # Create WKT for the document polygon and add it to the graph
            doc_wkt = polygon_utils.create_wkt_polygon(room_data.points, 0)  # Z-coordinate 0 for document coordinates
            self.graph.add((doc_geometry_uri, self.GEOSPARQL.asWKT, Literal(doc_wkt)))

    def _add_building_geometry(self, building_uri):
        """Add geometry information for the building."""
        # Check if we have valid geographic coordinates for the building
        valid_coordinates = (
            self.building.top_left_lat is not None and 
            self.building.top_left_lon is not None and
            self.building.bottom_right_lat is not None and
            self.building.bottom_right_lon is not None
        )
        
        if not valid_coordinates:
            raise ValueError("Building object must have valid top_left and bottom_right coordinates.")
        
        # Add geographic center point
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
        
        # Add building ground geometry - this is always available since we require it
        # Real-world geometry
        geo_geometry_uri = self.EX.VideoLabGroundGeo
        self.graph.add((building_uri, self.EX_ONT.hasGroundGeometry, geo_geometry_uri))
        
        # Create WKT for the building ground polygon (at ground level)
        geo_wkt = polygon_utils.create_wkt_polygon(self.building_ground_geo_points, ground_floor_height)
        self.graph.add((geo_geometry_uri, self.GEOSPARQL.asWKT, Literal(geo_wkt)))
        
        # Document geometry
        doc_geometry_uri = self.EX.VideoLabGroundDoc
        self.graph.add((building_uri, self.EX_ONT.hasGroundDocumentGeometry, doc_geometry_uri))
        
        # Create WKT for document polygon
        doc_wkt = polygon_utils.create_wkt_polygon(self.building_ground_polygon, 0)
        self.graph.add((doc_geometry_uri, self.GEOSPARQL.asWKT, Literal(doc_wkt)))
        
        # Calculate and add ground area using UTM coordinates
        area = polygon_utils.calculate_polygon_area(self.building_ground_utm_points)
        self.graph.add((geo_geometry_uri, self.GEOSPARQL.hasMetricArea, Literal(area, datatype=XSD.float)))
        
        # Add building geometry if available (this is optional)
        if hasattr(self, 'building_geo_points') and self.building_geo_points:
            # Real-world geometry
            geo_geometry_uri = self.EX.VideoLabBuildingGeo
            self.graph.add((building_uri, self.GEOSPARQL.hasGeometry, geo_geometry_uri))
            
            # Create WKT for the building polygon (at ground floor level)
            geo_wkt = polygon_utils.create_wkt_polygon(self.building_geo_points, ground_floor_height)
            self.graph.add((geo_geometry_uri, self.GEOSPARQL.asWKT, Literal(geo_wkt)))
            
            # Document geometry
            doc_geometry_uri = self.EX.VideoLabBuildingDoc
            self.graph.add((building_uri, self.EX_ONT.hasDocumentGeometry, doc_geometry_uri))
            
            # Create WKT for document polygon
            doc_wkt = polygon_utils.create_wkt_polygon(self.building_polygon, 0)
            self.graph.add((doc_geometry_uri, self.GEOSPARQL.asWKT, Literal(doc_wkt)))
            
            # Calculate and add building area using UTM coordinates
            area = polygon_utils.calculate_polygon_area(self.building_utm_points)
            self.graph.add((geo_geometry_uri, self.GEOSPARQL.hasMetricArea, Literal(area, datatype=XSD.float)))
            
    def save_to_file(self):
        """Save the RDF graph to a TTL file."""
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, f"{self.building.name}_floor_{self.floor_number}.ttl")
        
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
        "--floor",
        type=int,
        default=7,
        help="Floor number to extract (default: 7)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/topology/TTLs",
        help="Output directory for RDF files"
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate the topology even if it already exists"
    )
    args = parser.parse_args()

    # Initialize the Building class instance for VideoLab building
    videolab = Building(name="VideoLab")

    # Initialize the converter and run
    converter = SvgToRdf(
        building=videolab,
        floor_number=args.floor,
        output_dir=args.output_dir
    )
    ttl_path = converter.run()
