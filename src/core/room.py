from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Any
from rdflib import URIRef
import logging
import math

# Set up logging
logger = logging.getLogger(__name__)

# Import the utility module
from ..data.FloorPlan import polygon_utils
from ..config.namespaces import NamespaceMixin


@dataclass(slots=True)
class Room(NamespaceMixin):
    uri: URIRef # In the RDF files, they are written as, e.g., "ic:roomname_7.003", so uri is the full "ic:roomname_7.003"
    room_number: Optional[str] = None # the room number is just the number: "7.003"
    # NOTE: 'rdfs:label' (e.g.,`rdfs:label "7.073"`) is the room number.
    is_support_zone: bool = False  # True for support_zone, False for room
    floor: Optional[URIRef] = None
    devices: Set[URIRef] = field(default_factory=set)

    # Attributes from CSV
    isProperRoom: Optional[bool] = None
    hasWindows: Optional[bool] = None
    hasWindowsFacingDirection: List[float] = field(default_factory=list)
    hasWindowsRelativeDirection: List[float] = field(default_factory=list)

    # Window direction components (calculated from hasWindowsFacingDirection)
    window_direction_sin: float = 0.0
    window_direction_cos: float = 0.0
    has_multiple_windows: bool = False

    # Calculate based on hasWindowsRelativeDirection
    hasBackWindows: bool = False
    hasFrontWindows: bool = False
    hasRightWindows: bool = False
    hasLeftWindows: bool = False

    # Spatial relationships
    adjacent_rooms: List[str] = field(default_factory=list)

    # Altitude of the room (from RDF)
    altitude: Optional[float] = None

    # Raw polygon data from the RDF
    geo_wkt_polygon: Optional[str] = None
    doc_wkt_polygon: Optional[str] = None

    # Dictionaries to store all polygon-related data
    polygons_geo: Dict[str, Any] = field(default_factory=dict)
    polygons_doc: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Process polygon data from RDF, if available."""
        # Initialize the polygon dictionaries with default values
        self._initialize_polygon_dictionaries()
        
        # Process geographic polygon
        self._process_wkt_polygon('geo')
            
        # Process document polygon
        self._process_wkt_polygon('doc')

        # Calculate window direction components
        self._calculate_window_direction_components()

        # Calculate window relative direction flags
        self._calculate_window_relative_direction_flags()

    def _calculate_window_direction_components(self):
        """
        Calculate sine and cosine components for window directions.
        This converts the hasWindowsFacingDirection list into:
        - window_direction_sin: Sine component of avg direction
        - window_direction_cos: Cosine component of avg direction
        - has_multiple_window_directions: Flag for multiple directions
        """
        # Default values (no windows case)
        self.window_direction_sin = 0.0
        self.window_direction_cos = 0.0
        self.has_multiple_windows = False
        
        # Skip if no window directions available
        if not self.hasWindowsFacingDirection:
            return
        
        # Extract valid directions - converting strings to float/int if needed
        valid_directions = []
        for dir_ in self.hasWindowsFacingDirection:
            try:
                # Convert to float or int if it's a string
                if isinstance(dir_, str):
                    dir_ = float(dir_)
                
                # Ensure it's a number
                if isinstance(dir_, (int, float)):
                    valid_directions.append(dir_)
            except (ValueError, TypeError):
                logger.warning(f"Invalid window direction value: {dir_}")
        
        # Skip if no valid directions
        if not valid_directions:
            logger.warning(f"No valid window directions for room {self.room_number}")
            return
        
        # Check for multiple directions
        self.has_multiple_windows = len(valid_directions) > 1
        
        # Convert directions to radians (assuming input is in degrees 0-360)
        directions_rad = [math.radians(float(dir_)) for dir_ in valid_directions]
        
        # Calculate average direction components
        sin_values = [math.sin(rad) for rad in directions_rad]
        cos_values = [math.cos(rad) for rad in directions_rad]
            
        # Compute raw averages
        raw_sin = sum(sin_values) / len(sin_values)
        raw_cos = sum(cos_values) / len(cos_values)

        # Round to 3 decimal places (change the 3 to whatever precision you like)
        self.window_direction_sin = round(raw_sin, 3)
        self.window_direction_cos = round(raw_cos, 3)
        self.has_multiple_windows = len(valid_directions) > 1

        logger.debug(
            f"Room {self.room_number}: "
            f"sin={self.window_direction_sin}, cos={self.window_direction_cos}, "
            f"multiple={self.has_multiple_windows}"
        )

    def _calculate_window_relative_direction_flags(self):
        """
        Calculate boolean flags for each window direction based on hasWindowsRelativeDirection.
        Sets hasBackWindows, hasFrontWindows, hasRightWindows, and hasLeftWindows.
        """
        # Reset all flags
        self.hasBackWindows = False
        self.hasFrontWindows = False
        self.hasRightWindows = False
        self.hasLeftWindows = False
        
        # Skip if no relative directions available
        if not self.hasWindowsRelativeDirection:
            return
        
        # Convert all directions to lowercase for case-insensitive comparison
        directions = [d.lower() for d in self.hasWindowsRelativeDirection if isinstance(d, str)]
        
        # Set flags based on directions
        self.hasBackWindows = any('back' in d for d in directions)
        self.hasFrontWindows = any('front' in d for d in directions)
        self.hasRightWindows = any('right' in d for d in directions)
        self.hasLeftWindows = any('left' in d for d in directions)
                
    def _initialize_polygon_dictionaries(self):
        """Initialize polygon dictionaries with default values."""
        default_polygon_data = {
            'wkt': None,              # Original WKT string
            'points_2d': [],          # List of (x, y) tuples
            'points_3d': [],          # List of (x, y, z) tuples
            # NOTE: Z-coordinate is the floor number for document coordinates
            'centroid': (None, None), # (x, y) of centroid
            'area': None,             # Area in square meters/units
            'perimeter': None,        # Perimeter in meters/units
            'width': None,            # Bounding box width
            'height': None,           # Bounding box height
            'compactness': None,      # Circularity measure
            'rect_fit': None,         # How rectangular the room is
            'aspect_ratio': None,     # Width to height ratio
            'boundary_points': []     # Evenly spaced points on boundary
        }
        
        # Use dict() to create a copy of the default dictionary
        self.polygons_geo = dict(default_polygon_data)
        self.polygons_doc = dict(default_polygon_data)
        
    def _process_wkt_polygon(self, polygon_type: str):
        """
        Process WKT polygon and calculate metrics.
        
        Args:
            polygon_type: Either 'geo' or 'doc' to indicate which polygon to process
        """
        # Determine which polygon data and dictionary to use based on type
        if polygon_type == 'geo':
            wkt_polygon = self.geo_wkt_polygon
            polygon_dict = self.polygons_geo
            altitude = self.altitude  # Only used for geo polygons
        elif polygon_type == 'doc':
            wkt_polygon = self.doc_wkt_polygon
            polygon_dict = self.polygons_doc
            altitude = None  # Not needed for doc polygons
        else:
            logger.error(f"Invalid polygon type: {polygon_type}")
            return
        
        # Skip processing if no polygon data
        if not wkt_polygon:
            logger.debug(f"No {polygon_type} WKT polygon data for room {self.room_number or self.uri}")
            return
        
        try:
            # Store the WKT string
            polygon_dict['wkt'] = wkt_polygon
            
            # Parse the WKT string to get point lists
            points_2d, points_3d = polygon_utils.parse_wkt_polygon(
                wkt_polygon, 
                altitude=altitude if polygon_type == 'geo' else None
            )
            
            # Check if we have valid points
            if not points_2d or len(points_2d) < 3:
                logger.warning(f"Not enough valid points in {polygon_type} WKT for room {self.room_number or self.uri}")
                return
                
            polygon_dict['points_2d'] = points_2d
            polygon_dict['points_3d'] = points_3d
            
            # Calculate metrics
            try:
                metrics = polygon_utils.calculate_shape_metrics(points_2d)
                
                # Store metrics in the dictionary
                polygon_dict['centroid'] = metrics['centroid']
                polygon_dict['area'] = metrics.get('area')
                polygon_dict['perimeter'] = metrics.get('perimeter')
                polygon_dict['width'] = metrics.get('width')
                polygon_dict['height'] = metrics.get('height')
                polygon_dict['compactness'] = metrics.get('compactness')
                polygon_dict['rect_fit'] = metrics.get('rect_fit')
                polygon_dict['aspect_ratio'] = metrics.get('aspect_ratio')
                
                # Create boundary points
                try:
                    polygon_dict['boundary_points'] = polygon_utils.create_boundary_points(points_2d)
                except Exception as e:
                    logger.warning(f"Failed to create boundary points for {polygon_type} polygon in room {self.room_number or self.uri}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error calculating metrics for {polygon_type} polygon in room {self.room_number or self.uri}: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error processing {polygon_type} WKT for room {self.room_number or self.uri}: {str(e)}")

    @classmethod
    def from_uri(cls, uri: URIRef, is_support_zone: bool = False) -> 'Room':
        """Create a Room instance from a URI, extracting room number if possible."""
        room_number = None
        room_str = str(uri)
        if "roomname_" in room_str:
            try:
                room_number = room_str.split("roomname_")[-1]
            except Exception:
                logger.warning(f"Failed to extract room number from URI: {uri}")
                
        return cls(
            uri=uri,
            room_number=room_number,
            is_support_zone=is_support_zone
        )

    def add_device(self, device_uri: URIRef) -> None:
        """Add a device to this room."""
        self.devices.add(device_uri)

    def remove_device(self, device_uri: URIRef) -> None:
        """Remove a device from this room."""
        self.devices.discard(device_uri)

    def get_floor_number(self) -> str:
        """
        Extracts and returns the floor number from the self.floor URIRef.
        Falls back to 'n/a' if self.floor is None or parsing fails.
        """
        floor_display = 'n/a'
        if self.floor:
            floor_str = str(self.floor)
            if 'floor_' in floor_str:
                floor_display = floor_str.split('floor_')[-1].strip()
            else:
                floor_display = floor_str.split('/')[-1].strip()
        return floor_display

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        room_dict = {
            "uri": str(self.uri),
            "room_number": self.room_number,
            "is_support_zone": self.is_support_zone,
            "floor": str(self.floor) if self.floor else None,
            "device_count": len(self.devices),
            
            # Attributes from CSV
            "isProperRoom": self.isProperRoom,
            "hasWindows": self.hasWindows,
            "hasWindowsFacingDirection": self.hasWindowsFacingDirection,
            "hasWindowsRelativeDirection": self.hasWindowsRelativeDirection,
            
            # Window direction flags
            "hasBackWindows": self.hasBackWindows,
            "hasFrontWindows": self.hasFrontWindows,
            "hasRightWindows": self.hasRightWindows,
            "hasLeftWindows": self.hasLeftWindows,
            
            # Window direction components
            "window_direction_sin": self.window_direction_sin,
            "window_direction_cos": self.window_direction_cos,
            "has_multiple_windows": self.has_multiple_windows,
            
            # Spatial relationships
            "adjacent_rooms": self.adjacent_rooms,
            
            # Altitude if available
            "altitude": self.altitude,
            
            # Include the polygon dictionaries directly
            "polygons_geo": self.polygons_geo,
            "polygons_doc": self.polygons_doc
        }
        
        return room_dict
            
    def __str__(self):
        """String representation of the room."""
        area_str = 'n/a'
        if self.polygons_geo.get('area') is not None:
            area_str = f"{self.polygons_geo['area']:.2f} m²"
        
        # Add window information to string representation
        window_info = ""
        if self.hasWindows:
            window_dirs = []
            if self.hasFrontWindows:
                window_dirs.append("front")
            if self.hasBackWindows:
                window_dirs.append("back")
            if self.hasLeftWindows:
                window_dirs.append("left")
            if self.hasRightWindows:
                window_dirs.append("right")
            window_info = f" │ windows={','.join(window_dirs)}"
            
        return (
            f"Room {self.room_number or self.uri!r} │ "
            f"floor={self.get_floor_number()} │ "
            f"devices={len(self.devices)} │ "
            f"area={area_str}{window_info}"
        )

    def __repr__(self):
        """Detailed representation of the room."""
        lines = ["Room("]
        # Core attributes
        for attr in [
            f"    uri={self.uri!r},",
            f"    room_number={self.room_number!r},",
            f"    is_support_zone={self.is_support_zone!r},",
            f"    floor={self.floor!r},",
            f"    n_devices={len(self.devices)!r},",
            f"    isProperRoom={self.isProperRoom!r},",
            f"    hasWindows={self.hasWindows!r},",
            f"    hasWindowsFacingDirection={self.hasWindowsFacingDirection!r},",
            f"    hasWindowsRelativeDirection={self.hasWindowsRelativeDirection!r},",
            f"    hasBackWindows={self.hasBackWindows!r},",
            f"    hasFrontWindows={self.hasFrontWindows!r},",
            f"    hasRightWindows={self.hasRightWindows!r},",
            f"    hasLeftWindows={self.hasLeftWindows!r},",
            f"    window_direction_sin={self.window_direction_sin!r},",
            f"    window_direction_cos={self.window_direction_cos!r},",
            f"    has_multiple_windows={self.has_multiple_windows!r},",
            f"    adjacent_rooms={self.adjacent_rooms!r},",
            f"    altitude={self.altitude!r},",
        ]:
            lines.append(attr)
        
        # Polygon dictionaries
        lines.append(f"    polygons_geo={{\n")
        for key, value in self.polygons_geo.items():
            if key.startswith('points_') or key == 'boundary_points':
                if value:
                    lines.append(f"        '{key}': [")
                    for pt in value[:3]:  # Only show first few points for brevity
                        lines.append(f"            {pt!r},")
                    if len(value) > 3:
                        lines.append(f"            # ... ({len(value) - 3} more points)")
                    lines.append("        ],")
                else:
                    lines.append(f"        '{key}': [],")
            else:
                lines.append(f"        '{key}': {value!r},")
        lines.append("    },")
        
        lines.append(f"    polygons_doc={{\n")
        for key, value in self.polygons_doc.items():
            if key.startswith('points_') or key == 'boundary_points':
                if value:
                    lines.append(f"        '{key}': [")
                    for pt in value[:3]:  # Only show first few points for brevity
                        lines.append(f"            {pt!r},")
                    if len(value) > 3:
                        lines.append(f"            # ... ({len(value) - 3} more points)")
                    lines.append("        ],")
                else:
                    lines.append(f"        '{key}': [],")
            else:
                lines.append(f"        '{key}': {value!r},")
        lines.append("    }")
        
        lines.append(")")
        return "\n".join(lines)

    def __hash__(self):
        return hash(self.uri)

    def __eq__(self, other):
        """Check equality based on URI."""
        if isinstance(other, Room):
            return self.uri == other.uri
        return False
    