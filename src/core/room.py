from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
from rdflib import URIRef, Literal, Namespace, RDF, RDFS


@dataclass(slots=True)
class Room:
    uri: URIRef # In the RDF files, they are written as, e.g., "ic:roomname_7.003", so uri is the full "ic:roomname_7.003"
    room_number: Optional[str] = None # the room number is just the number: "7.003"
    # NOTE: 'rdfs:label' (e.g.,`rdfs:label "7.073"`) is the room number.
    is_support_zone: bool = False  # True for support_zone, False for room
    floor: Optional[URIRef] = None
    devices: Set[URIRef] = field(default_factory=set)

    # Namespace constants
    S4BLDG = Namespace("https://saref.etsi.org/saref4bldg/")
    BOT = Namespace("https://w3id.org/bot#")
    GEO1 = Namespace("http://www.w3.org/2003/01/geo/wgs84_pos#")
    GEOSPARQL = Namespace("http://www.opengis.net/ont/geosparql#")
    EX = Namespace("http://example.org/")  # Custom namespace for extension properties
    EXONT = Namespace("http://example.org/ontology#")  # Custom namespace for ontology extensions

    # Attributes from CSV
    isRoom: Optional[bool] = None
    hasWindows: Optional[bool] = None
    hasWindowsFacingDirection: List[float] = field(default_factory=list)

    # Spatial relationships
    adjacent_rooms: List[str] = field(default_factory=list)

    # Data directly from the RDF
    wkt_polygon: Optional[str] = None
    metric_area: Optional[float] = None
    altitude: Optional[float] = None

    # To be calculated at __post_init__:
    ## Polygon data
    polygon_2d: List[Tuple[float, float]] = field(default_factory=list)
    polygon_3d: List[Tuple[float, float, float]] = field(default_factory=list)

    ## Centroid
    centroid_x: Optional[float] = None
    centroid_y: Optional[float] = None
    
    ## Additional shape metrics
    perimeter: Optional[float] = None
    compactness: Optional[float] = None  # Circularity measure
    width: Optional[float] = None        # Bounding box width
    height: Optional[float] = None       # Bounding box height
    rect_fit: Optional[float] = None     # How rectangular the room is
    aspect_ratio: Optional[float] = None # Width to height ratio

    ## Boundary points (maybe useful for STGCN)
    boundary_points: List[Tuple[float, float]] = field(default_factory=list)
    
    def __post_init__(self):
        """Process WKT polygon data if available."""
        if self.wkt_polygon:
            self._process_wkt_polygon()
            self._calculate_spatial_metrics()
            self._create_boundary_points()
            
    def _process_wkt_polygon(self):
        """Extract polygon coordinates from WKT string and process them."""
        if not self.wkt_polygon:
            return
            
        # Extract coordinates from WKT string
        # Example WKT: "POLYGON Z((x1 y1 z1, x2 y2 z2, ...))" or "POLYGON ((x1 y1 z1, x2 y2 z2, ...))"
        try:
            # Clean and extract coordinates
            coords_str = self.wkt_polygon.strip()
            
            # Handle 'POLYGON Z' format
            if "POLYGON Z" in coords_str:
                # Remove 'POLYGON Z' prefix
                coords_str = coords_str.replace("POLYGON Z", "").strip()
            elif coords_str.startswith("POLYGON"):
                # Remove 'POLYGON' prefix
                coords_str = coords_str[7:].strip()
                
            # Remove outer parentheses
            coords_str = coords_str.strip("()")
            
            # Extract coordinates directly into the polygon_2d and polygon_3d lists
            for point_str in coords_str.split(","):
                point = point_str.strip().split()
                
                if len(point) >= 2:  # Must have at least x, y
                    x, y = float(point[0]), float(point[1])
                    self.polygon_2d.append((x, y))
                    
                    if len(point) >= 3:  # Has z coordinate
                        z = float(point[2])
                        self.polygon_3d.append((x, y, z))
                    elif self.altitude is not None:
                        # Use the room's altitude as z if available
                        self.polygon_3d.append((x, y, self.altitude))
        except Exception as e:
            # Log error but continue
            print(f"Error processing WKT polygon for room {self.uri}: {e}")
                    
    def _calculate_spatial_metrics(self):
        """Calculate centroid and shape metrics in a single pass for efficiency."""
        if not self.polygon_2d or len(self.polygon_2d) < 3:
            return
        
        # Use metric_area from RDF if available; otherwise calculate it
        area = self.metric_area
        if area is None:
            # Calculate area using Shoelace formula
            area = 0.0
            n = len(self.polygon_2d)
            for i in range(n):
                j = (i + 1) % n
                area += self.polygon_2d[i][0] * self.polygon_2d[j][1]
                area -= self.polygon_2d[j][0] * self.polygon_2d[i][1]
            area = abs(area) / 2.0
            
            # Store calculated area if not available from RDF
            if self.metric_area is None:
                self.metric_area = area
        
        # Early exit if area is zero
        if area == 0:
            return
        
        # Calculate perimeter and centroid simultaneously
        perimeter = 0.0
        cx = cy = 0.0
        
        # Get coordinates for bounding box calculation
        x_coords = []
        y_coords = []
        
        # Single pass through polygon vertices
        n = len(self.polygon_2d)
        for i in range(n):
            p1 = self.polygon_2d[i]
            p2 = self.polygon_2d[(i + 1) % n]
            
            # Add to coordinate lists for bounding box
            x_coords.append(p1[0])
            y_coords.append(p1[1])
            
            # Calculate perimeter segment
            segment_length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            perimeter += segment_length
            
            # Calculate centroid factor
            factor = (p1[0] * p2[1] - p2[0] * p1[1])
            cx += (p1[0] + p2[0]) * factor
            cy += (p1[1] + p2[1]) * factor
        
        # Finalize centroid calculation
        self.centroid_x = cx / (6 * area)
        self.centroid_y = cy / (6 * area)
        
        # Calculate bounding box
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Calculate remaining shape metrics
        width = max_x - min_x
        height = max_y - min_y
        
        # Compactness (circularity)
        compactness = (4 * 3.14159 * area) / (perimeter**2) if perimeter > 0 else 0
        
        # Rectangular fit
        rect_fit = area / (width * height) if width > 0 and height > 0 else 0
        
        # Store all the calculated metrics
        self.perimeter = perimeter
        self.compactness = compactness
        self.width = width
        self.height = height
        self.rect_fit = rect_fit
        self.aspect_ratio = width / height if height > 0 else 0
    
    def _create_boundary_points(self, num_points: int = 20):
        """
        Create a list of evenly spaced points along the room boundary.
        This can be useful for more accurate distance calculations in GNNs.
        
        Args:
            num_points: Number of points to generate along the boundary
            
        Returns:
            List of (x, y) coordinates of boundary points
        """
        if not self.polygon_2d or len(self.polygon_2d) < 3:
            return []
        
        # Calculate total perimeter length and segment lengths
        segments = []
        total_length = 0.0
        
        for i in range(len(self.polygon_2d)):
            p1 = self.polygon_2d[i]
            p2 = self.polygon_2d[(i + 1) % len(self.polygon_2d)]
            segment_length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
            segments.append((p1, p2, segment_length))
            total_length += segment_length
        
        # Generate evenly spaced points
        boundary_points = []
        spacing = total_length / num_points
        
        current_distance = 0.0
        for i, (p1, p2, length) in enumerate(segments):
            # How many points to place on this segment
            segment_points = max(1, int(length / spacing))
            
            for j in range(segment_points):
                if len(boundary_points) >= num_points:
                    break
                    
                # Calculate position along this segment
                t = j / segment_points
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                boundary_points.append((x, y))
        
        self.boundary_points = boundary_points

    @classmethod
    def from_uri(cls, uri: URIRef, is_support_zone: bool = False) -> 'Room':
        """Create a Room instance from a URI, extracting room number if possible."""
        room_number = None
        room_str = str(uri)
        if "roomname_" in room_str:
            try:
                room_number = room_str.split("roomname_")[-1]
            except Exception:
                pass
                
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

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        room_dict = {
            "uri": str(self.uri),
            "room_number": self.room_number,
            "is_support_zone": self.is_support_zone,
            "floor": str(self.floor) if self.floor else None,
            "device_count": len(self.devices),
            
            # Attributes from CSV
            "isRoom": self.isRoom,
            "hasWindows": self.hasWindows,
            "hasWindowsFacingDirection": self.hasWindowsFacingDirection,
            
            # Spatial relationships
            "adjacent_rooms": self.adjacent_rooms
        }
        
        # Add spatial data if available
        if self.wkt_polygon:
            room_dict["wkt_polygon"] = self.wkt_polygon
        if self.metric_area is not None:
            room_dict["metric_area"] = self.metric_area
        if self.altitude is not None:
            room_dict["altitude"] = self.altitude
        
        # Add centroid if available
        if self.centroid_x is not None and self.centroid_y is not None:
            room_dict["centroid"] = (self.centroid_x, self.centroid_y)
        
        # Add shape metrics if available
        if self.perimeter is not None:
            room_dict["perimeter"] = self.perimeter
        if self.compactness is not None:
            room_dict["compactness"] = self.compactness
        if self.width is not None and self.height is not None:
            room_dict["dimensions"] = (self.width, self.height)
        if self.rect_fit is not None:
            room_dict["rect_fit"] = self.rect_fit
        if self.aspect_ratio is not None:
            room_dict["aspect_ratio"] = self.aspect_ratio
        
        return room_dict
    
    def to_rdf_triples(self):
        """
        Yield RDF triples representing all room properties and attributes.
        
        This method generates a complete RDF representation of the Room object,
        including all available attributes regardless of their original source.
        """
        # Define the room as a BuildingSpace
        yield (self.uri, RDF.type, self.S4BLDG.BuildingSpace)
        
        # Add a comment for the type (room vs. support_zone)
        comment = "support_zone" if self.is_support_zone else "room"
        yield (self.uri, RDFS.comment, Literal(comment))
        
        # Add room number as label if available
        if self.room_number:
            yield (self.uri, RDFS.label, Literal(self.room_number))
        
        # Connect to floor
        if self.floor:
            yield (self.uri, self.S4BLDG.isSpaceOf, self.floor)
            
        # Devices contained in this room
        for device_uri in self.devices:
            yield (self.uri, self.S4BLDG.contains, device_uri)
            yield (device_uri, self.S4BLDG.isContainedIn, self.uri)
        
        # Add spatial data
        if self.wkt_polygon:
            # Create a geometry node for the WKT
            geometry_uri = URIRef(f"{str(self.uri)}_geometry")
            yield (self.uri, self.GEOSPARQL.hasGeometry, geometry_uri)
            yield (geometry_uri, self.GEOSPARQL.asWKT, Literal(self.wkt_polygon))
        
        # Add metric area if available
        if self.metric_area is not None:
            yield (self.uri, self.GEOSPARQL.hasMetricArea, Literal(self.metric_area))
        
        # Add altitude if available
        if self.altitude is not None:
            yield (self.uri, self.GEO1.alt, Literal(self.altitude))
        
        # Add CSV enrichment data
        if self.isRoom is not None:
            yield (self.uri, self.EX.isRoom, Literal(self.isRoom))
        
        # Add derived spatial metrics
        if self.centroid_x is not None and self.centroid_y is not None:
            centroid_wkt = f"POINT({self.centroid_x} {self.centroid_y})"
            yield (self.uri, self.EXONT.hasCentroid, Literal(centroid_wkt))
        
        if self.perimeter is not None:
            yield (self.uri, self.EXONT.hasPerimeter, Literal(self.perimeter))
        
        if self.compactness is not None:
            yield (self.uri, self.EXONT.hasCompactness, Literal(self.compactness))
        
        if self.width is not None:
            yield (self.uri, self.EXONT.hasWidth, Literal(self.width))
        
        if self.height is not None:
            yield (self.uri, self.EXONT.hasHeight, Literal(self.height))
        
        if self.rect_fit is not None:
            yield (self.uri, self.EXONT.hasRectangularFit, Literal(self.rect_fit))
        
        if self.aspect_ratio is not None:
            yield (self.uri, self.EXONT.hasAspectRatio, Literal(self.aspect_ratio))
        
        # Add window information if available
        if self.hasWindows:
            yield (self.uri, self.EXONT.hasWindows, Literal(True))
            
            # Add window facing directions
            if self.hasWindowsFacingDirection:
                for i, heading in enumerate(self.hasWindowsFacingDirection):
                    # Create a window instance URI
                    window_uri = URIRef(f"{str(self.uri)}_window_{i}")
                    
                    # Link window to room
                    yield (self.uri, self.EXONT.hasWindow, window_uri)
                    
                    # Add window type
                    yield (window_uri, RDF.type, self.BOT.Element)
                    
                    # Add window properties
                    yield (window_uri, self.EXONT.hasFacingDirection, Literal(heading, datatype=RDFS.XSD.integer))
                    
                    # Determine relative direction based on heading
                    # This maps headings to relative directions (approximation)
                    relative_dir = None
                    if 315 <= heading < 45 or heading == 0:
                        relative_dir = "front"
                    elif 45 <= heading < 135:
                        relative_dir = "right"
                    elif 135 <= heading < 225:
                        relative_dir = "back"
                    elif 225 <= heading < 315:
                        relative_dir = "left"
                    
                    if relative_dir:
                        yield (window_uri, self.EXONT.facingRelativeDirection, Literal(relative_dir))
        elif self.hasWindows is not None and not self.hasWindows:
            yield (self.uri, self.EXONT.hasWindows, Literal(False))
        
    def __str__(self):
        # only show URI, room number, floor, device count and area
        floor_display = 'n/a'
        if self.floor:
            floor_str = str(self.floor)
            if 'floor_' in floor_str:
                floor_display = floor_str.split('floor_')[-1].strip()
            else:
                floor_display = floor_str.split('/')[-1].strip()
        
        return (
            f"Room {self.room_number or self.uri!r} │ "
            f"floor={floor_display} │ "
            f"devices={len(self.devices)} │ "
            f"area={self.metric_area or 'n/a'} m²"
        )

    def __repr__(self):
        lines = ["Room("]
        # Core attributes
        for attr in [
            f"    uri={self.uri!r},",
            f"    room_number={self.room_number!r},",
            f"    is_support_zone={self.is_support_zone!r},",
            f"    floor={self.floor!r},",
            f"    n_devices={len(self.devices)!r},",
            f"    isRoom={self.isRoom!r},",
            f"    hasWindows={self.hasWindows!r},",
            f"    hasWindowsFacingDirection={self.hasWindowsFacingDirection!r},",
            f"    adjacent_rooms={self.adjacent_rooms!r},",
            f"    wkt_polygon={self.wkt_polygon!r},",
            f"    metric_area={self.metric_area!r},",
            f"    altitude={self.altitude!r},",
        ]:
            lines.append(attr)

        # polygon_2d
        if self.polygon_2d:
            lines.append("    polygon_2d=[")
            for pt in self.polygon_2d:
                lines.append(f"        {pt!r},")
            lines.append("    ],")
        else:
            lines.append(f"    polygon_2d={self.polygon_2d!r},")

        # polygon_3d
        if self.polygon_3d:
            lines.append("    polygon_3d=[")
            for pt in self.polygon_3d:
                lines.append(f"        {pt!r},")
            lines.append("    ],")
        else:
            lines.append(f"    polygon_3d={self.polygon_3d!r},")

        # centroid and other scalars
        lines.extend([
            f"    centroid=({self.centroid_x!r}, {self.centroid_y!r}),",
            f"    perimeter={self.perimeter!r},",
            f"    compactness={self.compactness!r},",
            f"    width={self.width!r},",
            f"    height={self.height!r},",
            f"    rect_fit={self.rect_fit!r},",
            f"    aspect_ratio={self.aspect_ratio!r},",
        ])

        # boundary_points
        if self.boundary_points:
            lines.append("    boundary_points=[")
            for pt in self.boundary_points:
                lines.append(f"        {pt!r},")
            lines.append("    ]")
        else:
            lines.append(f"    boundary_points={self.boundary_points!r}")

        lines.append(")")
        return "\n".join(lines)
    
    def __hash__(self):
        return hash(self.uri)
    
    def __eq__(self, other):
        """Check equality based on URI."""
        if isinstance(other, Room):
            return self.uri == other.uri
        return False
