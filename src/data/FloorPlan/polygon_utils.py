# wkt_utils.py
import numpy as np
import re
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def parse_wkt_polygon(wkt_string: str, transform: List[float] = None, 
                     altitude: Optional[float] = None) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, float]]]:
    """
    Parse a WKT polygon string into 2D and 3D coordinate lists.
    
    Args:
        wkt_string: The WKT polygon string to parse
        transform: Optional transformation values [tx, ty] to apply to coordinates
        altitude: Optional altitude value to use if not present in WKT
        
    Returns:
        Tuple of (polygon_2d, polygon_3d) where:
            polygon_2d is a list of (x, y) tuples
            polygon_3d is a list of (x, y, z) tuples
    """
    if not wkt_string:
        return [], []
        
    polygon_2d = []
    polygon_3d = []
    
    try:
        # Clean and extract coordinates
        coords_str = wkt_string.strip()
        
        # Handle different WKT formats
        if "POLYGON Z" in coords_str:
            # Remove 'POLYGON Z' prefix
            coords_str = coords_str.replace("POLYGON Z", "").strip()
        elif coords_str.startswith("POLYGON"):
            # Remove 'POLYGON' prefix
            coords_str = coords_str[7:].strip()
            
        # Remove outer parentheses - handles both "(())" and "()" formats
        coords_str = coords_str.strip("()")
        if coords_str.startswith("(") and coords_str.endswith(")"):
            coords_str = coords_str[1:-1]
            
        # Extract coordinates
        for point_str in coords_str.split(","):
            point = point_str.strip().split()
            
            if len(point) >= 2:  # Must have at least x, y
                x, y = float(point[0]), float(point[1])
                
                # Apply transform if provided
                if transform:
                    x -= transform[0]
                    y -= transform[1]
                    
                polygon_2d.append((x, y))
                
                if len(point) >= 3:  # Has z coordinate
                    z = float(point[2])
                    polygon_3d.append((x, y, z))
                elif altitude is not None:
                    # Use the provided altitude as z
                    polygon_3d.append((x, y, altitude))
    except Exception as e:
        logger.error(f"Error parsing WKT polygon: {e}")
        return [], []
        
    return polygon_2d, polygon_3d

def create_wkt_polygon(points: List[Tuple[float, float]], z_coord: float = 0.0) -> str:
    """
    Create a WKT POLYGON representation from 2D points with Z-coordinate.
    
    Args:
        points: List of (x, y) tuples
        z_coord: Z-coordinate for 3D polygon (height in meters)
        
    Returns:
        WKT string for the polygon
    """
    if not points:
        return ""
        
    # Format: POLYGON Z((x1 y1 z1, x2 y2 z2, ...))
    point_strings = [f"{p[0]} {p[1]} {z_coord}" for p in points]
    return f"POLYGON Z(({', '.join(point_strings)}))"

def calculate_polygon_area(points: List[Tuple[float, float]]) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula (Gauss's area formula).
    
    Args:
        points: List of (x, y) points
        
    Returns:
        Area
    """
    if not points or len(points) < 3:
        return 0.0
        
    # Apply Shoelace formula
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2.0

def calculate_polygon_centroid(points: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate the centroid of a polygon.
    
    Args:
        points: List of (x, y) points
        
    Returns:
        Tuple of (centroid_x, centroid_y) or (None, None) if calculation fails
    """
    if not points or len(points) < 3:
        return None, None
        
    # Calculate area first
    area = calculate_polygon_area(points)
    if area == 0:
        return None, None
    
    # Calculate centroid
    cx = cy = 0.0
    n = len(points)
    for i in range(n):
        j = (i + 1) % n
        factor = (points[i][0] * points[j][1] - points[j][0] * points[i][1])
        cx += (points[i][0] + points[j][0]) * factor
        cy += (points[i][1] + points[j][1]) * factor
    
    # Finalize centroid calculation
    cx = cx / (6 * area)
    cy = cy / (6 * area)
    
    return cx, cy

def calculate_shape_metrics(points: List[Tuple[float, float]]) -> dict:
    """
    Calculate various shape metrics for a polygon.
    
    Args:
        points: List of (x, y) points
        
    Returns:
        Dictionary containing:
            - area: The polygon area
            - perimeter: The polygon perimeter
            - centroid: (x, y) coordinates of the centroid
            - width: Bounding box width
            - height: Bounding box height
            - compactness: Circularity measure
            - rect_fit: How rectangular the polygon is
            - aspect_ratio: Width to height ratio
    """
    if not points or len(points) < 3:
        return {}
    
    # Get coordinates for bounding box
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Calculate bounding box
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate area
    area = calculate_polygon_area(points)
    
    # Calculate perimeter
    perimeter = 0.0
    n = len(points)
    for i in range(n):
        p1 = points[i]
        p2 = points[(i + 1) % n]
        segment_length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
        perimeter += segment_length
    
    # Calculate centroid
    centroid_x, centroid_y = calculate_polygon_centroid(points)
    
    # Compactness (circularity)
    compactness = (4 * 3.14159 * area) / (perimeter**2) if perimeter > 0 else 0
    
    # Rectangular fit
    rect_fit = area / (width * height) if width > 0 and height > 0 else 0
    
    # Aspect ratio
    aspect_ratio = width / height if height > 0 else 0
    
    return {
        'area': area,
        'perimeter': perimeter,
        'centroid': (centroid_x, centroid_y),
        'width': width,
        'height': height,
        'compactness': compactness,
        'rect_fit': rect_fit,
        'aspect_ratio': aspect_ratio
    }

def create_boundary_points(points: List[Tuple[float, float]], num_points: int = 20) -> List[Tuple[float, float]]:
    """
    Create evenly spaced points along a polygon boundary.
    
    Args:
        points: The polygon vertices as (x, y) tuples
        num_points: Number of points to generate
        
    Returns:
        List of (x, y) coordinates of boundary points
    """
    if not points or len(points) < 3:
        return []
    
    # Calculate total perimeter length and segment lengths
    segments = []
    total_length = 0.0
    
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        segment_length = ((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)**0.5
        segments.append((p1, p2, segment_length))
        total_length += segment_length
    
    # Generate evenly spaced points
    boundary_points = []
    spacing = total_length / num_points
    
    for p1, p2, length in segments:
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
    
    return boundary_points

def simplify_polygon(polygon, epsilon=0.0):
    """
    Simplify a polygon using the Ramer-Douglas-Peucker algorithm.
    This reduces the number of vertices while preserving shape.
    
    Args:
        polygon (list): List of (x, y) coordinate tuples
        epsilon (float): Maximum distance for point simplification
        
    Returns:
        list: Simplified polygon
    """
    def point_line_distance(point, start, end):
        if np.array_equal(start, end):
            return np.linalg.norm(point - start)
        
        line_vec = end - start
        point_vec = point - start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        
        # Project point to line
        point_proj_len = np.dot(point_vec, line_unitvec)
        
        if point_proj_len < 0:
            return np.linalg.norm(point - start)
        elif point_proj_len > line_len:
            return np.linalg.norm(point - end)
        else:
            # Point projection falls on line segment
            point_proj = start + point_proj_len * line_unitvec
            return np.linalg.norm(point - point_proj)
    
    def rdp(points, epsilon, first=0, last=None):
        if last is None:
            last = len(points) - 1
        
        if last - first <= 1:
            return [first, last]
        
        dmax = 0
        index = first
        
        for i in range(first + 1, last):
            d = point_line_distance(
                np.array(points[i]), 
                np.array(points[first]), 
                np.array(points[last])
            )
            if d > dmax:
                index = i
                dmax = d
        
        result = []
        if dmax > epsilon:
            rec1 = rdp(points, epsilon, first, index)
            rec2 = rdp(points, epsilon, index, last)
            
            # Concatenate results, excluding duplicates
            result = rec1[:-1] + rec2
        else:
            result = [first, last]
        
        return result
    
    indices = rdp(polygon, epsilon)
    return [polygon[i] for i in indices]
