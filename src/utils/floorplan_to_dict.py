import xml.etree.ElementTree as ET
import re
import numpy as np
from svg.path import parse_path, Line

def svg_to_room_polygons(svg_file_path, sample_density=10):
    """
    Extract room polygons from an SVG file created in Inkscape.
    
    Args:
        svg_file_path (str): Path to the SVG file
        sample_density (int): Number of points to sample per curve segment
        
    Returns:
        dict: Dictionary with room names as keys and polygon points as values
    """
    # Parse the SVG file
    namespaces = {
        'svg': 'http://www.w3.org/2000/svg',
        'inkscape': 'http://www.inkscape.org/namespaces/inkscape'
    }
    
    tree = ET.parse(svg_file_path)
    root = tree.getroot()
    
    # Find the "rooms" layer - look specifically for layer1 with inkscape:label="rooms"
    rooms_layer = None
    for g in root.findall('.//svg:g', namespaces):
        # Check if this is the rooms layer by looking at inkscape:label
        label = g.get('{http://www.inkscape.org/namespaces/inkscape}label')
        g_id = g.get('id')
        
        if (label and label == 'rooms') or (g_id and g_id == 'layer1' and label == 'rooms'):
            rooms_layer = g
            break
    
    if rooms_layer is None:
        # If no specific "rooms" layer, raise an error
        raise ValueError("No 'rooms' layer found in the SVG file.")
    
    # Dictionary to store room polygons
    room_polygons = {}
    
    # Process each path in the rooms layer
    for path_elem in rooms_layer.findall('./svg:path', namespaces):
        # Get the room name from inkscape:label
        room_name = path_elem.get('{http://www.inkscape.org/namespaces/inkscape}label')
        
        # If no label, try ID
        if not room_name:
            room_name = path_elem.get('id')
            
        # If still no name, use a default
        if not room_name:
            room_name = f"Room_{len(room_polygons) + 1}"
        
        # Get the path data
        path_data = path_elem.get('d')
        if not path_data:
            continue
        
        # Parse the path and extract points
        try:
            path = parse_path(path_data)
            
            # Sample points along the path
            points = []
            
            # For each segment in the path
            for segment in path:
                # For straight lines, just use the endpoints
                if isinstance(segment, Line):
                    points.append((segment.start.real, segment.start.imag))
                else:
                    # For curves and other shapes, sample points
                    for i in range(sample_density):
                        t = i / (sample_density - 1)
                        point = segment.point(t)
                        points.append((point.real, point.imag))
            
            # Add the last point
            if path:
                points.append((path[-1].end.real, path[-1].end.imag))
            
            # Remove duplicate consecutive points and round to improve precision
            unique_points = []
            for p in points:
                p_rounded = (round(p[0], 2), round(p[1], 2))
                if not unique_points or p_rounded != (round(unique_points[-1][0], 2), round(unique_points[-1][1], 2)):
                    unique_points.append(p)
            
            # Store in dictionary
            room_polygons[room_name] = unique_points
            
        except Exception as e:
            print(f"Error parsing path for room {room_name}: {e}")
            # Try a more basic approach as fallback
            try:
                # Extract vertices from path data using regex
                vertices = []
                commands = re.findall(r'([mMhHvVlLzZ])\s*([^mMhHvVlLzZ]*)', path_data)
                
                current_x, current_y = 0, 0
                start_x, start_y = 0, 0
                
                for cmd, coords in commands:
                    coords = coords.strip()
                    
                    if cmd == 'm':  # relative moveto
                        parts = re.findall(r'(-?\d+\.?\d*)[,\s](-?\d+\.?\d*)', coords)
                        if parts:
                            x, y = float(parts[0][0]), float(parts[0][1])
                            current_x += x
                            current_y += y
                            vertices.append((current_x, current_y))
                            start_x, start_y = current_x, current_y
                    
                    elif cmd == 'M':  # absolute moveto
                        parts = re.findall(r'(-?\d+\.?\d*)[,\s](-?\d+\.?\d*)', coords)
                        if parts:
                            current_x, current_y = float(parts[0][0]), float(parts[0][1])
                            vertices.append((current_x, current_y))
                            start_x, start_y = current_x, current_y
                    
                    elif cmd == 'h':  # relative horizontal lineto
                        for x in re.findall(r'(-?\d+\.?\d*)', coords):
                            current_x += float(x)
                            vertices.append((current_x, current_y))
                    
                    elif cmd == 'H':  # absolute horizontal lineto
                        for x in re.findall(r'(-?\d+\.?\d*)', coords):
                            current_x = float(x)
                            vertices.append((current_x, current_y))
                    
                    elif cmd == 'v':  # relative vertical lineto
                        for y in re.findall(r'(-?\d+\.?\d*)', coords):
                            current_y += float(y)
                            vertices.append((current_x, current_y))
                    
                    elif cmd == 'V':  # absolute vertical lineto
                        for y in re.findall(r'(-?\d+\.?\d*)', coords):
                            current_y = float(y)
                            vertices.append((current_x, current_y))
                    
                    elif cmd in ['z', 'Z']:  # closepath
                        if vertices and (current_x, current_y) != (start_x, start_y):
                            vertices.append((start_x, start_y))
                
                room_polygons[room_name] = vertices
            except Exception as e2:
                print(f"Failed to parse path for room {room_name} using fallback method: {e2}")
                continue
    
    return room_polygons

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

def save_room_polygons(room_polygons, output_file):
    """
    Save room polygons dictionary to a Python file.
    
    Args:
        room_polygons (dict): Dictionary with room names as keys and polygon points as values
        output_file (str): Path to the output Python file
    """
    with open(output_file, 'w') as f:
        f.write("# Room polygons extracted from SVG floor plan\n\n")
        f.write("room_polygons = {\n")
        
        for room_name, points in room_polygons.items():
            f.write(f"    '{room_name}': [\n")
            for point in points:
                f.write(f"        ({point[0]}, {point[1]}),\n")
            f.write("    ],\n")
        
        f.write("}\n")

# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract room polygons from an Inkscape SVG file.")

    parser.add_argument(
        "--svg_file", type=str,
        default="data/floor_plan/floor_7.svg",
        help="Path to the input SVG file (default: data/floor_plan/floor_7.svg)"
    )
    parser.add_argument(
        "--output_file", type=str,
        default="data/floor_plan/room_polygons.py",
        help="Path to the output Python file (default: data/floor_plan/room_polygons.py)"
    )
    parser.add_argument(
        "--epsilon", type=float,
        default=0.0,
        help="Simplification epsilon (optional, default: no simplification)"
    )

    args = parser.parse_args()
    
    try:
        # Extract room polygons from SVG
        room_polygons = svg_to_room_polygons(args.svg_file)
        
        if not room_polygons:
            print("Warning: No rooms were extracted. Check if the SVG has paths with inkscape:label attributes.")
        
        # Optionally simplify polygons to reduce number of points
        simplified_room_polygons = {}
        for room, polygon in room_polygons.items():
            simplified_room_polygons[room] = simplify_polygon(polygon, epsilon=args.epsilon)
        print(f"Simplified polygons with epsilon = {args.epsilon}")
        
        # Save to Python file
        save_room_polygons(simplified_room_polygons, args.output_file)
        
        # Print summary
        print(f"Extracted {len(room_polygons)} rooms from {args.svg_file}")
        for room, polygon in simplified_room_polygons.items():
            print(f"  {room}: {len(polygon)} points")
        print(f"Saved room polygons to {args.output_file}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("The SVG structure should be something like:")
        print("""
  <g
     inkscape:groupmode="layer"
     id="layer1"
     inkscape:label="rooms"
     style="display:inline">
    <path
       style="fill:#000000;stroke-linejoin:round"
       d="m 372.32366,632.4232 h 82.15301 V 528.56939 h 42.16155 V 404.25483 H 375.42377 Z"
       id="path3874"
       inkscape:label="7.066" />
    ...
        """)
        print("\nDouble-check that the SVG has this structure with 'inkscape:label' for room names.")
