import xml.etree.ElementTree as ET
import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class SVGParser:
    """Class for parsing SVG files and extracting polygon data."""
    
    def __init__(self, svg_path: str):
        """Initialize the SVG parser.
        
        Args:
            svg_path: Path to the SVG file
        """
        self.svg_path = svg_path
        self.svg_width = None
        self.svg_height = None
        self.ns = {
            'svg': 'http://www.w3.org/2000/svg',
            'sodipodi': 'http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd',
            'inkscape': 'http://www.inkscape.org/namespaces/inkscape'
        }
        
        # Parse the SVG file once to get dimensions
        self._parse_svg_dimensions()
    
    def _parse_svg_dimensions(self):
        """Parse the SVG file to get dimensions."""
        try:
            tree = ET.parse(self.svg_path)
            root = tree.getroot()
            
            # Get SVG dimensions
            viewbox = root.get('viewBox', '').split()
            if len(viewbox) == 4:
                self.svg_width = float(viewbox[2])
                self.svg_height = float(viewbox[3])
            else:
                self.svg_width = float(root.get('width', '0').replace('pt', ''))
                self.svg_height = float(root.get('height', '0').replace('pt', ''))
            
            logger.info(f"SVG dimensions: {self.svg_width}x{self.svg_height}")
        except Exception as e:
            logger.error(f"Error parsing SVG dimensions: {e}")
    
    def get_svg_dimensions(self) -> Tuple[float, float]:
        """Get the SVG dimensions.
        
        Returns:
            Tuple of (width, height)
        """
        return self.svg_width, self.svg_height
        
    def get_building_polygon(self) -> List[Tuple[float, float]]:
        """Get the building polygon from the SVG file.
        
        This function looks for a path with inkscape:label="building" at the root level.
        
        Returns:
            List of points as (x, y) tuples representing the building polygon
        """
        try:
            tree = ET.parse(self.svg_path)
            root = tree.getroot()
            
            # Look for a path directly labeled as "building" at the root level
            building_path = root.find('.//svg:path[@inkscape:label="building"]', self.ns)
            
            if building_path is not None:
                logger.info("Found building polygon at root level")
                # Get the transform for the building path
                transform = building_path.get('transform', '')
                transform_values = [0, 0]
                if transform:
                    match = re.search(r'translate\(([-\d\.]+),([-\d\.]+)\)', transform)
                    if match:
                        transform_values = [float(match.group(1)), float(match.group(2))]
                
                # Parse the path data
                d = building_path.get('d', '')
                if d:
                    points = self._parse_svg_path(d, transform_values)
                    logger.info(f"Extracted building polygon with {len(points)} points")
                    return points
            
            # If we get here, no building polygon was found
            logger.warning("No building polygon found in SVG")
            return []
        
        except Exception as e:
            logger.error(f"Error extracting building polygon: {e}")
            return []
            
    def get_room_polygons(self, floor_numbers: List[int]) -> Dict[int, Dict[str, List[Tuple[float, float]]]]:
        """Get room polygons for specified floor numbers.
        
        Args:
            floor_numbers: List of floor numbers to extract
            
        Returns:
            Dictionary mapping floor numbers to room IDs to polygon points
        """
        try:
            tree = ET.parse(self.svg_path)
            root = tree.getroot()
            
            # Store room polygons per floor
            rooms_by_floor = {floor: {} for floor in floor_numbers}
            
            # Process each floor layer
            for floor_number in floor_numbers:
                floor_layer_name = f"floor_{floor_number}"
                floor_layer = root.find(f'.//svg:g[@inkscape:label="{floor_layer_name}"]', self.ns)
                
                if floor_layer is None:
                    # Try with just the layer name as the number
                    floor_layer = root.find(f'.//svg:g[@inkscape:label="{floor_number}"]', self.ns)
                
                if floor_layer is None:
                    logger.warning(f"No '{floor_layer_name}' layer found in the SVG. Skipping floor {floor_number}.")
                    continue
                    
                # Get transform values for the floor layer
                transform = floor_layer.get('transform', '')
                transform_values = [0, 0]
                if transform:
                    match = re.search(r'translate\(([-\d\.]+),([-\d\.]+)\)', transform)
                    if match:
                        transform_values = [float(match.group(1)), float(match.group(2))]
                
                logger.info(f"Floor {floor_number} transform values: {transform_values}")
                
                # Extract room paths from floor layer
                for path in floor_layer.findall('.//svg:path', self.ns):
                    room_id = path.get('{http://www.inkscape.org/namespaces/inkscape}label', '')
                    if not room_id or room_id == 'building':
                        continue
                        
                    d = path.get('d', '')
                    if d:
                        # Parse the path data to extract points
                        points = self._parse_svg_path(d, transform_values)
                        
                        # Store the room polygon
                        rooms_by_floor[floor_number][room_id] = points
            
            # Log summary of extracted polygons
            total_rooms = sum(len(rooms) for rooms in rooms_by_floor.values())
            logger.info(f"Extracted {total_rooms} room polygons across {len(floor_numbers)} floors")
            for floor, rooms in rooms_by_floor.items():
                logger.info(f"  Floor {floor}: {len(rooms)} rooms")
                
            return rooms_by_floor
        
        except Exception as e:
            logger.error(f"Error extracting room polygons: {e}")
            return {floor: {} for floor in floor_numbers}
    
    def _parse_svg_path(self, d_attr: str, transform: List[float]) -> List[Tuple[float, float]]:
        """Parse SVG path data to extract polygon points.
        
        This is a simplified approach that works for most basic paths in Inkscape.
        For more complex paths, you might need a more sophisticated parser.
        
        Args:
            d_attr: The SVG path data string
            transform: The transform values [tx, ty]
            
        Returns:
            List of points as (x, y) tuples
        """
        points = []
        
        # Split the path data into commands and parameters
        parts = re.findall(r'([mMlLhHvVzZ])([-\d\.\s,]*)', d_attr)
        
        current_x, current_y = 0, 0
        start_x, start_y = 0, 0
        
        for cmd, params in parts:
            params = params.strip()
            if not params and cmd.lower() != 'z':
                continue
                
            # Parse parameters into a list of values
            if ',' in params:
                # Format: x1,y1 x2,y2 ...
                param_values = []
                for pair in params.split():
                    if ',' in pair:
                        x, y = map(float, pair.split(','))
                        param_values.extend([x, y])
            else:
                # Format: x1 y1 x2 y2 ...
                param_values = list(map(float, params.split()))
            
            # Handle different path commands
            if cmd == 'm':  # relative moveto
                current_x += param_values[0]
                current_y += param_values[1]
                start_x, start_y = current_x, current_y
                
                # Add the initial point
                points.append((current_x - transform[0], current_y - transform[1]))
                
                # If there are more pairs, they are treated as relative lineto commands
                for i in range(2, len(param_values), 2):
                    current_x += param_values[i]
                    current_y += param_values[i+1]
                    points.append((current_x - transform[0], current_y - transform[1]))
                    
            elif cmd == 'M':  # absolute moveto
                current_x = param_values[0]
                current_y = param_values[1]
                start_x, start_y = current_x, current_y
                
                # Add the initial point
                points.append((current_x - transform[0], current_y - transform[1]))
                
                # If there are more pairs, they are treated as absolute lineto commands
                for i in range(2, len(param_values), 2):
                    current_x = param_values[i]
                    current_y = param_values[i+1]
                    points.append((current_x - transform[0], current_y - transform[1]))
            
            elif cmd == 'l':  # relative lineto
                for i in range(0, len(param_values), 2):
                    current_x += param_values[i]
                    current_y += param_values[i+1]
                    points.append((current_x - transform[0], current_y - transform[1]))
            
            elif cmd == 'L':  # absolute lineto
                for i in range(0, len(param_values), 2):
                    current_x = param_values[i]
                    current_y = param_values[i+1]
                    points.append((current_x - transform[0], current_y - transform[1]))
            
            elif cmd == 'h':  # relative horizontal lineto
                for value in param_values:
                    current_x += value
                    points.append((current_x - transform[0], current_y - transform[1]))
            
            elif cmd == 'H':  # absolute horizontal lineto
                for value in param_values:
                    current_x = value
                    points.append((current_x - transform[0], current_y - transform[1]))
            
            elif cmd == 'v':  # relative vertical lineto
                for value in param_values:
                    current_y += value
                    points.append((current_x - transform[0], current_y - transform[1]))
            
            elif cmd == 'V':  # absolute vertical lineto
                for value in param_values:
                    current_y = value
                    points.append((current_x - transform[0], current_y - transform[1]))
            
            elif cmd.lower() == 'z':  # closepath
                # Close the path by adding the start point if needed
                if points and (points[0][0] != current_x or points[0][1] != current_y):
                    points.append((start_x - transform[0], start_y - transform[1]))
        
        return points
