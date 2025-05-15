from dataclasses import dataclass, field
from typing import Optional, Dict, Set, List, Tuple
from rdflib import URIRef, Namespace

from ..config.namespaces import NamespaceMixin


@dataclass(slots=True)
class Building(NamespaceMixin):
    """
    Class representing a building with real-world geographic coordinates.
    """
    name: str
    uri: Optional[URIRef] = None
    
    # Geographic coordinates
    top_left_lat: Optional[float] = None  # e.g., 51.4455556 (51°26'44"N)
    top_left_lon: Optional[float] = None  # e.g., 5.4594444 (5°27'34"E)
    bottom_right_lat: Optional[float] = None  # e.g., 51.4447222 (51°26'41"N)
    bottom_right_lon: Optional[float] = None  # e.g., 5.4605556 (5°27'38"E)
    
    # Building dimensions
    height_meters: Optional[float] = None
    length_meters: Optional[float] = None
    width_meters: Optional[float] = None
    heading: Optional[float] = None
    height_from_sealevel: Optional[float] = None
 
    # To be calculated:
    area_meters: Optional[float] = None
    
    # Floor information
    floor_heights: Optional[List[float]] = None
    floor_heights_accounting_for_sea_level: Optional[List[float]] = None
    
    # Floors in this building
    floors: Set[URIRef] = field(default_factory=set)
    number_of_floors_excluding_ground_floor: Optional[int] = None
        
    def __post_init__(self):
        """Populate building information for VideoLab."""
        if self.name == "VideoLab":
            self.uri = self.EX["VideoLab"]
            
            # Height
            self.height_meters = 41.35 # from https://apps.webmapper.nl/gebouwen/#15.9/51.445963/5.458834/-58.4/60
            
            # Number of Floors
            self.number_of_floors_excluding_ground_floor = 7
            
            # Floor height calculation
            total_number_of_floors = self.number_of_floors_excluding_ground_floor + 1
            single_floor_height = self.height_meters / (self.number_of_floors_excluding_ground_floor + 1)
            self.floor_heights = [single_floor_height] * total_number_of_floors
            
            # Height from sea level (in meters)
            self.height_from_sealevel = 23 # from https://en-gb.topographic-map.com/map-wpfnh/Netherlands/?center=51.44523%2C5.46081&zoom=18&popup=51.44543%2C5.46012
            
            # Calculate floor heights accounting for sea level
            self.floor_heights_accounting_for_sea_level = [
                self.height_from_sealevel + i * single_floor_height
                for i in range(total_number_of_floors)
            ]
            
            # Coordinates
            self.top_left_lat = 51.4455556  # 51°26'44"N
            self.top_left_lon = 5.4594444   # 5°27'34"E
            self.bottom_right_lat = 51.4447222  # 51°26'41"N
            self.bottom_right_lon = 5.4605556   # 5°27'38"E
            # i.e., so, a line between these two points would be diagonal of the building grounds.
            
            # Length/Width and heading (measured via using Google Earth)
            self.length_meters = 108.5
            self.width_meters = 43
            self.heading = 27 # Heading that the enterence of the building faces (from Google Earth)

    def add_floor(self, floor_uri: URIRef) -> None:
        """Add a floor to this building."""
        self.floors.add(floor_uri)
    
    def remove_floor(self, floor_uri: URIRef) -> None:
        """Remove a floor from this building."""
        self.floors.discard(floor_uri)
 
    def __repr__(self):
        return f"Building({self.name})"
    
    def __hash__(self):
        return hash(self.uri)
    
    def __eq__(self, other):
        """Check equality based on URI."""
        if isinstance(other, Building):
            return self.uri == other.uri
        return False
