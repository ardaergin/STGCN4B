from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
from rdflib import URIRef, Literal, Namespace, RDF, RDFS


@dataclass(slots=True)
class Room:
    uri: URIRef # In the RDF files, they are written as, e.g., "ic:roomname_7.003", so uri is the full "ic:roomname_7.003"
    room_number: Optional[str] = None # the room number is just the number: "7.003"
    is_support_zone: bool = False  # True for support_zone, False for room
    floor: Optional[URIRef] = None
    devices: Set[URIRef] = field(default_factory=set)
    
    # Floor plan attributes:
    x_1: Optional[int] = None
    x_2: Optional[int] = None
    y_1: Optional[str] = None
    y_2: Optional[str] = None
    size_approx: Optional[float] = None
    isRoom: Optional[bool] = None
    isFacing: Optional[List[str]] = None
    adjacent_rooms: List[str] = field(default_factory=list)

    # Polygon data
    Polygon: List[int] = field(default_factory=list)
    
    # Define namespace constants at class level for reuse
    S4BLDG = Namespace("https://saref.etsi.org/saref4bldg/")
    
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
        return {
            "uri": str(self.uri),
            "room_number": self.room_number,
            "is_support_zone": self.is_support_zone,
            "floor": str(self.floor) if self.floor else None,
            "device_count": len(self.devices),
            # Floor plan attributes:
            "x_1": self.x_1,
            "x_2": self.x_2,
            "y_1": self.y_1,
            "y_2": self.y_2,
            "size_approx": self.size_approx,
            "isRoom": self.isRoom,
            "isFacing": self.isFacing,
            "adjacent_rooms": self.adjacent_rooms,
            # Polygon data:
            "Polygon": self.Polygon
        }
    
    def to_rdf_triples(self):
        """Yield RDF triples representing this room."""
        # Define the room as a BuildingSpace
        yield (self.uri, RDF.type, self.S4BLDG.BuildingSpace)
        
        # Add a comment for the type (room vs. support_zone)
        comment = "support_zone" if self.is_support_zone else "room"
        yield (self.uri, RDFS.comment, Literal(comment))
        
        # Connect to floor
        if self.floor:
            yield (self.uri, self.S4BLDG.isSpaceOf, self.floor)
            
        # Devices contained in this room
        for device_uri in self.devices:
            yield (self.uri, self.S4BLDG.contains, device_uri)
            yield (device_uri, self.S4BLDG.isContainedIn, self.uri)
    
    def __repr__(self):
        return f"Room({self.uri}, floor={self.floor}, devices={len(self.devices)})"
    
    def __hash__(self):
        return hash(self.uri)
    
    def __eq__(self, other):
        """Check equality based on URI."""
        if isinstance(other, Room):
            return self.uri == other.uri
        return False
