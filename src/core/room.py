from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Tuple
from rdflib import URIRef, Literal, Namespace


@dataclass(slots=True)
class Room:
    uri: URIRef
    name: Optional[str] = None
    floor: Optional[URIRef] = None
    room_number: Optional[str] = None
    devices: Set[URIRef] = field(default_factory=set)
    coordinates: Optional[Tuple[float, float]] = None
    area: Optional[float] = None
    room_type: Optional[str] = None

    def add_device(self, device_uri: URIRef) -> None:
        self.devices.add(device_uri)

    def remove_device(self, device_uri: URIRef) -> None:
        self.devices.discard(device_uri)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "uri": str(self.uri),
            "name": self.name,
            "floor": str(self.floor) if self.floor else None,
            "room_number": self.room_number,
            "device_count": len(self.devices),
            "coordinates": self.coordinates,
            "area": self.area,
            "room_type": self.room_type
        }
    
    def to_rdf_triples(self):
        """Yield RDF triples representing this room."""
        saref4bldg = Namespace("https://saref.etsi.org/saref4bldg/")
        yield (self.uri, saref4bldg.hasRoomName, Literal(self.name) if self.name else Literal(""))
        if self.floor:
            yield (self.uri, saref4bldg.isOnFloor, self.floor)

    def __repr__(self):
        return f"Room({self.uri}, floor={self.floor}, devices={len(self.devices)})"
    
    def __hash__(self):
        return hash(self.uri)
    
    def __eq__(self, other):
        """Check equality based on URI."""
        if isinstance(other, Room):
            return self.uri == other.uri
        return False
