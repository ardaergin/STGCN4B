from dataclasses import dataclass, field
from typing import Optional, Dict, Set
from rdflib import URIRef, Literal, Namespace


@dataclass(slots=True)
class Floor:
    uri: URIRef
    floor_number: Optional[int] = None
    building: Optional[URIRef] = None
    rooms: Set[URIRef] = field(default_factory=set)
    floor_plan_image: Optional[str] = None
    area: Optional[float] = None
    
    def add_room(self, room_uri: URIRef) -> None:
        self.rooms.add(room_uri)

    def remove_room(self, room_uri: URIRef) -> None:
        self.rooms.discard(room_uri)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "uri": str(self.uri),
            "floor_number": self.floor_number,
            "building": str(self.building) if self.building else None,
            "room_count": len(self.rooms),
            "floor_plan_image": self.floor_plan_image,
            "area": self.area
        }

    def to_rdf_triples(self):
        """Yield RDF triples representing this floor."""
        saref4bldg = Namespace("https://saref.etsi.org/saref4bldg/")
        yield (self.uri, saref4bldg.hasFloorNumber, Literal(self.floor_number) if self.floor_number is not None else Literal(-1))
        if self.building:
            yield (self.uri, saref4bldg.isPartOfBuilding, self.building)

    def __repr__(self):
        return f"Floor({self.uri}, number={self.floor_number}, rooms={len(self.rooms)})"

    def __hash__(self):
        return hash(self.uri)
    
    def __eq__(self, other):
        """Check equality based on URI."""
        if isinstance(other, Floor):
            return self.uri == other.uri
        return False
