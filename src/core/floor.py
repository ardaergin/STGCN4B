from dataclasses import dataclass, field
from typing import Optional, Dict, Set
from rdflib import URIRef, Literal, Namespace, RDF, RDFS


@dataclass(slots=True)
class Floor:
    uri: URIRef
    floor_number: Optional[int] = None
    building: Optional[URIRef] = None
    rooms: Set[URIRef] = field(default_factory=set)
    
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
        }
    
    def to_rdf_triples(self):
        """Yield RDF triples representing this floor."""
        s4bldg = Namespace("https://saref.etsi.org/saref4bldg/")
        
        # In s4bldg, floors don't appear to have an explicit class,
        # so we also define a floor as a "BuildingSpace".
        yield (self.uri, RDF.type, s4bldg.BuildingSpace)
        
        # Add a label with the floor number
        if self.floor_number is not None:
            yield (self.uri, RDFS.label, Literal(f"Floor {self.floor_number}"))
        
        # Connect to building
        if self.building:
            yield (self.uri, s4bldg.isSpaceOf, self.building)
        
        # Rooms in this floor
        for room_uri in self.rooms:
            yield (room_uri, s4bldg.isSpaceOf, self.uri)
    
    def __repr__(self):
        return f"Floor({self.uri}, number={self.floor_number}, rooms={len(self.rooms)})"
    
    def __hash__(self):
        return hash(self.uri)
    
    def __eq__(self, other):
        """Check equality based on URI."""
        if isinstance(other, Floor):
            return self.uri == other.uri
        return False
