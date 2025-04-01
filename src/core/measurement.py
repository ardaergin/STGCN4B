from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from rdflib import URIRef, Namespace, Literal, RDF
import datetime


@dataclass(order=True, slots=True)
class Measurement:
    meas_uri: URIRef = field(compare=True)
    device_uri: URIRef = field(compare=False)
    timestamp: datetime.datetime = field(compare=True)
    value: float = field(compare=True)
    unit: Optional[URIRef] = field(default=None, compare=False)
    property_type: Optional[URIRef] = field(default=None, compare=False)
    next_meas_uri: Optional[URIRef] = field(default=None, compare=False)
    prev_meas_uri: Optional[URIRef] = field(default=None, compare=False)

    ## For the classification task (from the OfficeGraph publication) ##
    isWorkhour: bool = field(default=False, init=False, compare=False)

    def __post_init__(self):
        """Ensuring that the timestamp attribute is a datetime instance."""
        if not isinstance(self.timestamp, datetime.datetime):
            raise TypeError("timestamp must be a datetime.datetime instance")

        ## For the classification task (from the OfficeGraph publication) ##
        # Compute whether it’s a weekday between 9 and 17 (i.e., 9 <= hour < 17).
        day_of_week = self.timestamp.weekday()  # Monday=0, Sunday=6
        hour_of_day = self.timestamp.hour
        self.isWorkhour = (day_of_week < 5) and (9 <= hour_of_day < 17)

    def get_rounded_value_uri(self, decimal_places: int = 1) -> URIRef:
        """Get URI representation of the rounded measurement value."""
        val_rounded = round(self.value, decimal_places)
        return URIRef(f"https://interconnectproject.eu/example/rounded_value_{val_rounded}")

    def get_rounded_timestamp_uri(self, granularity: str = 'hour') -> URIRef:
        """
        Get URI representation of the rounded timestamp.
        
        Args:
            granularity: Time granularity ('hour', 'day', etc.)
        """
        if granularity == 'hour':
            dt_rounded = self.timestamp.replace(minute=0, second=0, microsecond=0)
            dt_str = dt_rounded.strftime("%Y-%m-%d_%H0000")
        elif granularity == 'day':
            dt_rounded = self.timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            dt_str = dt_rounded.strftime("%Y-%m-%d_000000")
        else:
            dt_rounded = self.timestamp.replace(minute=0, second=0, microsecond=0)
            dt_str = dt_rounded.strftime("%Y-%m-%d_%H0000")
            
        return URIRef(f"https://interconnectproject.eu/example/{dt_str}")
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "meas_uri": str(self.meas_uri),
            "device_uri": str(self.device_uri),
            "timestamp": self.timestamp,
            "value": self.value,
            "unit": str(self.unit) if self.unit else None,
            "property_type": str(self.property_type) if self.property_type else None,
            "next_meas_uri": str(self.next_meas_uri) if self.next_meas_uri else None,
            "prev_meas_uri": str(self.prev_meas_uri) if self.prev_meas_uri else None,
            "isWorkhour": self.isWorkhour
        }

    def to_rdf_triples(self):
        """Yield RDF triples representing this measurement."""
        saref = Namespace("https://saref.etsi.org/core/")
        # Define the measurement as a Measurement
        yield (self.meas_uri, RDF.type, saref.Measurement)
        # Use the exact predicates from the data
        yield (self.meas_uri, saref.hasTimestamp, Literal(self.timestamp.isoformat()))
        yield (self.meas_uri, saref.hasValue, Literal(self.value))
        if self.unit:
            yield (self.meas_uri, saref.isMeasuredIn, self.unit)  # Changed from hasUnit
        if self.property_type:
            yield (self.meas_uri, saref.relatesToProperty, self.property_type)  # Changed from measuresProperty
        # This triple is typically defined in the Device triples, not measurement ones
        yield (self.device_uri, saref.makesMeasurement, self.meas_uri)
    
    def __repr__(self) -> str:
        return (
            f"Measurement(meas_uri={self.meas_uri}, "
            f"device_uri={self.device_uri}, "
            f"timestamp={self.timestamp.isoformat()}, "
            f"value={self.value}, "
            f"unit={self.unit}, "
            f"property_type={self.property_type}, "
            f"isWorkhour={self.isWorkhour})"
        )

    def __hash__(self):
        return hash(self.meas_uri)        
    
    def __eq__(self, other):
        """Check equality based on URI."""
        if isinstance(other, Measurement):
            return self.meas_uri == other.meas_uri
        return False
