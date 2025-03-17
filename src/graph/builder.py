from typing import Dict, List, Tuple, Optional
import numpy as np
import networkx as nx
from rdflib import URIRef

from .officegraph import OfficeGraph

class OfficeGraphBuilder:
    """
    A class that constructs various graph representations (adjacency matrices, 
    heterogeneous graphs) from OfficeGraph data (devices, rooms, floors, measurements).
    """

    def __init__(self, office_graph: OfficeGraph):
        self.office_graph = office_graph

    def build_room_adjacency(self) -> Tuple[np.ndarray, List[URIRef]]:
        """
        Build an adjacency matrix indicating which rooms share the same floor.
        
        Returns:
            adjacency: A 2D numpy array (n x n) with adjacency = 1 if two rooms are on the same floor
            room_uris: The list of room URIs corresponding to each row/column index in adjacency
        """
        room_uris = list(self.office_graph.rooms.keys())
        n = len(room_uris)
        adjacency = np.zeros((n, n), dtype=int)

        # Mark adjacency if two rooms are on the same floor
        for floor_uri, floor in self.office_graph.floors.items():
            # floor.rooms is a set of Room objects – we need to see which are in room_uris
            # or, if your Floor object stores URIs instead, adapt accordingly.
            floor_room_indices = []
            for i, r_uri in enumerate(room_uris):
                # floor.rooms is a set of Room objects, so we check r_uri in [room.uri for room in floor.rooms]
                # or if your floor.rooms is a set of URIs, you can check directly:
                if r_uri in floor.rooms:
                    floor_room_indices.append(i)

            # link them in adjacency
            for i in floor_room_indices:
                for j in floor_room_indices:
                    if i != j:
                        adjacency[i, j] = 1

        return adjacency, room_uris

    def build_device_adjacency(self) -> Tuple[np.ndarray, List[URIRef]]:
        """
        Build an adjacency matrix indicating which devices share the same room.
        
        Args:
            device_uris: optional subset of device URIs. If None, uses all devices.
        
        Returns:
            adjacency: A 2D numpy array (m x m) with adjacency=1 if two devices share the same room
            device_uris: The list of device URIs corresponding to each row/column in adjacency
        """
        device_uris = list(self.office_graph.devices.keys())

        n = len(device_uris)
        adjacency = np.zeros((n, n), dtype=int)

        for i, dev1 in enumerate(device_uris):
            d1_obj = self.office_graph.devices[dev1]
            for j, dev2 in enumerate(device_uris):
                if i == j:
                    continue
                d2_obj = self.office_graph.devices[dev2]
                # If they share the same room and neither is None
                if d1_obj.room and d2_obj.room and (d1_obj.room == d2_obj.room):
                    adjacency[i, j] = 1

        return adjacency, device_uris

    def build_heterogeneous_graph(self) -> nx.MultiDiGraph:
        """
        Build a NetworkX MultiDiGraph with nodes for devices, rooms, floors, measurements, etc.
        and edges that represent their relationships.
        
        Returns:
            A MultiDiGraph containing the combined information from devices, rooms, floors, and measurements.
        """
        G = nx.MultiDiGraph()

        # Add devices
        for uri, device in self.office_graph.devices.items():
            G.add_node(uri, type='device', **device.to_dict())

            # Connect device -> room
            if device.room and device.room in self.office_graph.rooms:
                G.add_edge(uri, device.room, type='located_in')
                # Possibly add the room node with its attributes
                G.add_node(device.room, type='room', **self.office_graph.rooms[device.room].to_dict())

                # Connect room -> floor
                room_obj = self.office_graph.rooms[device.room]
                if room_obj.floor and room_obj.floor in self.office_graph.floors:
                    G.add_edge(device.room, room_obj.floor, type='part_of')
                    G.add_node(room_obj.floor, type='floor', **self.office_graph.floors[room_obj.floor].to_dict())

                    # Connect floor -> building (if building is a URI)
                    floor_obj = self.office_graph.floors[room_obj.floor]
                    if floor_obj.building:
                        G.add_edge(room_obj.floor, floor_obj.building, type='part_of')
                        G.add_node(floor_obj.building, type='building')

        # Add measurements + connections
        for meas_uri, measurement in self.office_graph.measurements.items():
            # measurement node
            G.add_node(meas_uri, type='measurement', **measurement.to_dict())

            # edge device -> measurement
            G.add_edge(measurement.device_uri, meas_uri, type='makes_measurement')

            # measurement -> property
            if measurement.property_type:
                G.add_edge(meas_uri, measurement.property_type, type='relates_to_property')
                G.add_node(measurement.property_type, type='property')

            # next / previous measurement edges
            if measurement.next_meas_uri:
                G.add_edge(meas_uri, measurement.next_meas_uri, type='next')
            if measurement.prev_meas_uri:
                G.add_edge(meas_uri, measurement.prev_meas_uri, type='previous')

        return G
