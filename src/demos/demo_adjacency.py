"""
Demo script to test the adjacency matrices functionality.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from ..graph import OfficeGraph


def plot_adjacency_matrix(adj_matrix, titles, figsize=(10, 8), title="Adjacency Matrix"):
    """Plot an adjacency matrix as a heatmap."""
    plt.figure(figsize=figsize)
    plt.imshow(adj_matrix, cmap='Blues')
    plt.colorbar(label='Connection')
    
    # Add titles to axes
    if len(titles) <= 20:  # Only show labels if there aren't too many
        plt.xticks(range(len(titles)), [str(t).split('/')[-1] for t in titles], rotation=90)
        plt.yticks(range(len(titles)), [str(t).split('/')[-1] for t in titles])
    
    plt.title(title)
    plt.tight_layout()
    return plt


def plot_network(adj_matrix, titles, figsize=(12, 10), title="Network Graph"):
    """Plot a network graph visualization of the adjacency matrix."""
    G = nx.from_numpy_array(adj_matrix)
    
    # Relabel nodes with shortened URIs
    mapping = {i: str(uri).split('/')[-1] for i, uri in enumerate(titles)}
    G = nx.relabel_nodes(G, mapping)
    
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)  # Position nodes using force-directed layout
    
    # Draw network
    nx.draw(G, pos, with_labels=True, node_color='skyblue', 
            node_size=500, edge_color='gray', width=1, alpha=0.8)
    
    plt.title(title)
    plt.tight_layout()
    return plt


def main():
    """Run the demo to test adjacency matrices."""
    print("Initializing OfficeGraph...")
    office_graph = OfficeGraph(load_only_7th_floor=True)
    
    # 1. Get and visualize room adjacency matrix
    print("\nGenerating room adjacency matrix...")
    room_adj, room_uris = office_graph.get_room_adjacency()
    print(f"Room adjacency matrix shape: {room_adj.shape}")
    print(f"Number of room connections: {np.sum(room_adj)}")
    
    # Plot room adjacency
    plt_room = plot_adjacency_matrix(room_adj, room_uris, title="Room Adjacency Matrix")
    plt_room.savefig("../../output/images/room_adjacency_matrix.png")
    
    # Plot room network
    plt_room_net = plot_network(room_adj, room_uris, title="Room Adjacency Network")
    plt_room_net.savefig("../../output/images/room_adjacency_network.png")
    
    # 2. Get and visualize device-room adjacency matrix
    print("\nGenerating device-room adjacency matrix...")
    device_room_adj, device_uris, room_uris_all = office_graph.get_device_room_adjacency()
    print(f"Device-room adjacency matrix shape: {device_room_adj.shape}")
    print(f"Number of device-room connections: {np.sum(device_room_adj)}")
        
    # 3. Get and analyze heterogeneous graph
    print("\nBuilding heterogeneous graph...")
    hetero_graph = office_graph.get_heterogeneous_graph()
    print(f"Heterogeneous graph: {len(hetero_graph.nodes())} nodes, {len(hetero_graph.edges())} edges")
    
    # Count node types
    node_types = {}
    for node, attrs in hetero_graph.nodes(data=True):
        node_type = attrs.get('type')
        if node_type not in node_types:
            node_types[node_type] = 0
        node_types[node_type] += 1
    
    print("Node types in heterogeneous graph:")
    for node_type, count in node_types.items():
        print(f"  - {node_type}: {count}")
    
    # Count edge types
    edge_types = {}
    for _, _, attrs in hetero_graph.edges(data=True):
        edge_type = attrs.get('type')
        if edge_type not in edge_types:
            edge_types[edge_type] = 0
        edge_types[edge_type] += 1
    
    print("Edge types in heterogeneous graph:")
    for edge_type, count in edge_types.items():
        print(f"  - {edge_type}: {count}")
    
    print("\nDemo completed. Visualization files have been saved.")


if __name__ == "__main__":
    main()
