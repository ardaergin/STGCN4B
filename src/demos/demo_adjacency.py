#!/usr/bin/env python3
"""
Demo script to test the adjacency matrices functionality, now with CLI arguments.
"""

import argparse
import pickle
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import sys
import os

# allow imports from project root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def plot_adjacency_matrix(adj_matrix, titles, figsize=(10, 8), title="Adjacency Matrix"):
    plt.figure(figsize=figsize)
    plt.imshow(adj_matrix, cmap='Blues')
    plt.colorbar(label='Connection')
    if len(titles) <= 20:
        labels = [str(t).split('/')[-1] for t in titles]
        plt.xticks(range(len(titles)), labels, rotation=90)
        plt.yticks(range(len(titles)), labels)
    plt.title(title)
    plt.tight_layout()
    return plt

def plot_network(adj_matrix, titles, figsize=(12, 10), title="Network Graph"):
    # use spring_layout on a weighted or unweighted graph
    G = nx.from_numpy_array(adj_matrix)
    mapping = {i: str(uri).split('/')[-1] for i, uri in enumerate(titles)}
    G = nx.relabel_nodes(G, mapping)
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='skyblue',
        node_size=500,
        edge_color='gray',
        width=1,
        alpha=0.8
    )
    plt.title(title)
    plt.tight_layout()
    return plt

def parse_args():
    p = argparse.ArgumentParser(
        prog="demo_adjacency.py",
        description="Visualize floor-plan adjacency (CSV or polygons)."
    )
    p.add_argument(
        "--graph-pkl", "-g",
        default="data/processed/officegraph.pkl",
        help="Path to your pickled OfficeGraph instance"
    )
    p.add_argument(
        "--using", "-u",
        choices=["csv", "polygons"],
        default="polygons",
        help="Source of adjacency: CSV file or Shapely polygons."
    )
    p.add_argument(
        "--kind", "-k",
        choices=["binary", "distance", "boundary"],
        default="binary",
        help="(only for polygons) Type of adjacency to compute."
    )
    p.add_argument(
        "--csv-path",
        default="data/floor_plan/floor_7.csv",
        help="Path to floor_plan CSV (if using=csv)."
    )
    p.add_argument(
        "--polygons-path",
        default="data/floor_plan/room_polygons.py",
        help="Path to your Python file defining `all_floors` polygons."
    )
    p.add_argument(
        "--output-dir", "-o",
        default="output/adjacency",
        help="Directory to save .png outputs"
    )
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) load your OfficeGraph, which has builder inside
    print(f"Loading OfficeGraph from {args.graph_pkl!r}…")
    with open(args.graph_pkl, "rb") as f:
        office_graph = pickle.load(f)
    builder = office_graph.builder

    # 2) initialize the floorplan based on --using
    if args.using == "csv":
        print(f"→ Initializing from CSV: {args.csv_path}")
        builder.initialize_floorplan_from_CSV(args.csv_path)
        builder.build_room_to_room_adjacency(using="csv")
    else:
        print(f"→ Initializing from Polygons: {args.polygons_path}")
        builder.initialize_floorplan_from_Polygons(args.polygons_path)
        builder.build_room_to_room_adjacency(using="polygons", kind=args.kind)

    mat = builder.room_to_room_adj_matrix
    uris = builder.room_uris

    # 3) report
    print(f"Adjacency matrix shape: {mat.shape}")
    print(f"Total connections (sum of weights): {mat.sum():.2f}")

    # 4) build suffix to avoid overwrites
    suffix = args.using
    if args.using == "polygons":
        suffix += f"_{args.kind}"

    # 5) plot & save heatmap
    fig1 = plot_adjacency_matrix(
        mat,
        uris,
        title=f"{args.using.title()} adjacency ({args.kind if args.using=='polygons' else 'binary'})"
    )
    out1 = Path(args.output_dir) / f"adjacency_matrix_{suffix}.png"
    fig1.savefig(out1)
    print(f"Saved matrix heatmap to {out1}")

    # 6) plot & save network graph
    fig2 = plot_network(
        mat,
        uris,
        title=f"{args.using.title()} adjacency network ({args.kind if args.using=='polygons' else 'binary'})"
    )
    out2 = Path(args.output_dir) / f"adjacency_network_{suffix}.png"
    fig2.savefig(out2)
    print(f"Saved network plot to {out2}")

    print("Demo completed.")

if __name__ == "__main__":
    main()
