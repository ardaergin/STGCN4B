import logging
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from rdflib import URIRef
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import networkx as nx

logger = logging.getLogger(__name__)

class SpatialVisualizerMixin:

    ##############################
    # Floor plan Plotting
    ##############################

    def plot_floor_plan(self, 
                    floor_number,
                    normalization='min_max',
                    show_room_ids=True,
                    figsize=(12, 10), 
                    colormap='turbo'):
        """
        Plot the floor plan for a specific floor with rooms colored according to their normalized areas.
        
        Args:
            floor_number (int): Which floor to plot
            normalization (str): Which normalization to use - 'min_max' or 'proportion'
            show_room_ids (bool): Whether to show room IDs in the plot
            figsize (tuple): Figure size
            colormap (str): Matplotlib colormap name for coloring rooms
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        
        # Check if we have the required data
        if not hasattr(self, 'polygons') or not self.polygons:
            logger.error("No polygons available for plotting. Call initialize_room_polygons first.")
            return None
        
        # Check if floor exists
        if floor_number not in self.polygons:
            available_floors = list(self.polygons.keys())
            logger.error(f"Floor {floor_number} not found. Available floors: {available_floors}")
            return None
        
        # Get rooms for this floor
        rooms_to_plot = self.polygons[floor_number]
        
        if not rooms_to_plot:
            logger.error(f"No rooms found on floor {floor_number}")
            return None
            
        # Check if normalized areas are calculated
        if not hasattr(self, 'norm_areas_minmax') or not self.norm_areas_minmax:
            logger.info("Calculating area normalizations...")
            self.normalize_room_areas()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the color map
        cmap = plt.get_cmap(colormap)
        
        # Determine which normalization to use
        if normalization == 'min_max':
            normalized_areas = self.norm_areas_minmax
        elif normalization == 'proportion':
            normalized_areas = self.norm_areas_prop
        else:
            logger.warning(f"Unknown normalization type: {normalization}. Using 'min_max'.")
            normalized_areas = self.norm_areas_minmax
        
        # Plot each room on this floor
        for room_uri, polygon in rooms_to_plot.items():
            # Get room color based on normalized area (default to 0.5 if missing)
            norm_value = normalized_areas.get(floor_number, {}).get(room_uri, 0.5)
            
            # Get room ID for display
            if hasattr(self.office_graph, 'rooms') and room_uri in self.office_graph.rooms:
                room = self.office_graph.rooms[room_uri]
                display_id = room.room_number or str(room_uri).split('/')[-1]
            else:
                # Extract just the room number part if it's prefixed
                display_id = str(room_uri).split('/')[-1]
                if 'roomname_' in display_id:
                    display_id = display_id.split('roomname_')[-1]
            
            # Plot the polygon
            color = cmap(norm_value)
            x, y = polygon.exterior.xy
            ax.fill(x, y, alpha=0.7, fc=color, ec='black')
            
            # Show room ID if requested
            if show_room_ids:
                centroid = polygon.centroid
                ax.text(centroid.x, centroid.y, display_id,
                    ha='center', va='center', fontsize=8, 
                    color='black', fontweight='bold')
        
        # Set aspect equal to preserve shape
        ax.set_aspect('equal')
        
        # Get axis limits for this floor
        min_x = min(polygon.bounds[0] for polygon in rooms_to_plot.values())
        max_x = max(polygon.bounds[2] for polygon in rooms_to_plot.values())
        min_y = min(polygon.bounds[1] for polygon in rooms_to_plot.values())
        max_y = max(polygon.bounds[3] for polygon in rooms_to_plot.values())
        
        # Add some padding
        padding = 0.05 * max(max_x - min_x, max_y - min_y)
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)
        
        # Set title
        plt.title(f"Floor {floor_number} - {normalization.replace('_', '-')} Normalized Areas")
        
        # Add color bar to show area scale
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=Normalize(vmin=0, vmax=1)
        )
        sm.set_array([])  
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(f'{normalization.capitalize()} Normalized Room Area')
        
        # Add axes labels based on polygon type
        coord_type = "Geographic" if self.polygon_type == "geo" else "Document"
        ax.set_xlabel(f"{coord_type} X Coordinate")
        ax.set_ylabel(f"{coord_type} Y Coordinate")
        
        plt.tight_layout()
        return fig

    #############################
    # Horizontal Adjacency
    #############################

    def plot_horizontal_adjacency_matrix(self, figsize=(10, 8), title=None, show_room_ids=True, cmap='Blues'):
        """
        Plot the room-to-room adjacency matrix as a heatmap.
        
        Args:
            figsize (tuple): Figure size
            title (str, optional): Plot title. If None, a default title is used
            show_room_ids (bool): Whether to show room IDs on axes
            cmap (str): Matplotlib colormap name
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        
        if not hasattr(self, 'room_to_room_adj_matrix') or self.room_to_room_adj_matrix is None:
            raise ValueError("No room-to-room adjacency matrix found. Make sure to call build_horizontal_adjacency().")
        
        if not hasattr(self, 'adj_matrix_room_uris') or self.adj_matrix_room_uris is None:
            raise ValueError("Room URIs in adjacency matrix order not found. Make sure to call build_horizontal_adjacency().")
        
        fig = plt.figure(figsize=figsize)
        plt.imshow(self.room_to_room_adj_matrix, cmap=cmap)
        
        # Add colorbar with label based on adjacency type
        if hasattr(self, 'adjacency_type') and self.adjacency_type == 'weighted':
            plt.colorbar(label='Proportion of shared boundary')
        else:
            plt.colorbar(label='Connection strength')
        
        # Use pre-stored room names for labels
        if show_room_ids and len(self.adj_matrix_room_uris) <= 50:  # Only show labels if not too many rooms
            labels = [self.room_names.get(uri, str(uri)) for uri in self.adj_matrix_room_uris]
            plt.xticks(range(len(self.adj_matrix_room_uris)), labels, rotation=90)
            plt.yticks(range(len(self.adj_matrix_room_uris)), labels)
        elif not show_room_ids:
            plt.xticks([])
            plt.yticks([])
        else:
            # Too many rooms, just show indices
            plt.xticks(range(0, len(self.adj_matrix_room_uris), 5))
            plt.yticks(range(0, len(self.adj_matrix_room_uris), 5))
        
        # Set title
        if title is None:
            if hasattr(self, 'adjacency_type'):
                title = f"Room Adjacency Matrix ({self.adjacency_type})"
            else:
                title = "Room Adjacency Matrix"
        plt.title(title)
        
        # Add axis labels
        plt.xlabel("Room")
        plt.ylabel("Room")
        
        plt.tight_layout()

    #############################
    # Information propagation
    #############################

    def create_single_floor_propagation_visualization(self, output_file='output/builder/propagation_visualization.html'):
        """
        Create an interactive Plotly visualization of information propagation
        and export it to a standalone HTML file.
        
        Device rooms are colored green at Step 0, and subsequent information propagation
        is shown in shades of blue.
        
        Args:
            output_file: Path to save the HTML output file
            
        Returns:
            Plotly figure object
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np
        
        if not hasattr(self, 'masked_adjacencies') or not self.masked_adjacencies:
            raise ValueError("Masked adjacency matrices not found. Run build_masked_adjacencies first.")
            
        if not hasattr(self, 'room_polygons') or not self.room_polygons:
            raise ValueError("Room polygons not found. Run initialize_room_polygons first.")
        
        # Define the number of steps
        n_steps = len(self.masked_adjacencies)
        
        # Extract information about which rooms can pass info at each step
        room_info_by_step = {}
        for step in range(n_steps):
            mask = self.masked_adjacencies[step]
            can_pass_info = mask.sum(axis=1) > 0  # Rooms that can pass info
            room_info_by_step[step] = can_pass_info
        
        # Define colors
        device_color = '#2ca02c'  # Forest green for device rooms
        inactive_color = '#f0f0f0'  # Light gray for inactive rooms
        # Shades of blue for rooms activated in steps 1-N
        propagation_colors = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
        
        # Create a plotly figure with steps as frames
        fig = go.Figure()
        
        # Add frames for each step
        frames = []
        for step in range(n_steps):
            frame_data = []
            
            # For each room polygon
            for i, room_uri in enumerate(self.adj_matrix_room_uris):
                if room_uri in self.room_polygons:
                    polygon = self.room_polygons[room_uri]
                    
                    # Extract polygon coordinates
                    x, y = polygon.exterior.xy
                    
                    # Check if this is a device room (active at step 0)
                    is_device_room = room_info_by_step[0][i]
                    
                    # Determine when this room gets activated
                    activation_step = n_steps  # Default: not activated
                    for s in range(n_steps):
                        if room_info_by_step[s][i]:
                            activation_step = s
                            break
                    
                    # Determine color based on activation status for this step
                    if activation_step <= step:
                        # Room is active at this step
                        if is_device_room:
                            # Device rooms always show in green
                            color = device_color
                        else:
                            # Non-device rooms show in blue based on when they were activated
                            blue_idx = min(activation_step - 1, len(propagation_colors) - 1)
                            color = propagation_colors[blue_idx]
                    else:
                        # Not yet activated
                        color = inactive_color
                    
                    # Create room label
                    room_id = self.room_names.get(room_uri, str(room_uri).split('/')[-1])
                    if is_device_room:
                        room_id = f"{room_id}*"  # Mark device rooms
                    
                    # Create a polygon for this room
                    room_trace = go.Scatter(
                        x=list(x) + [x[0]],  # Close the polygon
                        y=list(y) + [y[0]],
                        fill='toself',
                        fillcolor=color,
                        line=dict(color='black', width=1),
                        text=room_id,
                        hoverinfo='text',
                        showlegend=False
                    )
                    
                    frame_data.append(room_trace)
            
            # Create frame for this step
            frame = go.Frame(
                data=frame_data,
                name=f"Step {step}",
                layout=go.Layout(
                    title=f"Information Propagation - Step {step}: "
                        f"{np.sum(room_info_by_step[step])}/{len(self.adj_matrix_room_uris)} rooms can pass information"
                )
            )
            frames.append(frame)
        
        # Add the initial data (step 0)
        fig.add_traces(frames[0].data)
        fig.frames = frames
        
        # Set up slider control
        sliders = [{
            'steps': [
                {
                    'method': 'animate',
                    'args': [
                        [f"Step {i}"],
                        {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }
                    ],
                    'label': f"Step {i}"
                } for i in range(n_steps)
            ],
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 16},
                'prefix': 'Step: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 300, 'easing': 'cubic-in-out'},
            'pad': {'b': 10, 't': 50},
            'len': 0.9,
            'x': 0.1,
            'y': 0
        }]
        
        # Play and pause buttons
        updatemenus = [{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [
                        None,
                        {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300, 'easing': 'quadratic-in-out'}
                        }
                    ]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [
                        [None],
                        {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }
                    ]
                }
            ],
            'direction': 'left',
            'pad': {'r': 10, 't': 87},
            'type': 'buttons',
            'x': 0.1,
            'y': 0,
            'xanchor': 'right',
            'yanchor': 'top'
        }]
        
        # Update figure layout
        fig.update_layout(
            title="Information Propagation Through Building",
            autosize=True,
            width=900,
            height=700,
            margin=dict(l=50, r=50, t=100, b=100),
            sliders=sliders,
            updatemenus=updatemenus
        )
        
        # Set axes properties
        fig.update_layout(
            title="Information Propagation Through Building",
            autosize=True,
            width=900,
            height=700,
            margin=dict(l=50, r=50, t=100, b=100),
            sliders=sliders,
            updatemenus=updatemenus,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        )
        
        # Add legend
        legend_traces = [
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=device_color),
                name='Device Rooms',
                showlegend=True
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=propagation_colors[0]),
                name='First Propagation',
                showlegend=True
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=propagation_colors[1]),
                name='Second Propagation',
                showlegend=True
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=propagation_colors[2]),
                name='Third Propagation',
                showlegend=True
            ),
            go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=15, color=inactive_color),
                name='Inactive Rooms',
                showlegend=True
            )
        ]
        
        for trace in legend_traces:
            fig.add_trace(trace)
                
        # Save as HTML
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Interactive visualization saved to {output_file}")
        
        return fig
    
    def create_building_propagation_visualization(
        self,
        output_file: str = "output/builder/propagation_3D.html",
        thickness: float = None,
        marker_size: int = 4
    ):
        """
        Interactive 3D Plotly visualization with room‐shaped extrusions.
        
        Args:
            output_file: path for standalone HTML
            thickness: vertical thickness of each room (if None, auto = floor_sep/10)
            marker_size: size of centroid markers (optional)
        Returns:
            Plotly Figure
        """
        import numpy as np
        import plotly.graph_objects as go
        from shapely.geometry import Polygon
        
        # --- ensure adjacency masks exist ---
        if not hasattr(self, "masked_adjacencies"):
            self.build_masked_adjacencies()
        masks = self.masked_adjacencies
        n_steps = len(masks)
        
        # --- gather centroids, floor info and room labels ---
        centroids, raw_zs, room_ids = [], [], []
        for uri in self.adj_matrix_room_uris:
            poly: Polygon = self.polygons[self.room_to_floor[uri]][uri]
            x, y = poly.centroid.coords[0]
            centroids.append((x, y))
            raw_zs.append(self.room_to_floor[uri])
            room_ids.append(self.room_names.get(uri, str(uri)))
        
        xs = np.array([c[0] for c in centroids])
        ys = np.array([c[1] for c in centroids])
        raw_zs = np.array(raw_zs, dtype=float)
        
        # --- compute floor separation & thickness ---
        floors = np.unique(raw_zs)
        n_floors = len(floors)
        # footprint extent
        footprint = max(xs.max() - xs.min(), ys.max() - ys.min())
        floor_sep = footprint / max(n_floors, 1)
        if thickness is None:
            thickness = floor_sep / 5.0  # arbitrary small slab thickness
        
        # scale z so each floor is separated
        floor_min = floors.min()
        zs = (raw_zs - floor_min) * floor_sep
        
        # --- activation step per room ---
        activation = np.full(len(xs), n_steps, dtype=int)
        for step in range(n_steps):
            active = masks[step].sum(axis=1) > 0
            activation[active & (activation == n_steps)] = step
        
        # --- color mapping ---
        device_color = "#2ca02c"
        inactive_color = "#cccccc"
        propagation_colors = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
        def color_for(step):
            if step == 0:
                return device_color
            elif step >= n_steps:
                return inactive_color
            else:
                return propagation_colors[min(step-1, len(propagation_colors)-1)]
        
        # --- build mesh & edges for each room once ---
        room_meshes = []
        room_edge_traces = []
        for idx, uri in enumerate(self.adj_matrix_room_uris):
            poly: Polygon = self.polygons[self.room_to_floor[uri]][uri]
            coords = list(poly.exterior.coords)[:-1]  # drop closing point
            N = len(coords)
            x2d, y2d = zip(*coords)
            z0 = zs[idx]
            # lower ring
            x_low = np.array(x2d)
            y_low = np.array(y2d)
            z_low = np.full(N, z0)
            # upper ring
            x_up = x_low
            y_up = y_low
            z_up = np.full(N, z0 + thickness)
            # vertices
            xv = np.concatenate([x_low, x_up])
            yv = np.concatenate([y_low, y_up])
            zv = np.concatenate([z_low, z_up])
            # faces (triangulation)
            i, j, k = [], [], []
            # top face (fan)
            for t in range(1, N-1):
                i += [N,   N + t,   N + t + 1]
                j += [N + t, N + t + 1, N]
                k += [N + t + 1, N,   N + t]
            # bottom face
            for t in range(1, N-1):
                i += [0, t + 1, t]
                j += [t + 1, t, 0]
                k += [t, 0, t + 1]
            # side faces
            for t in range(N):
                nt = (t + 1) % N
                # lower t, lower nt, upper nt
                i += [t, nt, N + nt]
                j += [nt, N + nt, t]
                k += [N + nt, t, N + t]
                # lower t, upper nt, upper t
                i += [t, N + nt, N + t]
                j += [N + nt, N + t, t]
                k += [N + t, t, N + nt]
            room_meshes.append((xv, yv, zv, i, j, k))
            
            # edges (just top‐face boundary)
            edge_x = list(x_up) + [x_up[0]]
            edge_y = list(y_up) + [y_up[0]]
            edge_z = list(z_up) + [z_up[0]]
            room_edge_traces.append((edge_x, edge_y, edge_z))
        
        # --- build frames ---
        frames = []
        for step in range(n_steps):
            data = []
            for idx in range(len(xs)):
                col = color_for(activation[idx]) if activation[idx] <= step else inactive_color
                xv, yv, zv, i, j, k = room_meshes[idx]
                # mesh
                data.append(
                    go.Mesh3d(
                        x=xv, y=yv, z=zv,
                        i=i, j=j, k=k,
                        color=col, opacity=1.0,
                        flatshading=True, showscale=False,
                        hoverinfo="skip",  # edges will handle hovers
                        name=""
                    )
                )
                # edges
                ex, ey, ez = room_edge_traces[idx]
                data.append(
                    go.Scatter3d(
                        x=ex, y=ey, z=ez,
                        mode="lines",
                        line=dict(color="black", width=2),
                        hoverinfo="text",
                        text=[f"{room_ids[idx]} (step {activation[idx]})"],
                        showlegend=False
                    )
                )
            frames.append(go.Frame(data=data, name=f"Step {step}"))
        
        # --- initial figure & animation controls ---
        fig = go.Figure(data=frames[0].data, frames=frames)
        steps = [
            dict(
                method="animate",
                label=str(s),
                args=[[f"Step {s}"],
                      dict(mode="immediate",
                           frame=dict(duration=300, redraw=True),
                           transition=dict(duration=300))]
            ) for s in range(n_steps)
        ]
        sliders = [dict(active=0, pad={"t":50}, currentvalue={"prefix":"Step: "}, steps=steps)]
        updatemenus = [dict(type="buttons", showactive=False,
                            y=0, x=0.1, xanchor="right", yanchor="top",
                            pad={"t":60,"r":10},
                            buttons=[
                                dict(label="Play", method="animate",
                                     args=[None,{"frame":{"duration":500,"redraw":True},"fromcurrent":True}]),
                                dict(label="Pause", method="animate",
                                     args=[[None],{"frame":{"duration":0,"redraw":False},"mode":"immediate"}])
                            ])]
        
        # --- final layout ---
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False, title="Floors")
            ),
            margin=dict(l=0,r=0,b=0,t=50),
            sliders=sliders,
            updatemenus=updatemenus,
            title="3D Propagation with Room Extrusions"
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved 3D polygon propagation viz to {output_file}")
        return fig
    


    #############################
    # Outside Adjacency
    #############################

    
    def create_building_outside_adjacency_visualization(
        self,
        output_file: str = "output/builder/outside_adjacency_3D.html",
        thickness: float = None,
        marker_size: int = 4,
        colormap: str = "YlOrRd"
    ):
        """
        Interactive 3D Plotly visualization with room‐shaped extrusions,
        colored by each room’s outside‐adjacency weight.

        Args:
            output_file: path for standalone HTML export
            thickness: vertical thickness of each room extrusion (if None, auto = floor_sep/10)
            marker_size: size of centroid markers (optional)
            colormap: Plotly continuous colorscale name for mapping weights (0.0–1.0)
        Returns:
            Plotly Figure
        """
        import numpy as np
        import plotly.graph_objects as go
        from shapely.geometry import Polygon
        from matplotlib import cm

        # --- ensure outside adjacency is computed ---
        if not hasattr(self, "room_to_outside_adjacency") or self.room_to_outside_adjacency is None:
            raise ValueError("Outside adjacency not found. Call calculate_outside_adjacency() first.")
        weights = np.array(self.room_to_outside_adjacency, dtype=float)
        uris = self.adj_matrix_room_uris

        # --- gather centroids, floors and labels ---
        centroids = []
        floors_raw = []
        labels = []
        for uri in uris:
            floor = self.room_to_floor[uri]
            poly: Polygon = self.polygons[floor][uri]
            x, y = poly.centroid.x, poly.centroid.y
            centroids.append((x, y))
            floors_raw.append(floor)
            # label fallback
            labels.append(self.office_graph.rooms.get(uri, {}).room_number or str(uri).split("/")[-1])

        xs = np.array([c[0] for c in centroids])
        ys = np.array([c[1] for c in centroids])
        floors = np.array(floors_raw, dtype=float)

        # --- compute vertical separation & thickness ---
        unique_floors = np.unique(floors)
        n_floors = len(unique_floors)
        extent = max(xs.max() - xs.min(), ys.max() - ys.min())
        floor_sep = extent / max(n_floors, 1)
        if thickness is None:
            thickness = floor_sep / 10.0

        # shift floors so lowest starts at z=0
        z0s = (floors - unique_floors.min()) * floor_sep

        # --- colormap setup via matplotlib, sampling at weight values ---
        cmap_mpl = cm.get_cmap(colormap)
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            norm_w = (weights - w_min) / (w_max - w_min)
        else:
            norm_w = weights
        colors = [f"rgb{tuple(int(255*c) for c in cmap_mpl(w0)[:3])}"
                  for w0 in norm_w
                  ]

        # --- build meshes & edge traces ---
        meshes = []
        edges = []
        for idx, uri in enumerate(uris):
            poly: Polygon = self.polygons[self.room_to_floor[uri]][uri]
            coords = list(poly.exterior.coords)[:-1]
            N = len(coords)
            x2d, y2d = zip(*coords)

            # lower & upper rings
            x_low = np.array(x2d);   y_low = np.array(y2d);   z_low = np.full(N, z0s[idx])
            x_up  = x_low;           y_up  = y_low;           z_up  = np.full(N, z0s[idx] + thickness)

            # vertices and faces
            xv = np.concatenate([x_low, x_up])
            yv = np.concatenate([y_low, y_up])
            zv = np.concatenate([z_low, z_up])

            i, j, k = [], [], []
            # top face fan
            for t in range(1, N-1):
                i += [N,   N + t,   N + t + 1]
                j += [N + t, N + t + 1, N]
                k += [N + t + 1, N,   N + t]
            # bottom face fan
            for t in range(1, N-1):
                i += [0, t + 1, t]
                j += [t + 1, t, 0]
                k += [t, 0, t + 1]
            # side faces
            for t in range(N):
                nt = (t + 1) % N
                # quad split into two triangles
                i += [t, nt, N + nt];        j += [nt, N + nt, t];         k += [N + nt, t, N + t]
                i += [t, N + nt, N + t];      j += [N + nt, N + t, t];      k += [N + t, t, N + nt]

            meshes.append(
                go.Mesh3d(
                    x=xv, y=yv, z=zv,
                    i=i, j=j, k=k,
                    color=colors[idx],
                    opacity=1.0,
                    flatshading=True,
                    showscale=False,
                    # <-- add these two lines:
                    hovertemplate=f"{labels[idx]}<br>Outside‐adj: {weights[idx]:.2f}<extra></extra>",
                    # remove any hoverinfo="skip"
                )
            )

            # edge trace for hover
            edge_x = list(x_up) + [x_up[0]]
            edge_y = list(y_up) + [y_up[0]]
            edge_z = list(z_up) + [z_up[0]]
            edges.append(
                go.Scatter3d(
                    x=edge_x, y=edge_y, z=edge_z,
                    mode="lines",
                    line=dict(color="black", width=2),
                    hoverinfo="text",
                    text=[f"{labels[idx]}: {weights[idx]:.2f}"],
                    showlegend=False
                )
            )

        # --- assemble figure ---
        fig = go.Figure(data=meshes + edges)
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=True, title="Floor")
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            title="3D Outside‐Adjacency Visualization"
        )

        # add a colorbar manually
        # Map weights 0→1 into colorscale for legend
        fig.update_traces(
            selector=dict(type="mesh3d"),
            coloraxis="coloraxis"
        )
        fig.update_layout(
            coloraxis=dict(
                colorscale=colormap,
                cmin=0, cmax=1,
                colorbar=dict(title="Outside Adjacency")
            )
        )

        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved 3D outside‐adjacency viz to {output_file}")

        return fig


    #############################
    # Network Graph (Homogeneous)
    #############################

    def build_homogeneous_graph(self, static_room_attributes: List[str] = None) -> nx.DiGraph:
        """
        Build a directed graph (nx.DiGraph) whose nodes are rooms,
        edges are taken from self.room_to_room_adj_matrix (directed),
        and node attributes come from self.static_room_attributes.
        """
        # 1) Preconditions
        if not hasattr(self, "room_to_room_adj_matrix"):
            raise ValueError(
                "room_to_room_adj_matrix not found. "
                "Run build_combined_room_to_room_adjacency() first."
            )
        if not hasattr(self, "adj_matrix_room_uris"):
            raise ValueError(
                "adj_matrix_room_uris not found. It must list URIs in the same "
                "order as rows/cols of room_to_room_adj_matrix."
            )


        # 2) Optionally override static attributes
        if static_room_attributes is not None:
            self.static_room_attributes = static_room_attributes


        # 3) If normalized‐area attrs are requested but not yet computed, compute them
        if (
            ("norm_areas_minmax" in self.static_room_attributes or
            "norm_areas_prop" in self.static_room_attributes)
            and (
                not hasattr(self, "norm_areas_minmax") or
                not self.norm_areas_minmax or
                not hasattr(self, "norm_areas_prop") or
                not self.norm_areas_prop
            )
        ):
            logger.info("Computing normalized room areas for node attributes...")
            self.normalize_room_areas()


        # 4) Create a directed graph
        G = nx.DiGraph()


        # 5) Add each room URI as a node, copying requested static attributes
        for room_uri in self.adj_matrix_room_uris:
            room_obj = self.office_graph.rooms.get(room_uri)
            if room_obj is None:
                raise KeyError(f"Room {room_uri} not found in office_graph.rooms")


            node_attrs: Dict[str, Any] = {}
            node_attrs["devices"] = list(getattr(room_obj, "devices", []))


            for attr in self.static_room_attributes:
                if attr == "norm_areas_minmax":
                    for floor_num, room_map in getattr(self, "norm_areas_minmax", {}).items():
                        if room_uri in room_map:
                            node_attrs["norm_areas_minmax"] = room_map[room_uri]
                            break
                elif attr == "norm_areas_prop":
                    for floor_num, room_map in getattr(self, "norm_areas_prop", {}).items():
                        if room_uri in room_map:
                            node_attrs["norm_areas_prop"] = room_map[room_uri]
                            break
                elif "." in attr:
                    container, key = attr.split(".", 1)
                    if hasattr(room_obj, container):
                        container_obj = getattr(room_obj, container)
                        if isinstance(container_obj, dict) and key in container_obj:
                            node_attrs[f"{container}.{key}"] = container_obj[key]
                elif hasattr(room_obj, attr):
                    node_attrs[attr] = getattr(room_obj, attr)


            G.add_node(room_uri, **node_attrs)

        # 6) Add directed edges for every non‐zero entry in the adjacency matrix
        adj = self.room_to_room_adj_matrix
        uris = self.adj_matrix_room_uris
        n = len(uris)
        for i in range(n):
            u = uris[i]
            for j in range(n):
                if i == j:
                    continue
                v = uris[j]
                weight = float(adj[i, j])
                if weight != 0.0:
                    G.add_edge(u, v, weight=weight)

        self.homogeneous_graph = G
        logger.info(
            f"Built homogeneous DiGraph with {G.number_of_nodes()} nodes "
            f"and {G.number_of_edges()} directed edges"
        )
        return G

    def plot_network_graph(self,
                        graph=None,
                        figsize=(12, 10),
                        layout='spring',
                        node_size_based_on='area',
                        node_size_factor=1000,
                        node_color='lightblue',
                        device_node_color='salmon',
                        edge_width=1.0,
                        show_room_ids=True):
        """
        Plot the room adjacency as a network graph.
        
        Args:
            graph (nx.Graph, optional): The networkx graph to plot. If None, uses 
                                    the graph from build_simple_homogeneous_graph()
            figsize (tuple): Figure size
            layout (str): Graph layout type ('spring', 'kamada_kawai', 'planar', 'spatial')
            node_size_based_on (str): 'area' or 'degree' to determine node sizes
            node_size_factor (float): Factor to control node sizes
            node_color (str): Color of nodes without devices
            device_node_color (str): Color of nodes with devices
            edge_width (float): Width of edges
            show_room_ids (bool): Whether to show room IDs in the plot
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """        
        # Use provided graph or build a new one if not available
        if self.base_graph is None:
            raise ValueError("Build homogeneous graph first.")
        else:
            graph = self.base_graph

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get positions for nodes based on chosen layout
        if layout == 'spring':
            pos = nx.spring_layout(graph)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif layout == 'planar':
            # Try planar layout, but fall back to spring if not possible
            try:
                pos = nx.planar_layout(graph)
            except nx.NetworkXException:
                logger.warning("Planar layout not possible. Falling back to spring layout.")
                pos = nx.spring_layout(graph)
        elif layout == 'spatial':
            # Use actual spatial positions from room polygons if available
            if hasattr(self, 'room_polygons') and self.room_polygons:
                pos = {}
                for node in graph.nodes():
                    if node in self.room_polygons:
                        centroid = self.room_polygons[node].centroid
                        pos[node] = (centroid.x, centroid.y)
                    else:
                        logger.warning(f"No polygon found for room {node}. Using centroid (0,0).")
                        pos[node] = (0, 0)
            else:
                logger.warning("Spatial layout requested but no room polygons available. Using spring layout.")
                pos = nx.spring_layout(graph)
        else:
            logger.warning(f"Unknown layout: {layout}. Using spring layout.")
            pos = nx.spring_layout(graph)
        
        # Get node sizes based on selected criteria
        node_sizes = []
        
        if node_size_based_on == 'area':
            # Use room areas if available
            if hasattr(self, 'areas') and self.areas:
                max_area = max(self.areas.values()) if self.areas else 1.0
                for node in graph.nodes():
                    # Get the area, defaulting to median if not found
                    if node in self.areas:
                        area = self.areas[node]
                        node_size = area * node_size_factor / max_area
                    else:
                        # Use median value if area not found
                        area = np.median(list(self.areas.values()))
                        node_size = area * node_size_factor / max_area
                    node_sizes.append(node_size)
            else:
                # If no areas available, use degree centrality instead
                logger.warning("Room areas not available. Using node degrees for sizing.")
                degrees = dict(graph.degree())
                max_degree = max(degrees.values()) if degrees else 1
                node_sizes = [degrees[node] * node_size_factor / max_degree for node in graph.nodes()]
        elif node_size_based_on == 'degree':
            # Size based on node degree (number of connections)
            degrees = dict(graph.degree())
            max_degree = max(degrees.values()) if degrees else 1
            node_sizes = [degrees[node] * node_size_factor / max_degree for node in graph.nodes()]
        else:
            # Default size if no valid option
            logger.warning(f"Unknown node_size_based_on value: {node_size_based_on}. Using constant size.")
            node_sizes = [node_size_factor * 0.3] * len(graph.nodes())
        
        # Separate nodes with and without devices
        nodes_with_devices = []
        nodes_without_devices = []
        
        for node in graph.nodes():
            if 'devices' in graph.nodes[node] and graph.nodes[node]['devices']:
                nodes_with_devices.append(node)
            else:
                nodes_without_devices.append(node)
        
        # Draw nodes without devices
        if nodes_without_devices:
            nx.draw_networkx_nodes(graph, pos, 
                            nodelist=nodes_without_devices,
                            node_size=[node_sizes[i] for i, node in enumerate(graph.nodes()) if node in nodes_without_devices],
                            node_color=node_color, 
                            alpha=0.8, 
                            ax=ax)
        
        # Draw nodes with devices
        if nodes_with_devices:
            nx.draw_networkx_nodes(graph, pos, 
                            nodelist=nodes_with_devices,
                            node_size=[node_sizes[i] for i, node in enumerate(graph.nodes()) if node in nodes_with_devices],
                            node_color=device_node_color, 
                            alpha=0.8, 
                            ax=ax)
        
        # Draw edges with weight consideration if available
        if nx.get_edge_attributes(graph, 'weight'):
            weights = [graph[u][v]['weight'] for u, v in graph.edges()]
            # Normalize weights for visualization
            if weights:
                max_weight = max(weights)
                min_weight = min(weights)
                if max_weight > min_weight:
                    norm_weights = [(w - min_weight) / (max_weight - min_weight) * 2 + 0.5 for w in weights]
                    nx.draw_networkx_edges(graph, pos, width=norm_weights, alpha=0.6, ax=ax)
                else:
                    nx.draw_networkx_edges(graph, pos, width=edge_width, alpha=0.5, ax=ax)
            else:
                nx.draw_networkx_edges(graph, pos, width=edge_width, alpha=0.5, ax=ax)
        else:
            nx.draw_networkx_edges(graph, pos, width=edge_width, alpha=0.5, ax=ax)
        
        # Draw labels if requested - now using the pre-stored room names
        if show_room_ids:
            labels = {node: self.room_names.get(node, str(node)) for node in graph.nodes()}
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)
        
        # Add title based on adjacency type
        title = "Room Adjacency Graph"
        if hasattr(self, 'adjacency_type'):
            title += f" ({self.adjacency_type})"
        plt.title(title)
        
        # Add legend for node colors
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=node_color, markersize=10, label='Rooms without devices'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=device_node_color, markersize=10, label='Rooms with devices')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Add legend for node sizes
        if node_size_based_on == 'area':
            legend_text = "Node size proportional to room area"
        elif node_size_based_on == 'degree':
            legend_text = "Node size proportional to number of connections"
        else:
            legend_text = "Uniform node size"
        plt.figtext(0.5, 0.01, legend_text, ha='center')
        
        plt.axis('off')
        plt.tight_layout()
        return fig
