import os
from typing import Dict
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


import logging; logger = logging.getLogger(__name__)


class SpatialVisualizerMixin:

    ##############################
    # Floor plan plotting
    ##############################

    def plot_floor_plan(
            self, 
            floor_number,
            normalization       = 'min_max',
            show_room_ids       = True,
            figsize             = (12, 10), 
            colormap            = 'turbo'
    ):
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
        if not hasattr(self, 'polygons'):
            raise ValueError("No polygons available for plotting. Call initialize_room_polygons first.")
        if floor_number not in self.polygons:
            available_floors = list(self.polygons.keys())
            raise ValueError(f"Floor {floor_number} not found. Available floors: {available_floors}")
        
        # Get rooms for this floor
        rooms_to_plot = self.polygons[floor_number]
        if not rooms_to_plot:
            raise ValueError(f"No rooms found on floor {floor_number}")
                    
        # Check if normalized areas are calculated
        if not hasattr(self, 'norm_areas_minmax'):
            raise ValueError("Normalized areas not calculated. Call calculate_normalized_areas first.")
        if normalization == 'min_max':
            normalized_areas = self.norm_areas_minmax
        elif normalization == 'proportion':
            normalized_areas = self.norm_areas_prop
        else:
            logger.warning(f"Unknown normalization type: {normalization}. Using 'min_max'.")
            normalized_areas = self.norm_areas_minmax
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get the color map
        cmap = plt.get_cmap(colormap)
        
        # Plot each room on this floor
        for room_uri_str, polygon in rooms_to_plot.items():
            # Get room color based on normalized area (default to 0.5 if missing)
            norm_value = normalized_areas.get(floor_number, {}).get(room_uri_str, 0.5)

            # Get room ID for display
            display_id = self.room_names.get(room_uri_str)
            
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
    # Information propagation
    #############################
    

    def create_single_floor_propagation_visualization(
            self, 
            floor_number:                   int,
            masked_adjacency_matrices:      Dict[int, np.ndarray],
            output_file:                    str = 'output/builder/propagation_visualization.html'
    ):
        """
        Create an interactive Plotly visualization of information propagation for a single floor.
        
        Args:
            floor_number (int): The floor to visualize.
            masked_adjacency_matrices (dict): Dictionary of {step: matrix} for the whole building.
            output_file (str): Path to save the HTML output file.
            
        Returns:
            go.Figure: The Plotly figure object.
        """
        if not hasattr(self, 'polygons'):
            raise ValueError("Room polygons not found. Run initialize_room_polygons first.")
        
        masks = masked_adjacency_matrices
        
        # The number of steps is the highest key + 1
        n_steps = max(masks.keys()) + 1 if masks else 0
        if n_steps == 0:
            logger.warning("No propagation steps to visualize.")
            return go.Figure()
        
        # Get rooms and their master indices for only the specified floor
        floor_rooms_uri = self.floor_to_rooms.get(floor_number, [])
        if not floor_rooms_uri:
            logger.warning(f"No rooms found for floor {floor_number}.")
            return go.Figure()
        
        # Pre-calculate activation steps for all rooms to speed up frame generation
        activation_step = {}
        for step in sorted(masks.keys()):
            can_pass_info = masks[step].sum(axis=0) > 0
            for master_idx, is_active in enumerate(can_pass_info):
                uri = self.room_URIs_str[master_idx]
                if is_active and uri not in activation_step:
                    activation_step[uri] = step
        
        device_rooms = {uri for uri, step in activation_step.items() if step == 0}
        
        # Define colors
        device_color = '#2ca02c'  # Forest green for device rooms
        inactive_color = '#f0f0f0'  # Light gray for inactive rooms
        # Shades of blue for rooms activated in steps 1-N
        propagation_colors = ['#c6dbef', '#6baed6', '#2171b5', '#08306b']
        
        # Create a plotly figure with steps as frames
        fig = go.Figure()
        frames = []
        for step in range(n_steps):
            frame_data = []
            rooms_active_this_frame = 0

            for room_uri in floor_rooms_uri:
                poly = self.polygons[floor_number][room_uri]
                room_act_step = activation_step.get(room_uri, n_steps)

                if room_act_step <= step:
                    rooms_active_this_frame += 1
                    if room_uri in device_rooms:
                        color = device_color
                    else:
                        color_idx = min(room_act_step - 1, len(propagation_colors) - 1)
                        color = propagation_colors[color_idx]
                else:
                    color = inactive_color
                
                room_id = self.room_names.get(room_uri, room_uri.split('/')[-1])
                hover_text = f"<b>{room_id}</b><br>Activates at Step: {room_act_step if room_act_step < n_steps else 'N/A'}"
                if room_uri in device_rooms:
                    hover_text += " (Device)"

                x, y = poly.exterior.xy
                frame_data.append(go.Scatter(
                    x=list(x), y=list(y), fill='toself',
                    fillcolor=color, line=dict(color='black', width=1.5),
                    text=hover_text, # <-- Use the detailed hover_text
                    hoverinfo='text',
                    showlegend=False
                ))

            frame = go.Frame(
                data=frame_data,
                name=f"Step {step}",
                layout=go.Layout(title_text=f"Floor {floor_number} Propagation - Step {step} ({rooms_active_this_frame}/{len(floor_rooms_uri)} active)")
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
            masked_adjacency_matrices:      Dict[int, np.ndarray],
            output_file:                    str = "output/builder/propagation_3D.html",
            thickness:                      float = None,
    ):
        """
        Interactive 3D Plotly visualization of propagation with room-shaped extrusions.

        Args:
            masked_adjacency_matrices (dict): A dictionary of {step: masked_adj_matrix}.
            output_file (str): Path for standalone HTML file.
            thickness (float): Vertical thickness of each room. If None, it's calculated automatically.
        
        Returns:
            go.Figure: The Plotly Figure object.
        """
        if not hasattr(self, 'polygons'):
            raise ValueError("Room polygons not found. Run initialize_room_polygons first.")
        
        masks = masked_adjacency_matrices
        
        # The number of steps is the highest key + 1
        n_steps = max(masks.keys()) + 1 if masks else 0
        if n_steps == 0:
            logger.warning("No propagation steps to visualize.")
            return go.Figure()
        
        # Gather centroids, floor info and room labels
        centroids, raw_zs, room_ids = [], [], []
        for room_uri_str in self.room_URIs_str:
            floor = self.room_to_floor[room_uri_str]
            poly: Polygon = self.polygons[floor][room_uri_str]
            x, y = poly.centroid.coords[0]
            centroids.append((x, y))
            raw_zs.append(floor)
            room_ids.append(self.room_names.get(room_uri_str))
        
        xs = np.array([c[0] for c in centroids])
        ys = np.array([c[1] for c in centroids])
        raw_zs = np.array(raw_zs, dtype=float)
        
        # Compute floor separation & thickness
        floors = np.unique(raw_zs)
        n_floors = len(floors)
        footprint = max(xs.max() - xs.min(), ys.max() - ys.min())
        floor_sep = footprint / max(n_floors - 1, 1) if n_floors > 1 else 5.0
        if thickness is None:
            thickness = floor_sep / 5.0  # Arbitrary small slab thickness
        
        # Scale z so each floor is separated
        floor_min = floors.min()
        zs = (raw_zs - floor_min) * floor_sep
        
        # Activation step per room
        activation = np.full(len(xs), n_steps, dtype=int)
        for step in sorted(masks.keys()):
            active = masks[step].sum(axis=0) > 0
            activation[active & (activation == n_steps)] = step
        
        # Color mapping
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
        
        # Build mesh & edges for each room once
        room_meshes = []
        room_edge_traces = []
        for idx, room_uri_str in enumerate(self.room_URIs_str):
            poly: Polygon = self.polygons[self.room_to_floor[room_uri_str]][room_uri_str]
            coords = list(poly.exterior.coords)[:-1]  # Drop closing point
            N = len(coords)
            x2d, y2d = zip(*coords)
            z0 = zs[idx]

            # Lower and upper rings
            x_low, y_low, z_low = np.array(x2d), np.array(y2d), np.full(N, z0)
            x_up, y_up, z_up    = x_low, y_low, np.full(N, z0 + thickness)
            
            # Vertices
            xv = np.concatenate([x_low, x_up])
            yv = np.concatenate([y_low, y_up])
            zv = np.concatenate([z_low, z_up])

            # Faces (triangulation)
            i, j, k = [], [], []

            # Top face (fan)
            for t in range(1, N-1):
                i += [N,   N + t,   N + t + 1]
                j += [N + t, N + t + 1, N]
                k += [N + t + 1, N,   N + t]
            # Bottom face
            for t in range(1, N-1):
                i += [0, t + 1, t]
                j += [t + 1, t, 0]
                k += [t, 0, t + 1]
            # Side faces
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
            
            # Edges (just top‐face boundary)
            edge_x = list(x_up) + [x_up[0]]
            edge_y = list(y_up) + [y_up[0]]
            edge_z = list(z_up) + [z_up[0]]
            room_edge_traces.append((edge_x, edge_y, edge_z))
        
        # Build frames
        frames = []
        for step in range(n_steps):
            data = []
            for idx in range(len(xs)):
                col = color_for(activation[idx]) if activation[idx] <= step else inactive_color
                xv, yv, zv, i, j, k = room_meshes[idx]
                # Mesh
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
                # Edges
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
        
        # Initial figure & animation controls
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
        
        # Final layout
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
            outside_adjacency:      np.ndarray,
            output_file:            str = "output/builder/outside_adjacency_3D.html",
            thickness:              float = None,
            colormap:               str = "YlOrRd"
    ):
        """
        Interactive 3D Plotly visualization colored by outside-adjacency weight.
        
        Args:
            outside_adjacency (np.ndarray): Vector of outside-adjacency weights for each room.
            output_file (str): Path for standalone HTML export.
            thickness (float): Vertical thickness of each room extrusion.
            colormap (str): Plotly continuous colorscale name.
        
        Returns:
            go.Figure: The Plotly Figure object.
        """
        weights = outside_adjacency
        uris = self.room_URIs_str
        
        # Gather centroids, floors and labels
        centroids, floors_raw, labels = [], [], []
        for room_uri_str in uris:
            floor = self.room_to_floor[room_uri_str]
            poly: Polygon = self.polygons[floor][room_uri_str]
            centroids.append(poly.centroid.coords[0])
            floors_raw.append(floor)
            labels.append(self.room_names.get(room_uri_str))

        xs = np.array([c[0] for c in centroids])
        ys = np.array([c[1] for c in centroids])
        floors = np.array(floors_raw, dtype=float)
        
        # Compute vertical separation & thickness
        unique_floors = np.unique(floors)
        n_floors = len(unique_floors)
        extent = max(xs.max() - xs.min(), ys.max() - ys.min())
        floor_sep = extent / max(n_floors - 1, 1) if n_floors > 1 else 5.0
        if thickness is None:
            thickness = floor_sep / 10.0
        
        z0s = (floors - unique_floors.min()) * floor_sep
        
        # Build meshes
        meshes = []
        for idx, room_uri_str in enumerate(uris):
            poly: Polygon = self.polygons[self.room_to_floor[room_uri_str]][room_uri_str]
            coords = list(poly.exterior.coords)[:-1]
            N = len(coords)
            x2d, y2d = zip(*coords)
            
            x_low, y_low, z_low = np.array(x2d), np.array(y2d), np.full(N, z0s[idx])
            x_up, y_up, z_up = x_low, y_low, np.full(N, z0s[idx] + thickness)
            
            xv = np.concatenate([x_low, x_up])
            yv = np.concatenate([y_low, y_up])
            zv = np.concatenate([z_low, z_up])
            
            i, j, k = [], [], []
            # Top and bottom faces
            for t in range(1, N - 1):
                i.extend([N, 0]); j.extend([N + t, t + 1]); k.extend([N + t + 1, t])
            # Side faces
            for t in range(N):
                nt = (t + 1) % N
                i.extend([t, t]); j.extend([nt, N + nt]); k.extend([N + nt, N + t])

            meshes.append(
                go.Mesh3d(
                    x=xv, y=yv, z=zv, i=i, j=j, k=k,
                    intensity=np.full(len(xv), weights[idx]),
                    colorscale=colormap, cmin=0.0, cmax=1.0,
                    opacity=1.0, flatshading=True, name="",
                    hovertemplate=f"<b>{labels[idx]}</b><br>Outside Adjacency: {weights[idx]:.3f}<extra></extra>"
                )
            )

        # Assemble figure
        fig = go.Figure(data=meshes)
        fig.update_layout(
            title="3D Outside-Adjacency Visualization",
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                zaxis=dict(title="Floor"), aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            coloraxis=dict(
                colorscale=colormap, cmin=0.0, cmax=1.0,
                colorbar=dict(title="Adjacency<br>Weight")
            )
        )
        
        if output_file:
            fig.write_html(output_file)
            logger.info(f"Saved 3D outside-adjacency visualization to {output_file}")
        return fig
    
    
    #############################
    # Measurement visualization
    #############################
    
    
    def create_daily_temperature_visualization(
            self,
            output_file:            str = "output/builder/temperature_3D.html",
            day_to_visualize:       str = None, # "YYYY-MM-DD", None for random
            thickness:              float = None,
            colorscale:             str = 'RdYlBu_r'
    ):
        """
        Creates an interactive 3D Plotly visualization showing the hourly temperature
        variation in each room over a single day.

        Args:
            output_file (str): Path to save the standalone HTML file.
            day_to_visualize (str, optional): The specific day to visualize in "YYYY-MM-DD" format.
                                            If None, a random day with data is chosen. Defaults to None.
            thickness (float, optional): Vertical thickness of each room's extrusion. If None, it's auto-calculated.
            colorscale (str, optional): The Plotly colorscale to use for temperature. Defaults to 'RdYlBu_r'.

        Returns:
            plotly.graph_objects.Figure: The generated Plotly figure.
        """
        if not hasattr(self, 'room_level_df'):
            raise ValueError("room_level_df not found. Run build_room_level_df() first.")
        
        bucket_to_time = {i: bucket[0] for i, bucket in enumerate(self.time_buckets)}
        df = self.room_level_df[['room_uri', 'bucket_idx', 'Temperature_mean']].copy()
        df = df.dropna(subset=['Temperature_mean'])
        
        df['datetime'] = df['bucket_idx'].map(bucket_to_time)
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        
        available_dates = df['date'].unique()
        if not available_dates.any():
            raise ValueError("No temperature data found in room_level_df.")

        if day_to_visualize:
            selected_date = pd.to_datetime(day_to_visualize).date()
            if selected_date not in available_dates:
                raise ValueError(f"Date {selected_date} not found in the data. Available dates start from {available_dates.min()}.")
        else:
            selected_date = np.random.choice(available_dates)
            logger.info(f"No day specified. Randomly selected: {selected_date}")

        day_df = df[df['date'] == selected_date]
        if day_df.empty:
            raise ValueError(f"No temperature data available for the selected date: {selected_date}")
            
        temp_pivot = day_df.pivot_table(
            index='room_uri', columns='hour', values='Temperature_mean'
        )
        
        cmin = temp_pivot.min().min()
        cmax = temp_pivot.max().max()
        logger.info(f"Visualizing temperatures for {selected_date}. Range: {cmin:.1f}°C to {cmax:.1f}°C")

        # Geometry and Layout Calculation
        centroids, raw_zs, room_ids, room_uris_ordered = [], [], [], []
        for room_uri_str in self.room_uris:
            room = self.office_graph.rooms.get(room_uri_str)
            if not room: continue

            floor_num = self.room_to_floor.get(room_uri_str)
            if floor_num is None or room_uri_str not in self.polygons.get(floor_num, {}):
                continue

            poly: Polygon = self.polygons[floor_num][room_uri_str]
            x, y = poly.centroid.coords[0]
            centroids.append((x, y))
            raw_zs.append(floor_num)
            room_ids.append(self.room_names.get(room_uri_str, str(room_uri_str)))
            room_uris_ordered.append(room_uri_str)
        
        xs = np.array([c[0] for c in centroids])
        ys = np.array([c[1] for c in centroids])
        raw_zs = np.array(raw_zs, dtype=float)
        
        floors = np.unique(raw_zs)
        n_floors = len(floors)
        footprint = max(xs.max() - xs.min(), ys.max() - ys.min())
        floor_sep = footprint / max(n_floors, 1) if n_floors > 1 else footprint
        if thickness is None:
            thickness = floor_sep / 5.0
        
        floor_min = floors.min()
        zs = (raw_zs - floor_min) * floor_sep
        
        room_meshes, room_edge_traces = [], []
        for idx, room_uri_str in enumerate(room_uris_ordered):
            poly: Polygon = self.polygons[self.room_to_floor[room_uri_str]][room_uri_str]
            coords = list(poly.exterior.coords)[:-1]
            N = len(coords)
            x2d, y2d = zip(*coords)
            z0 = zs[idx]
            
            xv = np.concatenate([np.array(x2d), np.array(x2d)])
            yv = np.concatenate([np.array(y2d), np.array(y2d)])
            zv = np.concatenate([np.full(N, z0), np.full(N, z0 + thickness)])
            
            i, j, k = [], [], []
            for t in range(1, N - 1): i.extend([N, N + t, N + t + 1]); j.extend([N + t, N + t + 1, N]); k.extend([N + t + 1, N, N + t])
            for t in range(1, N - 1): i.extend([0, t + 1, t]); j.extend([t + 1, t, 0]); k.extend([t, 0, t + 1])
            for t in range(N): nt = (t + 1) % N; i.extend([t, nt, N + nt]); j.extend([nt, N + nt, t]); k.extend([N + nt, t, N + t]); i.extend([t, N + nt, N + t]); j.extend([N + nt, N + t, t]); k.extend([N + t, t, N + nt])
            
            room_meshes.append((xv, yv, zv, i, j, k))
            
            edge_x = list(xv[N:]) + [xv[N]]
            edge_y = list(yv[N:]) + [yv[N]]
            edge_z = list(zv[N:]) + [zv[N]]
            room_edge_traces.append((edge_x, edge_y, edge_z))

        # Build Animation Frames
        frames = []
        for hour in range(24):
            data = []
            for idx, room_uri_str in enumerate(room_uris_ordered):
                if room_uri_str in temp_pivot.index and hour in temp_pivot.columns:
                    temp = temp_pivot.loc[room_uri_str, hour]
                else:
                    temp = np.nan
                
                if pd.isna(temp):
                    mesh_color = "#cccccc"
                    hover_text = f"<b>{room_ids[idx]}</b><br>Hour: {hour}<br>No data"
                else:
                    norm_temp = (temp - cmin) / (cmax - cmin) if (cmax - cmin) > 0 else 0.5
                    mesh_color = sample_colorscale(colorscale, norm_temp)[0]
                    hover_text = f"<b>{room_ids[idx]}</b><br>Hour: {hour}<br>Temp: {temp:.1f}°C"

                xv, yv, zv, i, j, k = room_meshes[idx]
                data.append(go.Mesh3d(x=xv, y=yv, z=zv, i=i, j=j, k=k, color=mesh_color, opacity=1.0, flatshading=True, showscale=False, hoverinfo="skip"))
                
                ex, ey, ez = room_edge_traces[idx]
                data.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="black", width=1.5), hoverinfo="text", text=hover_text, showlegend=False))
            
            frames.append(go.Frame(data=data, name=f"Hour {hour}"))

        # Assemble Figure
        fig = go.Figure(data=frames[0].data, frames=frames)

        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='markers',
            marker=dict(
                colorscale=colorscale, cmin=cmin, cmax=cmax,
                colorbar=dict(title='Temperature (°C)', thickness=20, len=0.75, y=0.5),
                showscale=True
            ),
            hoverinfo='none', showlegend=False
        ))
        
        sliders = [dict(
            active=0, pad={"t": 50}, currentvalue={"prefix": "Hour: "},
            steps=[dict(method="animate", label=str(h), args=[[f"Hour {h}"], dict(mode="immediate", frame=dict(duration=300, redraw=True))]) for h in range(24)]
        )]
        updatemenus = [dict(
            type="buttons", showactive=False, y=0, x=1.0, xanchor="right", yanchor="top", pad={"t": 20, "r": 20},
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
            ]
        )]
        
        fig.update_layout(
            title=f"Hourly Room Temperature Visualization ({selected_date})",
            scene=dict(
                xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            sliders=sliders,
            updatemenus=updatemenus
        )
        
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            fig.write_html(output_file)
            logger.info(f"Saved 3D temperature visualization to {output_file}")
            
        return fig