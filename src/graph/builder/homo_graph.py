from typing import Dict, Tuple, List
import numpy as np
from rdflib import URIRef
import pandas as pd

import logging
logger = logging.getLogger(__name__)


class HomogGraphBuilderMixin:
    
    def incorporate_weather_as_an_outside_room(
        self, 
        room_level_df: pd.DataFrame
        ) -> Tuple[np.ndarray, List[URIRef], pd.DataFrame, Dict[int, np.ndarray]]:
        """
        Computes a new graph structure and feature DataFrame by incorporating weather data as an "outside" room.
        This method is immutable: it does not modify the instance's state (self).

        Returns:
            A tuple containing four new objects:
            1. new_adj_matrix (np.ndarray): The (N+1)x(N+1) adjacency matrix.
            2. new_uri_list (List[URIRef]): The list of N+1 room URIs, including the new 'outside' URI.
            3. new_masked_adjs (Dict[int, np.ndarray]): The dictionary of patched, masked adjacency matrices.
            4. new_room_level_df (pd.DataFrame): The DataFrame with weather columns and new rows for the outside node.
        """
        # Getting the weather data
        if not hasattr(self, 'weather_data_dict'):
            raise ValueError("weather_data_dict not found. Run get_weather_data() first.")
        if not self.weather_data_dict:
            raise ValueError("weather_data is empty. Cannot integrate outside node without any weather features.")
        weather_df = pd.DataFrame.from_dict(self.weather_data_dict, orient="index")
        try:
            weather_df.index = weather_df.index.astype(int)
        except Exception:
            raise ValueError("weather_data keys (bucket_idx) must be integers.")
        weather_df = weather_df.sort_index()
        
        # Outside URI
        outside_URI_str = "outside"
        
        # --- 1) Create New Adjacency Matrix and URI List ---
        old_adj = self.room_to_room_adj_matrix
        old_uris = self.adj_matrix_room_URIs_str
        N = old_adj.shape[0]
        
        # Validations (no changes here)
        if len(old_uris) != N:
            raise ValueError(f"Mismatch: len(self.adj_matrix_room_URIs_str)={len(old_uris)} vs adjacency shape={old_adj.shape}")
        if self.combined_outside_adj.shape[0] != N:
            raise ValueError(f"self.combined_outside_adj length {self.combined_outside_adj.shape[0]} ≠ adjacency size {N}")
        if outside_URI_str in old_uris:
            raise ValueError(f"outside_uri {outside_URI_str!r} already exists in adj_matrix_room_URIs_str.")
        
        # Build new (N+1)×(N+1) adjacency
        new_adj_matrix = np.zeros((N + 1, N + 1), dtype=old_adj.dtype)
        new_adj_matrix[:N, :N] = old_adj
        new_adj_matrix[N, :N] = self.combined_outside_adj  # outside → old rooms
        
        # Create new URI list by concatenating
        new_uri_list = old_uris + [outside_URI_str]
        
        logger.info(f"Created new adjacency matrix with shape {new_adj_matrix.shape}.")
        logger.info(f"Created new URI list with length {len(new_uri_list)}.")
        
        # --- 2) Create New Information Propagation Masks ---
        # Call the mask calculation function with the *new* adjacency matrix.
        new_masked_adjs = self.create_masked_adjacency_matrices(new_adj_matrix, new_uri_list)
        
        logger.info(f"Computed and patched {len(new_masked_adjs)} new information propagation masks.")
        
        # --- 3) Create New DataFrame ---
        df = room_level_df.copy()
        
        # Validations
        if 'room_URIRef' not in df.columns or 'bucket_idx' not in df.columns:
            raise ValueError("room_level_df must have columns 'room_URIRef' and 'bucket_idx'.")
        orig_cols = [c for c in df.columns if c not in ('room_URIRef', 'bucket_idx')]
        if not orig_cols:
            raise ValueError("No original feature-columns found in room_level_df.")
        
        # Convert URIRef column to string for simplicity and compatibility
        df['room_URIRef'] = df['room_URIRef'].astype(str)
        df['room_URIRef'] = df['room_URIRef'].astype('category')
        
        orig_cols = [c for c in df.columns if c not in ('room_URIRef', 'bucket_idx')]
        if not orig_cols:
            raise ValueError("No original feature-columns found in room_level_df.")
        
        # Create new DataFrame with weather features
        buckets = sorted(df['bucket_idx'].unique())
        weather_cols = list(weather_df.columns)
        
        # Add weather columns (initially NaN) to a temporary DataFrame for existing rooms
        temp_df = df.reindex(columns=df.columns.tolist() + weather_cols)
        
        # Build new rows for the 'outside' node
        new_rows = []
        for b in buckets:
            row_dict = {'room_URIRef': outside_URI_str, 'bucket_idx': b}
            # Original room features are NaN for the 'outside' node
            for fc in orig_cols:
                row_dict[fc] = np.nan
            # Weather features are sourced from weather_df
            for wcol in weather_cols:
                row_dict[wcol] = float(weather_df.at[b, wcol])
            new_rows.append(row_dict)
        
        outside_df = pd.DataFrame(new_rows)
        outside_df['room_URIRef'] = outside_df['room_URIRef'].astype(temp_df['room_URIRef'].dtype)

        # Concatenate old data (with new columns) and the new 'outside' data
        new_room_level_df = pd.concat([temp_df, outside_df], ignore_index=True)
                
        logger.info(f"Created new room_level_df: shape {new_room_level_df.shape}, {len(weather_cols)} weather columns added.")
        
        return new_adj_matrix, new_uri_list, new_masked_adjs, new_room_level_df
