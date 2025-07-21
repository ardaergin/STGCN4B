from typing import Dict, Any
import numpy as np
import pandas as pd

import logging; logger = logging.getLogger(__name__)


class HomogGraphBuilderMixin:
    
    def add_weather_as_node_to_df(
        self, 
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a new DataFrame that appends one “outside” pseudo‑room whose
        features are the per‑bucket weather variables.

        Parameters
        ----------
        df : pd.DataFrame
            Room-level DataFrame, must contain at least ['room_uri_str', 'bucket_idx'].
        outside_URI_str : str
            The identifier inserted in room_uri_str for the outside node.

        Returns
        -------
        pd.DataFrame
            Original rows + one row per bucket for the outside node.
        """
        # Validation: Weather DataFrame
        if not hasattr(self, 'weather_df'):
            raise ValueError("weather_df not found. Run get_weather_data() first.")
        if self.weather_df.empty:
            raise ValueError("weather_df is empty. Cannot integrate the outside node.")
        outside_df = self.weather_df.copy()

        # Validation: Room-level DataFrame
        if not {"room_uri_str", "bucket_idx"}.issubset(df.columns):
            raise ValueError("room_level_df must contain 'room_uri_str' and 'bucket_idx'.")
        room_df = df.copy()
        
        # Prepare the 'outside' node for concat
        outside_URI_str = "outside"
        outside_df['room_uri_str'] = outside_URI_str
        
        # Ensure the order of columns matches room_df
        outside_df = outside_df.reindex(columns=room_df.columns.union(outside_df.columns))
        
        # Categorical check
        if pd.api.types.is_categorical_dtype(room_df['room_uri_str'].dtype):
            room_df['room_uri_str'] = room_df['room_uri_str'].cat.add_categories([outside_URI_str])
            outside_df['room_uri_str'] = outside_df['room_uri_str'].astype(room_df['room_uri_str'].dtype)
        
        # Concatenate the DataFrames
        room_level_df_with_weather_node = pd.concat([room_df, outside_df], ignore_index=True, sort=False)
        logger.info(f"DataFrame with 'outside' node created. New shape: {room_level_df_with_weather_node.shape}")
        return room_level_df_with_weather_node
    
    def update_adjacencies_for_weather_as_node(
        self, 
        adjacency_dict: Dict[str, Any],
        outside_URI_str: str = "outside"
    ) -> Dict[str, Any]:
        """
        Updates adjacency matrices and URI list by incorporating weather data as an "outside" room.
        This method modifies the adjacency_dict and returns the updated dictionary.

        Args:
            adjacency_dict: Dictionary containing adjacency information with keys:
                - "room_URIs_str": List of room URI strings
                - "adjacency_matrix": The NxN adjacency matrix
                - "outside_adjacency_vector": Vector of length N for outside connections
            outside_URI_str: URI string for the outside node (default: "outside")

        Returns:
            The updated adjacency_dict with modified:
            - "room_URIs_str": Updated to include the outside URI
            - "adjacency_matrix": Updated to (N+1)x(N+1) matrix
            - "masked_adjacency_matrices": Updated masked adjacency matrices
        """
        adj_dict = adjacency_dict.copy()

        # Extract required keys from adjacency dictionary
        old_uris = adj_dict["room_URIs_str"]
        old_adj = adj_dict["adjacency_matrix"]
        outside_adj_vector = adj_dict["outside_adjacency_vector"]
        
        N = old_adj.shape[0]
        
        # Validations
        if len(old_uris) != N:
            raise ValueError(f"Mismatch: len(room_URIs_str)={len(old_uris)} vs adjacency shape={old_adj.shape}")
        if outside_adj_vector.shape[0] != N:
            raise ValueError(f"outside_adjacency_vector length {outside_adj_vector.shape[0]} ≠ adjacency size {N}")
        if outside_URI_str in old_uris:
            raise ValueError(f"outside_uri {outside_URI_str!r} already exists in room_URIs_str.")
        
        # Build new (N+1)×(N+1) adjacency
        new_adj_matrix = np.zeros((N + 1, N + 1), dtype=old_adj.dtype)
        new_adj_matrix[:N, :N] = old_adj
        new_adj_matrix[N, :N] = outside_adj_vector.copy()  # outside → old rooms
        
        # Create new URI list by concatenating
        new_uri_list = old_uris + [outside_URI_str]
        
        logger.info(f"Created new adjacency matrix with shape {new_adj_matrix.shape}.")
        logger.info(f"Created new URI list with length {len(new_uri_list)}.")
        
        # Create new information propagation masks
        new_masked_adjs = self.create_masked_adjacency_matrices(new_adj_matrix, new_uri_list)
        
        logger.info(f"Computed and patched {len(new_masked_adjs)} new information propagation masks.")
        
        # Update the adjacency_dict in place
        adj_dict["room_URIs_str"] = new_uri_list
        adj_dict["n_nodes"] += 1
        adj_dict["adjacency_matrix"] = new_adj_matrix
        adj_dict["masked_adjacency_matrices"] = new_masked_adjs
        
        return adj_dict