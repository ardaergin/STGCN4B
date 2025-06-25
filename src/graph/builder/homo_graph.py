import os
import numpy as np
from rdflib import URIRef
from typing import Dict, Any
import torch
import pandas as pd

import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class HomogGraphBuilderMixin:
    """
    Assumes:
      - `self.room_feature_df` (room‐level features per time bucket) has already been built.
      - `self.room_to_room_adj_matrix` (combined horizontal + vertical adjacency) already exists.
      - `self.adj_matrix_room_uris` lists the room URIs in the same order as rows/columns of that matrix.
    """

    def expand_room_feature_df(self) -> None:
        """
        Helper function to take the DataFrame of measured rooms and expand
        it to include all rooms defined in the graph, filling missing ones with NaNs.
        This is required for the STGCN pipeline.
        """
        if not hasattr(self, "room_feature_df"):
            raise ValueError("Run build_room_feature_df() first.")

        logger.info("Expanding feature DataFrame to include all graph nodes for STGCN...")

        # Re-index to ensure every room × every bucket appears.
        # If a room had no measurements in a bucket, those columns will now be NaN.
        all_rooms   = list(self.office_graph.rooms.keys())      # every room URIRef, regardless of measurements
        all_buckets = list(range(len(self.time_buckets)))       # 0..T-1
        
        full_index = pd.MultiIndex.from_product(
            [all_rooms, all_buckets],
            names=["room_uri", "bucket_idx"]
        )
        
        # Re-index the DataFrame of measured rooms to the full index
        expanded_df = (
            self.room_feature_df
            .set_index(["room_uri", "bucket_idx"])  # make these two levels the index
            .reindex(full_index)                    # add missing (room, bucket) pairs
            .reset_index()                           # restore columns 'room_uri' & 'bucket_idx'
        )
        self.room_feature_df = expanded_df

        logger.info(f"Reindexed to full (room, bucket) grid: {len(all_rooms)} rooms x {len(all_buckets)} buckets = {len(expanded_df)} total rows.")

        return None
    
    def get_targets_and_mask_for_a_variable(self, stat: str = "mean"):
        """
        Pivots `room_feature_df` to produce two matrices aligned with `adj_matrix_room_uris`:
          1. Target values: Pivoted from the "{variable}_{stat}" column (e.g., "Temperature_mean").
          2. Mask: Pivoted from the "{variable}_has_measurement" column, indicating which
             (time, room) pairs have valid ground-truth data.

        Stores:
          - self.measurement_values: (T, R) numpy array of target values.
          - self.measurement_mask:   (T, R) numpy array of 1s (has data) and 0s (no data).
        """
        if not hasattr(self, "room_feature_df"):
            raise ValueError("room_feature_df not found. Run build_room_feature_df() first.")
        df = self.room_feature_df

        # Set measurement variable
        variable = self.measurement_variable

        # 1. Create the target values matrix
        value_column = f"{variable}_{stat}"
        value_pivot = (
            df.pivot(index="bucket_idx",
                     columns="room_uri",
                     values=value_column)
            .sort_index(axis=1, key=lambda cols: cols.map(self.adj_matrix_room_uris.index))
        )
        self.measurement_values = value_pivot.values
        logger.info(f"Generated measurement_values matrix of shape {self.measurement_values.shape}")


        # 2. Create the mask matrix from the 'has_measurement' column
        mask_column = f"{variable}_has_measurement"
        if mask_column not in df.columns:
            raise ValueError(f"Mask column '{mask_column}' not found in room_feature_df. "
                             "Ensure `build_room_feature_df` creates it.")

        mask_pivot = (
            df.pivot(index="bucket_idx",
                     columns="room_uri",
                     values=mask_column)
            .sort_index(axis=1, key=lambda cols: cols.map(self.adj_matrix_room_uris.index))
            .fillna(0.0) # Rooms with no entries at all get a mask value of 0.
        )
        
        # Ensure the mask is strictly binary (0.0 or 1.0)
        self.measurement_mask = (mask_pivot.values > 0).astype(float)
        logger.info(f"Generated measurement_mask matrix of shape {self.measurement_mask.shape}")
        
        num_masked_points = np.sum(self.measurement_mask)
        logger.info(f"The mask identifies {num_masked_points:.0f} valid data points for loss calculation.")

        return None

    def incorporate_weather_as_an_outside_room(self) -> None:
        """
        1) Takes `weather_data`, which is a dict mapping bucket_idx → {feature_name: value}.
           Internally turns it into a DataFrame whose index is bucket_idx and whose columns are the weather features.
        2) Appends one new “outside” node (URIRef) to both:
             • self.room_to_room_adj_matrix  (adding a new row = self.combined_outside_adj, new col = zeros)
             • self.adj_matrix_room_uris     (appending outside_uri)
        3) Adds one row per bucket_idx to self.room_feature_df with room_uri=outside_uri:
             • All “original” room‐feature columns = 0.0
             • New weather‐columns = values from weather_data for that bucket.
           Also adds those new weather‐columns (initially 0.0) to every existing room‐row.
        4) Re‐runs calculate_information_propagation_masks(), patches each mask so that
           at step k, outside→any “active” room is allowed (but outside never receives).
           Then rebuilds self.masked_adjacencies = {k: adjacency * patched_mask_k}.

        Pre‐conditions (must have been done already):
          • combine_outside_adjacencies() has been called, so that
            – self.adj_matrix_room_uris  is a list of length N,
            – self.room_to_room_adj_matrix is an N×N ndarray,
            – self.combined_outside_adj   is a length‐N 1D ndarray aligned with that list.
          • build_room_feature_df() has already been called, so that
            – self.room_feature_df exists with columns ['room_uri','bucket_idx', <orig_features…>].
            - self.get_weather_data() has been called.

        After calling this, we can simply do:
            builder.generate_feature_matrices()
        and the new outside‐node (index N) will appear throughout.

        Side‐effects:
          • Mutates self.adj_matrix_room_uris  (appends outside_uri at end).
          • Mutates self.room_to_room_adj_matrix (to size (N+1)×(N+1)).
          • Mutates self.room_feature_df        (adds new weather columns & new rows for outside).
          • Rebuilds self.masked_adjacencies accordingly.
        """
        # Creating an 'outside' URI
        outside_uri = URIRef(str("outside"))

        #
        # ─── A) Convert weather_data (dict) → DataFrame if needed ───────────────────
        #
        # weather_data is dict[bucket_idx → {feature: value}].  Turn into:
        #    weather_df: index=bucket_idx (ints 0..T−1), columns = weather_feature_names.
        if not hasattr(self, 'weather_data_dict'):
            raise ValueError("weather_data_dict not found. Run get_weather_data() first.")
        if len(self.weather_data_dict) == 0:
            raise ValueError("weather_data is empty. Cannot integrate outside node without any weather features.")

        # Build a DataFrame directly from the dict
        weather_df = pd.DataFrame.from_dict(self.weather_data_dict, orient="index")
        # Now weather_df.index = bucket_idx values, weather_df.columns = e.g. ['temperature_2m', 'relative_humidity_2m', …, plus  
        # any engineered columns like wind_direction_10m_sin, or wc_…].  
        # We assume the index runs 0..T−1 in order.

        # 0.1) Ensure the index is integer and sorted
        try:
            weather_df.index = weather_df.index.astype(int)
        except Exception:
            raise ValueError("weather_data keys (bucket_idx) must be integers.")
        weather_df = weather_df.sort_index()

        #
        # ─── B) UPDATE ADJACENCY ────────────────────────────────────────────────────
        #
        old_adj = self.room_to_room_adj_matrix
        old_uris = self.adj_matrix_room_uris
        N = old_adj.shape[0]
        if len(old_uris) != N:
            raise ValueError(f"Mismatch: len(self.adj_matrix_room_uris)={len(old_uris)} vs adjacency shape={old_adj.shape}")

        outside_weights = self.combined_outside_adj
        if outside_weights.shape[0] != N:
            raise ValueError(
                f"self.combined_outside_adj length {outside_weights.shape[0]} ≠ adjacency size {N}"
            )

        # Build new (N+1)×(N+1) adjacency
        new_adj = np.zeros((N + 1, N + 1), dtype=old_adj.dtype)
        new_adj[:N, :N] = old_adj
        # New column (old→outside) stays zeros
        new_adj[N, :N] = outside_weights  # outside→old rooms
        # new_adj[N, N] remains zero

        # Append outside_uri to URI list
        if outside_uri in old_uris:
            raise ValueError(f"outside_uri {outside_uri!r} already exists in adj_matrix_room_uris.")
        self.adj_matrix_room_uris.append(outside_uri)

        # Overwrite adjacency
        self.room_to_room_adj_matrix = new_adj

        logger.info(f"Updated room_to_room_adj_matrix to shape {self.room_to_room_adj_matrix.shape}.")
        logger.info(f"Appended {outside_uri} to adj_matrix_room_uris (new length: {len(self.adj_matrix_room_uris)}).")

        #
        # ─── C) UPDATE room_feature_df ─────────────────────────────────────────────
        #
        df = self.room_feature_df.copy()

        # (1) Must have columns 'room_uri' and 'bucket_idx'
        if 'room_uri' not in df.columns or 'bucket_idx' not in df.columns:
            raise ValueError("self.room_feature_df must have columns 'room_uri' and 'bucket_idx' before integration.")

        # (2) Original feature columns (everything except 'room_uri'/'bucket_idx')
        orig_cols = [c for c in df.columns if c not in ('room_uri', 'bucket_idx')]
        if not orig_cols:
            raise ValueError("No original feature‐columns found in room_feature_df.")

        # (3) Get sorted bucket indices from df
        buckets = sorted(df['bucket_idx'].unique())
        # Ensure weather_df.index matches exactly
        missing = set(buckets) - set(weather_df.index)
        if missing:
            raise ValueError(
                f"room_feature_df uses bucket_idx={buckets} but weather_df.index={list(weather_df.index)}. "
                f"Missing buckets in weather data: {missing}"
            )

        # (4) Identify weather columns to add
        weather_cols = list(weather_df.columns)
        if not weather_cols:
            raise ValueError("No columns found in weather_df; cannot add empty weather features.")

        # (5) Add each weather_col to every existing row, initialized to np.nan
        for wcol in weather_cols:
            if wcol in df.columns:
                raise ValueError(f"Column {wcol!r} already exists in room_feature_df.")
            df[wcol] = np.nan

        # (6) Build new rows for outside_uri, one per bucket
        new_rows = []
        for b in buckets:
            row_dict = {
                'room_uri':   outside_uri,
                'bucket_idx': b
            }
            # set all original features = np.nan
            for fc in orig_cols:
                row_dict[fc] = np.nan
            # set weather features from weather_df
            for wcol in weather_cols:
                row_dict[wcol] = float(weather_df.at[b, wcol])
            new_rows.append(row_dict)

        new_df = pd.DataFrame(new_rows)

        # (7) Concatenate: old df (with zeros in new weather columns) + new_df
        df = pd.concat([df, new_df], ignore_index=True)

        # Overwrite
        self.room_feature_df = df

        logger.info(f"Updated room_feature_df: new shape {self.room_feature_df.shape}, {len(weather_cols)} weather columns added.")
        logger.info(f"Added {len(new_rows)} rows for outside_uri to room_feature_df.")

        #
        # ─── D) RE‐COMPUTE & PATCH INFORMATION‐PROPAGATION MASKS ───────────────────
        #
        # Now that the adjacency has an extra node (index=N), we must recalc masks,
        # then patch so that at each step k, outside(N)→any “active” room is allowed.

        # Recompute masks (each masks[k] is (N+1)×(N+1))
        masks = self.calculate_information_propagation_masks()

        patched_masks = {}
        total_rooms = N + 1  # new total

        for step_k, mask_k in masks.items():
            # Identify all i for which mask_k[i, :].sum() > 0  ⇒  “i is active at step k”
            active = np.where(mask_k.sum(axis=1) > 0)[0]

            patched = mask_k.copy()
            # Allow outside(N) to send to every active i
            for i in active:
                patched[N, i] = 1
            # (Do NOT allow anything i→N; leave patched[i, N] = 0 for all i)

            patched_masks[step_k] = patched

        # Finally rebuild self.masked_adjacencies
        adjacency = self.room_to_room_adj_matrix
        self.masked_adjacencies = {
            k: adjacency * patched_masks[k]
            for k in patched_masks
        }
        logger.info(f"Recomputed and patched {len(self.masked_adjacencies)} information propagation masks.")

        return None

    def build_feature_array(self) -> None:
        """
        Converts the `room_feature_df` into a single, comprehensive 3D NumPy array
        of shape (T, R, F_static + F_temporal) containing all feature data.

        This method should be called once during the initial data preparation phase.
        It stores the final array in `self.feature_array`.

        To turn these into feature matrices later, run:
        >>> self.feature_matrices = {t: processed_array[t] for t in range(T)}
        """
        if not hasattr(self, "room_feature_df"):
            raise ValueError("room_feature_df not found. Run build_room_feature_df() first.")
        
        df = self.room_feature_df
        
        # --- Dimensions ---
        T = len(self.time_buckets)
        R = len(self.adj_matrix_room_uris)

        # --- Temporal Features ---
        temporal_cols = sorted([c for c in df.columns if c not in {"room_uri", "bucket_idx"}])
        F_temporal = len(temporal_cols)
        
        # --- Static Features ---
        static_feature_names = getattr(self, "static_room_attributes", [])
        F_static = len(static_feature_names)

        # Total features
        self.feature_names = static_feature_names + temporal_cols
        F = F_static + F_temporal
        self.n_features = F
        self.static_feature_count = F_static
        self.temporal_feature_count = F_temporal

        logger.info(f"Preparing to build feature array with dimensions T={T}, R={R}, F={F}")
        
        # --- Build Static Feature Matrix ---
        static_mat = np.zeros((R, F_static), dtype=float)
        if F_static > 0:
            for i, room_uri in enumerate(self.adj_matrix_room_uris):
                room_obj = self.office_graph.rooms.get(room_uri)
                if room_obj:
                    for j, feat_string in enumerate(static_feature_names):
                        val = self._get_nested_attr(room_obj, feat_string, default=0.0)
                        static_mat[i, j] = float(val) if val is not None and not np.isnan(val) else 0.0
        
        # Tile static features across the time dimension
        static_array = np.tile(static_mat, (T, 1, 1)) # Shape -> (T, R, F_static)

        # --- Build Temporal Feature Array ---
        # Pivot the DataFrame to get a (T, R, F_temporal) structure directly
        # Set index for easy slicing
        df = df.set_index(['bucket_idx', 'room_uri'])
        
        # Create a complete index to ensure all (T, R) pairs are present
        all_rooms = self.adj_matrix_room_uris
        idx = pd.MultiIndex.from_product([range(T), all_rooms], names=['bucket_idx', 'room_uri'])
        
        # Reindex and sort to ensure canonical order
        temporal_df = df[temporal_cols].reindex(idx).sort_index()
        
        # Reshape into a 3D NumPy array
        temporal_array = temporal_df.values.reshape(T, R, F_temporal)

        # --- Concatenate Static and Temporal Arrays ---
        feature_array = np.concatenate([static_array, temporal_array], axis=2)
        
        self.feature_array = feature_array
        logger.info(f"Successfully built feature_array of shape {self.feature_array.shape}")
        
        return None
    
    def prepare_and_save_numpy_input(self, output_path: str):
        """
        Gathers all pre-computed NumPy arrays and metadata and saves them to a
        single file. This is the final step of the data building process.

        Args:
            output_path (str): The full path to save the output file (e.g., 'data/processed/numpy_input.pt').
        """
        if not hasattr(self, 'feature_array'):
            raise ValueError("feature_array not found. Run build_feature_array() first.")
        
        logger.info(f"Packaging NumPy arrays and metadata for saving to {output_path}...")
        
        numpy_input = {
            # Data indices in block format
            "blocks": self.blocks,
            "block_size": self.block_size,

            # Main data (X)
            "feature_array": self.feature_array,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "static_feature_count": self.static_feature_count,
            "temporal_feature_count": self.temporal_feature_count,

            # Graph structure
            "room_uris": self.adj_matrix_room_uris,
            "n_nodes": len(self.adj_matrix_room_uris),

            # Adjacency
            "adjacency_matrix": self.room_to_room_adj_matrix,
            "masked_adjacencies": self.masked_adjacencies,
        }

        # Add task-specific targets and masks
        if self.build_mode == "workhour_classification":
            numpy_input["workhour_labels"] = self.workhour_labels
        elif self.build_mode == "consumption_forecast":
            numpy_input["consumption_values"] = self.consumption_values
        elif self.build_mode == "measurement_forecast":
            target_name = f"{self.measurement_variable}_values"
            numpy_input[target_name] = self.measurement_values
            numpy_input["target_mask"] = self.measurement_mask
        else:
            raise ValueError(f"Unknown build_mode: {self.build_mode}")
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the dictionary using torch.save
        torch.save(numpy_input, output_path)
        logger.info("Successfully saved all NumPy inputs.")
        return None