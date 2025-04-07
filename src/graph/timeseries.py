import os
import pickle
import torch
import numpy as np
import networkx as nx
import holidays
from rdflib import URIRef
from typing import Dict, List, Tuple, Set, Optional, Iterable, Any
from datetime import datetime, timedelta, date
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class TimeSeriesPreparation:
    """
    Class to prepare OfficeGraph data for time series modeling.
    """
    
    def __init__(self, office_graph, 
                 start_time: str = "2022-03-01 00:00:00",
                 end_time: str = "2023-01-30 00:00:00",
                 interval_hours: int = 1):
        """
        Initialize with an OfficeGraph instance.
        
        Args:
            office_graph: The OfficeGraph instance
            start_time: Start time for analysis in format "YYYY-MM-DD HH:MM:SS"
            end_time: End time for analysis in format "YYYY-MM-DD HH:MM:SS"
            interval_hours: Size of time buckets in hours
        """
        self.office_graph = office_graph
        self.start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        self.end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
        self.interval_hours = interval_hours
        
        # Property types to ignore completely
        self.ignored_property_types: Set[str] = {"DeviceStatus", "BatteryLevel"}
        
        # Property types to use (these should be available in the dataset)
        self.used_property_types: List[str] = ["Temperature", "CO2Level", "Contact", "Humidity"]
        
        # Create time buckets once at initialization
        self.time_buckets = self._create_time_buckets()
        
        # Get the room adjacency matrix
        self.adjacency_matrix, self.room_uris = self.office_graph.builder.build_room_adjacency()
        
        # Initialize Dutch holidays for the years covered by the dataset
        self.dutch_holidays = self._initialize_dutch_holidays()
        
        # Log initialization information
        logger.info(f"Initialized STGCNDataPreparation with {len(self.room_uris)} rooms")
        logger.info(f"Time range: {self.start_time} to {self.end_time} with {len(self.time_buckets)} buckets")
        logger.info(f"Loaded Dutch holidays for {self.start_time.year}-{self.end_time.year}")

    def _initialize_dutch_holidays(self):
        """
        Initialize Dutch holidays for the time range of the dataset.
        
        Returns:
            holidays.HolidayBase: Dutch holiday calendar object
        """
        # Create Dutch holiday calendar for the years in our dataset
        start_year = self.start_time.year
        end_year = self.end_time.year
        
        # Initialize Dutch holidays (NL) for the relevant years
        dutch_holidays = holidays.NL(years=range(start_year, end_year + 1))
        
        return dutch_holidays

    def _create_time_buckets(self) -> List[Tuple[datetime, datetime]]:
        """
        Create time buckets from start_time to end_time with interval_hours.
        
        Returns:
            List of (start_time, end_time) tuples for each bucket
        """
        current_time = self.start_time
        time_buckets = []
        
        while current_time < self.end_time:
            bucket_end = current_time + timedelta(hours=self.interval_hours)
            time_buckets.append((current_time, bucket_end))
            current_time = bucket_end
            
        return time_buckets

    def prepare_feature_matrix(self) -> Tuple[np.ndarray, List[URIRef], List[str]]:
        """
        Create a feature matrix for rooms with properties as features.
        
        Returns:
            tuple: (feature_matrix, room_uris, property_types)
                - feature_matrix: numpy array with shape (n_rooms, n_properties * 2)
                - room_uris: list of room URIs in the same order as rows in feature_matrix
                - property_types: list of property types in the same order as columns in feature_matrix
        """
        # Get all room URIs from the adjacency matrix
        room_uris = list(self.room_uris)
        
        # Filter to only use the specified property types
        property_types = self.used_property_types
        
        # Initialize the feature matrix with zeros
        n_rooms = len(room_uris)
        n_properties = len(property_types)
        feature_matrix = np.zeros((n_rooms, n_properties * 2), dtype=float)
        
        # Fill the feature matrix
        for i, room_uri in enumerate(room_uris):
            for j, prop_type in enumerate(property_types):
                # Check if the room has this property type
                has_property = False
                if room_uri in self.office_graph.room_to_property_type:
                    has_property = prop_type in self.office_graph.room_to_property_type[room_uri]
                
                # Set the binary flag in the second half of features
                feature_matrix[i, n_properties + j] = 1.0 if has_property else 0.0
        
        logger.info(f"Created feature matrix with shape {feature_matrix.shape}")
        return feature_matrix, room_uris, property_types

    def aggregate_temporal_features(self) -> Dict[URIRef, Dict[int, Dict[str, Dict[str, float]]]]:
        """
        Aggregate measurements into temporal buckets for each room and property type.
        
        Returns:
            dict: A nested dictionary mapping:
                room_uri -> time_bucket_index -> property_type -> {mean, sd, min, max}
        """
        # Initialize nested dictionary with defaultdict for easy access
        temporal_features = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"measurements": []})))
        
        # For each room with measurements
        for room_uri, prop_measurements in self.office_graph.room_to_property_measurements.items():
            for prop_type, measurements in prop_measurements.items():
                # Skip ignored property types
                if prop_type in self.ignored_property_types:
                    continue
                    
                # Only include the property types we're interested in
                if prop_type not in self.used_property_types:
                    continue
                    
                # Group measurements by time bucket
                for meas in measurements:
                    for i, (bucket_start, bucket_end) in enumerate(self.time_buckets):
                        # Check if measurement falls within this time bucket
                        if bucket_start <= meas.timestamp < bucket_end:
                            # Initialize if not already
                            if "measurements" not in temporal_features[room_uri][i][prop_type]:
                                temporal_features[room_uri][i][prop_type]["measurements"] = []
                            
                            # Add the measurement value
                            temporal_features[room_uri][i][prop_type]["measurements"].append(meas.value)
                            break
        
        # Calculate statistics for each bucket
        for room_uri in temporal_features:
            for time_idx in temporal_features[room_uri]:
                for prop_type in temporal_features[room_uri][time_idx]:
                    values = temporal_features[room_uri][time_idx][prop_type]["measurements"]
                    if values:
                        # Calculate statistics
                        temporal_features[room_uri][time_idx][prop_type]["mean"] = np.mean(values)
                        temporal_features[room_uri][time_idx][prop_type]["sd"] = np.std(values) if len(values) > 1 else 0
                        temporal_features[room_uri][time_idx][prop_type]["min"] = np.min(values)
                        temporal_features[room_uri][time_idx][prop_type]["max"] = np.max(values)
                    else:
                        # Set default values for rooms without this property measurement
                        temporal_features[room_uri][time_idx][prop_type]["mean"] = 0
                        temporal_features[room_uri][time_idx][prop_type]["sd"] = 0
                        temporal_features[room_uri][time_idx][prop_type]["min"] = 0
                        temporal_features[room_uri][time_idx][prop_type]["max"] = 0
                    
                    # Remove raw measurements to save memory
                    del temporal_features[room_uri][time_idx][prop_type]["measurements"]
        
        logger.info(f"Aggregated temporal features for {len(temporal_features)} rooms")
        return temporal_features

    def generate_time_feature_matrices(self, temporal_features) -> Dict[int, np.ndarray]:
        """
        Generate feature matrices for each time bucket.
        
        Args:
            temporal_features: The temporal features dictionary from aggregate_temporal_features
        
        Returns:
            dict: A dictionary mapping time bucket index to feature matrix
        """        
        room_uris = self.room_uris
        property_types = self.used_property_types

        n_rooms = len(room_uris)
        n_properties = len(property_types)
        
        # Create a mapping from room URI to index
        room_indices = {uri: i for i, uri in enumerate(room_uris)}
        
        # Initialize feature matrices for each time bucket
        feature_matrices = {}
        for time_idx in range(len(self.time_buckets)):
            # Each feature matrix has dimensions (n_rooms, n_properties * 5)
            # For each property: [mean, sd, min, max, has_property]
            feature_matrices[time_idx] = np.zeros((n_rooms, n_properties * 5), dtype=float)
            
            # Set the binary flags (has_property) for all rooms
            for i, room_uri in enumerate(room_uris):
                for j, prop_type in enumerate(property_types):
                    # Check if the room has this property type
                    has_property = False
                    if room_uri in self.office_graph.room_to_property_type:
                        has_property = prop_type in self.office_graph.room_to_property_type[room_uri]
                    
                    # Set binary flag in the last position for each property
                    feature_matrices[time_idx][i, j * 5 + 4] = 1.0 if has_property else 0.0
        
        # Fill in the temporal features for each time bucket
        for room_uri, time_data in temporal_features.items():
            if room_uri not in room_indices:
                continue
                
            room_idx = room_indices[room_uri]
            
            for time_idx, prop_data in time_data.items():
                if time_idx not in feature_matrices:
                    continue
                    
                for prop_type, stats in prop_data.items():
                    if prop_type not in property_types:
                        continue
                        
                    prop_idx = property_types.index(prop_type)
                    
                    # Set the statistics in the feature matrix
                    feature_matrices[time_idx][room_idx, prop_idx * 5] = stats["mean"]
                    feature_matrices[time_idx][room_idx, prop_idx * 5 + 1] = stats["sd"]
                    feature_matrices[time_idx][room_idx, prop_idx * 5 + 2] = stats["min"]
                    feature_matrices[time_idx][room_idx, prop_idx * 5 + 3] = stats["max"]
        
        logger.info(f"Generated feature matrices for {len(feature_matrices)} time buckets")
        return feature_matrices

    def generate_labels(self) -> np.ndarray:
        """
        Generate labels for each time bucket based on whether it's a work hour,
        accounting for Dutch national holidays from the holidays package.
        
        Returns:
            numpy.ndarray: Binary labels (1 for work hour, 0 for non-work hour)
        """        
        labels = np.zeros(len(self.time_buckets), dtype=int)
        
        # Count holidays and work hours for logging
        holiday_hours = 0
        
        for i, (start_time, _) in enumerate(self.time_buckets):
            # Convert datetime to date for holiday checking
            current_date = start_time.date()
            
            # First check if it's a holiday
            if current_date in self.dutch_holidays:
                # It's a holiday, mark as non-work hour
                labels[i] = 0
                holiday_hours += 1
                continue
            
            # If not a holiday, check if it's a weekday (0-4 is Monday-Friday)
            day_of_week = start_time.weekday()
            
            # Check if it's between standard Dutch office hours (9:00-17:00)
            hour_of_day = start_time.hour
            
            if (day_of_week < 5) and (9 <= hour_of_day < 17):
                labels[i] = 1
        
        # Log label distribution
        work_hours = np.sum(labels)
        non_work_hours = len(labels) - work_hours
        
        logger.info(f"Generated labels with holiday awareness: {work_hours} work hours, {non_work_hours} non-work hours")
        logger.info(f"Filtered out {holiday_hours} hours that fell on Dutch holidays")
        logger.info(f"Baseline accuracy: {max(work_hours, non_work_hours) / len(labels) * 100:.2f}%")
        
        return labels
    
    def prepare_stgcn_input(self) -> Dict[str, Any]:
        """
        Prepare all necessary inputs for a STGCN model.
                
        Returns:
            dict: A dictionary containing:
                - adjacency_matrix: Room adjacency matrix
                - room_uris: List of room URIs
                - property_types: List of property types
                - feature_matrices: Dictionary mapping time bucket index to feature matrix
                - time_indices: List of time bucket indices
                - labels: Array of binary labels for each time bucket
                - time_buckets: List of (start_time, end_time) tuples
        """  
        logger.info("Preparing STGCN input...")
        
        # Aggregate temporal features
        temporal_features = self.aggregate_temporal_features()
        
        # Generate time-aware feature matrices
        time_indices = list(range(len(self.time_buckets)))
        feature_matrices = self.generate_time_feature_matrices(temporal_features)
        
        # Generate labels with holiday awareness
        labels = self.generate_labels()
        
        # Package everything into a dictionary
        stgcn_input = {
            "adjacency_matrix": self.adjacency_matrix,
            "room_uris": self.room_uris,
            "property_types": self.used_property_types,
            "feature_matrices": feature_matrices,
            "time_indices": time_indices,
            "labels": labels,
            "time_buckets": self.time_buckets
        }
        
        logger.info("STGCN input preparation complete")
        return stgcn_input
    
    def convert_to_torch_tensors(self, stgcn_input, device="cpu") -> Dict[str, Any]:
        """
        Convert numpy arrays to PyTorch tensors for model input.
        
        Args:
            stgcn_input: Dictionary output from prepare_stgcn_input
            device: PyTorch device to move tensors to
            
        Returns:
            dict: The same dictionary with numpy arrays converted to PyTorch tensors
        """
        torch_input = {}
        
        # Convert adjacency matrix
        torch_input["adjacency_matrix"] = torch.tensor(stgcn_input["adjacency_matrix"], 
                                                      dtype=torch.float32, 
                                                      device=device)
        
        # Convert feature matrices
        torch_input["feature_matrices"] = {}
        for time_idx, feature_matrix in stgcn_input["feature_matrices"].items():
            torch_input["feature_matrices"][time_idx] = torch.tensor(feature_matrix, 
                                                                    dtype=torch.float32, 
                                                                    device=device)
        
        # Convert labels
        torch_input["labels"] = torch.tensor(stgcn_input["labels"], 
                                           dtype=torch.long,
                                           device=device)
        
        # Copy non-tensor data
        torch_input["room_uris"] = stgcn_input["room_uris"]
        torch_input["property_types"] = stgcn_input["property_types"]
        torch_input["time_indices"] = stgcn_input["time_indices"]
        torch_input["time_buckets"] = stgcn_input["time_buckets"]
        
        logger.info("Converted data to PyTorch tensors on device: " + str(device))

        return torch_input

if __name__ == "__main__":

    with open("data/OfficeGraph/processed_data/officegraph.pkl", "rb") as f:
        office_graph = pickle.load(f)
    
    data_prep = TimeSeriesPreparation(office_graph)
    stgcn_input = data_prep.prepare_stgcn_input()

    # Convert to torch tensors
    device = torch.device('cpu') # must move the files to cuda later!
    torch_input = data_prep.convert_to_torch_tensors(stgcn_input, device=device)

    save_path = os.path.join("data", "OfficeGraph", "processed_data", "torch_input.pt")
    torch.save(torch_input, save_path)
    print(f"Saved torch_input to {save_path}")
