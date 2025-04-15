import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import scipy.sparse as sp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from ..utils import prepare_graph_data

def load_and_split_data(args):
    """
    Prepare data for STGCN model using cyclic block sampling to ensure balanced distribution
    across the entire time range. 
    
    Supports both classification and forecasting tasks.
    """
    logger.info("Loading processed OfficeGraph data...")
    
    # Load the pre-processed torch input
    file_name = f"torch_input_for_{args.task_type}.pt"
    torch_input_path = os.path.join(args.data_dir, "processed", file_name)
    logger.info(f"Loading torch input from {torch_input_path}")
    
    # Load on CPU first
    torch_input = torch.load(torch_input_path, map_location='cpu')
    
    # Determine device
    device = torch.device('cuda') if args.enable_cuda and torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Move tensors to the appropriate device
    torch_input["adjacency_matrix"] = torch_input["adjacency_matrix"].to(device)
    
    # Move appropriate tensor based on task
    if args.task_type == 'classification':
        torch_input["labels"] = torch_input["labels"].to(device)
    elif args.task_type == 'forecasting':
        torch_input["values"] = torch_input["values"].to(device)
    
    for time_idx in torch_input["feature_matrices"]:
        torch_input["feature_matrices"][time_idx] = torch_input["feature_matrices"][time_idx].to(device)
    
    # Get all time indices and sort them
    time_indices = sorted(torch_input["time_indices"])
    total_samples = len(time_indices)
    
    # Define block size based on whether to include Sundays
    if args.include_sundays:
        block_size = 24 * 7  # Full week
        logger.info("Using 7-day blocks (including Sundays)")
    else:
        block_size = 24 * 6  # Excluding Sundays
        logger.info("Using 6-day blocks (excluding Sundays)")

    # Create blocks of contiguous time points
    blocks = []
    for i in range(0, len(time_indices), block_size):
        # Take up to block_size indices (last block might be smaller)
        block = time_indices[i:i+block_size]
        blocks.append(block)
    
    logger.info(f"Created {len(blocks)} blocks of data (each ~1 week)")
    
    # Define the sampling pattern for train/val/test
    # 3:1:1 ratio (train:test:val)
    sampling_pattern = ["train", "train", "train", "test", "val"]
    
    # Initialize sets for each split
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Assign blocks to splits using cyclic sampling without replacement
    current_pattern = sampling_pattern.copy()
    
    for i, block in enumerate(blocks):
        # If we've used all elements in the pattern, replenish it
        if not current_pattern:
            current_pattern = sampling_pattern.copy()
        
        # Randomly select an element from the current pattern
        split_type = np.random.choice(current_pattern)
        current_pattern.remove(split_type)
        
        # Assign the block to the corresponding split
        if split_type == "train":
            train_indices.extend(block)
        elif split_type == "val":
            val_indices.extend(block)
        else:  # "test"
            test_indices.extend(block)
        
        # Log the assignment
        if i < 5 or i == len(blocks)-1:  # Just log a few blocks to avoid verbose output
            start_date = torch_input["time_buckets"][block[0]][0]
            end_date = torch_input["time_buckets"][block[-1]][0]
            logger.info(f"Block {i+1}/{len(blocks)}: {len(block)} points from {start_date} to {end_date} assigned to {split_type}")
    
    # Sort indices within each split to maintain temporal order
    train_indices.sort()
    val_indices.sort()
    test_indices.sort()
    
    logger.info(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Check temporal distribution in each split
    train_dates = [torch_input["time_buckets"][i][0] for i in train_indices]
    val_dates = [torch_input["time_buckets"][i][0] for i in val_indices]
    test_dates = [torch_input["time_buckets"][i][0] for i in test_indices]
    
    # Log date ranges
    logger.info(f"Train date range: {min(train_dates)} to {max(train_dates)}")
    logger.info(f"Val date range: {min(val_dates)} to {max(val_dates)}")
    logger.info(f"Test date range: {min(test_dates)} to {max(test_dates)}")
    
    # Check month distribution in each split
    train_months = [date.month for date in train_dates]
    val_months = [date.month for date in val_dates]
    test_months = [date.month for date in test_dates]
    
    for split_name, split_months in [("Train", train_months), ("Val", val_months), ("Test", test_months)]:
        month_counts = {}
        for m in range(1, 13):
            month_counts[m] = split_months.count(m)
        logger.info(f"{split_name} set month distribution: {month_counts}")
    
    # Log data distribution based on task type
    if args.task_type == 'classification':
        # Log class distribution in each split
        train_labels = [torch_input["labels"][i].item() for i in train_indices]
        val_labels = [torch_input["labels"][i].item() for i in val_indices]
        test_labels = [torch_input["labels"][i].item() for i in test_indices]
        
        for split_name, split_labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
            n_work_hours = sum(split_labels)
            n_non_work_hours = len(split_labels) - n_work_hours
            logger.info(f"{split_name} set class distribution: Work hours={n_work_hours}, Non-work hours={n_non_work_hours}")

    elif args.task_type == 'forecasting':
        # Log value statistics in each split
        train_values = [torch_input["values"][i].item() for i in train_indices]
        val_values = [torch_input["values"][i].item() for i in val_indices]
        test_values = [torch_input["values"][i].item() for i in test_indices]
        
        for split_name, split_values in [("Train", train_values), ("Val", val_values), ("Test", test_values)]:
            if split_values:
                logger.info(f"{split_name} set value statistics: Min={min(split_values):.2f}, Max={max(split_values):.2f}, Mean={np.mean(split_values):.2f}, Std={np.std(split_values):.2f}")
            else:
                logger.warning(f"{split_name} set has no values")
    
    # Extract features and targets for each set
    X_train, y_train = extract_features_targets(torch_input, train_indices, n_his=args.n_his, task_type=args.task_type)
    X_val, y_val = extract_features_targets(torch_input, val_indices, n_his=args.n_his, task_type=args.task_type)
    X_test, y_test = extract_features_targets(torch_input, test_indices, n_his=args.n_his, task_type=args.task_type)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Create data loaders (no shuffling for sequence data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get adjacency matrix for room graph
    adjacency_matrix = torch_input["adjacency_matrix"].cpu().numpy()
    
    # Convert adjacency matrix to sparse format and prepare for graph convolution
    adj_sparse = sp.coo_matrix(adjacency_matrix)
    gso = prepare_graph_data(adj_sparse, graph_conv_type=args.graph_conv_type, K=args.Ks)
    gso = gso.to(device)
    
    # Get dimensions
    n_vertex = adjacency_matrix.shape[0]  # Number of rooms
    n_features = X_train.shape[1]  # Number of features per vertex
    
    logger.info(f"Data preparation completed. Number of rooms: {n_vertex}, Features: {n_features}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'gso': gso,
        'n_vertex': n_vertex,
        'n_features': n_features,
        'room_uris': torch_input["room_uris"],
        'property_types': torch_input["property_types"],
        'time_buckets': torch_input["time_buckets"],
        'device': device
    }

def extract_features_targets(torch_input, indices, n_his=12, task_type='classification'):
    """
    Extract sequences of features and corresponding targets for STGCN input.
    Supports both classification (labels) and forecasting (values) tasks.

    Args:
        torch_input (dict): Dictionary of tensors from `convert_to_torch_tensors()`
        indices (List[int]): Time indices to extract data from
        n_his (int): Number of historical time steps to include in the sequence
        task_type (str): Type of task - 'classification' or 'forecasting'

    Returns:
        Tuple[Tensor, Tensor]: (X, y)
            X shape: [num_samples, n_features, n_his, n_rooms]
            y shape: [num_samples] for classification, [num_samples, 1] for forecasting
    """
    device = torch_input["feature_matrices"][indices[0]].device
    feature_matrices = torch_input["feature_matrices"]
    
    # Get the appropriate target based on task type
    if task_type == 'classification':
        targets = torch_input["labels"]
    else:  # forecasting
        targets = torch_input["values"]

    X_list = []
    y_list = []

    # Sort the indices to preserve time order
    indices = sorted(indices)

    for i in range(n_his - 1, len(indices)):
        time_window = indices[i - n_his + 1 : i + 1]

        # Make sure all time steps are present in the feature matrices
        if not all(t in feature_matrices for t in time_window):
            continue

        # Stack the sequence: shape will be [n_his, n_rooms, n_features]
        x_seq = torch.stack([feature_matrices[t] for t in time_window], dim=0)

        # Transpose to [n_features, n_his, n_rooms]
        x_seq = x_seq.permute(2, 0, 1)  # [n_features, n_his, n_rooms]

        # Add batch dimension later via stacking
        X_list.append(x_seq)
        y_list.append(targets[indices[i]])

    if not X_list:
        logger.warning(f"No valid sequences found for indices of length {len(indices)}")
        # Return empty tensors with correct shapes if no valid sequences
        n_features = feature_matrices[indices[0]].shape[1]
        n_rooms = feature_matrices[indices[0]].shape[0]
        return (
            torch.empty((0, n_features, n_his, n_rooms), device=device),
            torch.empty((0), device=device) if task_type == 'classification' else torch.empty((0, 1), device=device)
        )

    # Final shape: [batch, n_features, n_his, n_rooms]
    X = torch.stack(X_list).to(device)
    
    # For classification, stack normally
    # For forecasting, reshape to [batch, 1] to match regression output
    if task_type == 'classification':
        y = torch.stack(y_list).to(device)
    else:  # forecasting
        y = torch.stack(y_list).to(device).view(-1, 1)

    logger.info(f"Final feature tensor shape: {X.shape}")  # [batch, features, n_his, rooms]
    logger.info(f"Final target shape: {y.shape}")

    return X, y
