import os
import sys
import logging
import torch
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def load_and_split_data(args):
    """
    Load data for STGCN model using pre-computed splits from OfficeGraphBuilder.
    
    Args:
        args: Command line arguments
        
    Returns:
        Dictionary containing data loaders and metadata
    """
    logger.info("Loading processed OfficeGraph data...")
    
    # Load the pre-processed torch input
    file_name = f"torch_input_{args.adjacency_type}.pt"
    torch_input_path = os.path.join(args.data_dir, "processed", file_name)
    logger.info(f"Loading torch input from {torch_input_path}")
    
    # Load on CPU first
    torch_input = torch.load(torch_input_path, map_location='cpu')
    
    # Determine device
    device = torch.device('cuda') if args.enable_cuda and torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Move tensors to the appropriate device
    torch_input["adjacency_matrix"] = torch_input["adjacency_matrix"].to(device)
    
    # Move dynamic adjacencies to device
    if "dynamic_adjacencies" in torch_input:
        for step in torch_input["dynamic_adjacencies"]:
            torch_input["dynamic_adjacencies"][step] = torch_input["dynamic_adjacencies"][step].to(device)
    
    # Move appropriate tensor based on task
    if args.task_type == 'classification':
        torch_input["workhour_labels"] = torch_input["workhour_labels"].to(device)
    elif args.task_type == 'forecasting':
        torch_input["consumption_values"] = torch_input["consumption_values"].to(device)
    
    for time_idx in torch_input["feature_matrices"]:
        torch_input["feature_matrices"][time_idx] = torch_input["feature_matrices"][time_idx].to(device)
    
    # Get all time indices
    time_indices = sorted(torch_input["time_indices"])
    total_samples = len(time_indices)
    
    # Use pre-computed splits from the torch_input
    train_indices = torch_input["train_idx"]
    val_indices = torch_input["val_idx"]
    test_indices = torch_input["test_idx"]
    
    logger.info(f"Using pre-computed data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
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
    
    # Get dimensions
    n_vertex = torch_input["adjacency_matrix"].shape[0]  # Number of rooms
    n_features = X_train.shape[1]  # Number of features per vertex
    
    logger.info(f"Data preparation completed. Number of rooms: {n_vertex}, Features: {n_features}")
    
    # Return only the data loaders and metadata
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'adjacency_matrix': torch_input["adjacency_matrix"],  # Raw adjacency matrix
        'dynamic_adjacencies': torch_input["dynamic_adjacencies"] if "dynamic_adjacencies" in torch_input else None,
        'n_vertex': n_vertex,
        'n_features': n_features,
        'room_uris': torch_input["room_uris"],
        'property_types': torch_input["property_types"] if "property_types" in torch_input else None,
        'time_buckets': torch_input["time_buckets"] if "time_buckets" in torch_input else None,
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
            X shape: [num_samples, 1, n_his, n_rooms] - mean across features
            y shape: [num_samples] for classification, [num_samples, 1] for forecasting
    """
    device = torch_input["feature_matrices"][indices[0]].device
    feature_matrices = torch_input["feature_matrices"]
    
    # Get the appropriate target based on task type
    if task_type == 'classification':
        # Use "workhour_labels" for classification task
        targets = torch_input["workhour_labels"]
    else:  # forecasting
        # Use "consumption_values" for forecasting task
        targets = torch_input["consumption_values"]

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

        # Take mean across features to get shape [n_his, n_rooms]
        x_seq = x_seq.mean(dim=2, keepdim=False)  # [n_his, n_rooms]

        # Add batch and channel dimensions
        x_seq = x_seq.unsqueeze(0)  # [1, n_his, n_rooms]

        # Add to list
        X_list.append(x_seq)
        y_list.append(targets[indices[i]])

    if not X_list:
        logger.warning(f"No valid sequences found for indices of length {len(indices)}")
        # Return empty tensors with correct shapes if no valid sequences
        n_rooms = feature_matrices[indices[0]].shape[0]
        return (
            torch.empty((0, 1, n_his, n_rooms), device=device),
            torch.empty((0), device=device) if task_type == 'classification' else torch.empty((0, 1), device=device)
        )

    # Final shape: [batch, 1, n_his, n_rooms]
    X = torch.stack(X_list).to(device)
    
    # For classification, stack normally
    # For forecasting, reshape to [batch, 1] to match regression output
    if task_type == 'classification':
        y = torch.stack(y_list).to(device)
    else:  # forecasting
        y = torch.stack(y_list).to(device).view(-1, 1)

    logger.info(f"Final feature tensor shape: {X.shape}")  # [batch, 1, n_his, rooms]
    logger.info(f"Final target shape: {y.shape}")

    return X, y
