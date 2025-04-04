#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
import logging
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
import scipy.sparse as sp

# Import OfficeGraph classes
from ...graph import OfficeGraph, TimeSeriesPreparation

# Import our STGCN implementation
from .utils import prepare_graph_data
from .models import STGCNChebGraphConv, STGCNGraphConv

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.best_model_state = model.state_dict().copy()
        self.val_loss_min = val_loss


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='STGCN for OfficeGraph Classification')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='data/OfficeGraph', 
                        help='Path to OfficeGraph data directory')
    parser.add_argument('--start_time', type=str, default='2022-03-01 00:00:00', 
                        help='Start time for analysis')
    parser.add_argument('--end_time', type=str, default='2023-01-30 00:00:00', 
                        help='End time for analysis')
    parser.add_argument('--interval_hours', type=int, default=1, 
                        help='Size of time buckets in hours')
    parser.add_argument('--output_dir', type=str, default='./output', 
                        help='Directory to save results')
    
    # Model parameters
    parser.add_argument('--n_his', type=int, default=12, 
                        help='Number of historical time steps to use')
    parser.add_argument('--Kt', type=int, default=3, 
                        help='Kernel size in temporal convolution')
    parser.add_argument('--Ks', type=int, default=3, 
                        help='Kernel size in graph convolution')
    parser.add_argument('--stblock_num', type=int, default=2, 
                        help='Number of ST-Conv blocks')
    parser.add_argument('--act_func', type=str, default='glu', 
                        choices=['glu', 'gtu', 'relu', 'silu'], 
                        help='Activation function')
    parser.add_argument('--graph_conv_type', type=str, default='cheb_graph_conv', 
                        choices=['cheb_graph_conv', 'graph_conv'], 
                        help='Graph convolution type')
    parser.add_argument('--droprate', type=float, default=0.5, 
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--enable_cuda', action='store_true', 
                        help='Enable CUDA')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--test_ratio', type=float, default=0.2, 
                        help='Test set ratio')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, 
                        help='Weight decay')
    parser.add_argument('--step_size', type=int, default=10, 
                        help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.9, 
                        help='Gamma for learning rate scheduler')
    parser.add_argument('--patience', type=int, default=15, 
                        help='Patience for early stopping')
    parser.add_argument('--optimizer', type=str, default='adamw', 
                        choices=['adam', 'adamw', 'sgd'], 
                        help='Optimizer type')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    return args

def prepare_data(args):
    """Prepare data for STGCN model with seasonal-aware time splits that preserve temporal ordering."""
    logger.info("Loading processed OfficeGraph data...")
    
    # Load the pre-processed torch input
    torch_input_path = os.path.join(args.data_dir, "processed_data", "torch_input.pt")
    logger.info(f"Loading torch input from {torch_input_path}")
    
    # Load on CPU first
    torch_input = torch.load(torch_input_path, map_location='cpu')
    
    # Determine device
    device = torch.device('cuda') if args.enable_cuda and torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Move tensors to the appropriate device
    torch_input["adjacency_matrix"] = torch_input["adjacency_matrix"].to(device)
    torch_input["labels"] = torch_input["labels"].to(device)
    
    for time_idx in torch_input["feature_matrices"]:
        torch_input["feature_matrices"][time_idx] = torch_input["feature_matrices"][time_idx].to(device)
    
    # Get all time indices
    time_indices = sorted(torch_input["time_indices"])
    total_samples = len(time_indices)
    
    # Group time indices by month to analyze distribution
    months = []
    month_to_indices = {}
    for idx in time_indices:
        # Time buckets contain (start_time, end_time) tuples
        start_time = torch_input["time_buckets"][idx][0]
        month = start_time.month
        months.append(month)
        
        if month not in month_to_indices:
            month_to_indices[month] = []
        month_to_indices[month].append(idx)
    
    # Log the distribution of months in the dataset
    month_counts = {}
    for m in range(1, 13):
        month_counts[m] = months.count(m)
    logger.info(f"Month distribution in dataset: {month_counts}")
    
    # Create temporally contiguous blocks for each month
    month_blocks = []
    for month, indices in month_to_indices.items():
        # Sort indices within each month to maintain temporal order
        sorted_indices = sorted(indices)
        
        # Split into smaller contiguous blocks (e.g., 1-week blocks)
        # This allows us to distribute each month's data while maintaining temporal coherence
        block_size = 24 * 7  # 1 week of hourly data (adjust as needed)
        month_blocks.extend([sorted_indices[i:i+block_size] for i in range(0, len(sorted_indices), block_size)])
    
    # Randomly shuffle the blocks (not the time points within blocks)
    np.random.shuffle(month_blocks)
    
    # Calculate split sizes
    test_ratio = args.test_ratio
    val_ratio = args.test_ratio
    total_blocks = len(month_blocks)
    test_blocks = int(total_blocks * test_ratio)
    val_blocks = int(total_blocks * val_ratio)
    train_blocks = total_blocks - test_blocks - val_blocks
    
    # Split the blocks into train/val/test
    test_block_indices = month_blocks[:test_blocks]
    val_block_indices = month_blocks[test_blocks:test_blocks+val_blocks]
    train_block_indices = month_blocks[test_blocks+val_blocks:]
    
    # Flatten the block indices
    test_indices = [idx for block in test_block_indices for idx in block]
    val_indices = [idx for block in val_block_indices for idx in block]
    train_indices = [idx for block in train_block_indices for idx in block]
    
    # Sort indices within each split to maintain temporal order for sequence modeling
    test_indices.sort()
    val_indices.sort()
    train_indices.sort()
    
    logger.info(f"Data split: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    
    # Check seasonal distribution in each split
    train_months = [torch_input["time_buckets"][i][0].month for i in train_indices]
    val_months = [torch_input["time_buckets"][i][0].month for i in val_indices]
    test_months = [torch_input["time_buckets"][i][0].month for i in test_indices]
    
    for split_name, split_months in [("Train", train_months), ("Val", val_months), ("Test", test_months)]:
        month_counts = {}
        for m in range(1, 13):
            month_counts[m] = split_months.count(m)
        logger.info(f"{split_name} set month distribution: {month_counts}")
    
    # Extract features and labels for each set
    X_train, y_train = extract_features_labels(torch_input, train_indices, n_his=args.n_his)
    X_val, y_val = extract_features_labels(torch_input, val_indices, n_his=args.n_his)
    X_test, y_test = extract_features_labels(torch_input, test_indices, n_his=args.n_his)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Note: for train_loader, we don't shuffle since order matters for temporal sequences
    # The randomization is already handled at block level
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

def extract_features_labels(torch_input, indices, n_his=12):
    """
    Extract sequences of features and corresponding labels for STGCN input.

    Args:
        torch_input (dict): Dictionary of tensors from `convert_to_torch_tensors()`
        indices (List[int]): Time indices to extract data from
        n_his (int): Number of historical time steps to include in the sequence

    Returns:
        Tuple[Tensor, Tensor]: (X, y)
            X shape: [num_samples, n_features, n_his, n_rooms]
            y shape: [num_samples]
    """
    device = torch_input["labels"].device
    feature_matrices = torch_input["feature_matrices"]
    labels = torch_input["labels"]

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
        y_list.append(labels[indices[i]])

    # Final shape: [batch, n_features, n_his, n_rooms]
    X = torch.stack(X_list).to(device)
    y = torch.stack(y_list).to(device)

    print(f"Final feature tensor shape: {X.shape}")  # [batch, features, n_his, rooms]
    print(f"Final label shape: {y.shape}")

    return X, y


def setup_model(args, data):
    """Set up the STGCN model and training components."""
    logger.info("Setting up model...")
    
    # Get device
    device = data['device']
    
    # Create block structure for STGCN
    blocks = []
    blocks.append([data["n_features"]])  # Input features
    
    # Add intermediate blocks
    for _ in range(args.stblock_num):
        blocks.append([64, 16, 64])
        
    # Add output blocks
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
        
    # Output is binary classification (1 or 0)
    blocks.append([1])  
    
    # Create model based on graph convolution type
    if args.graph_conv_type == 'cheb_graph_conv':
        model = STGCNChebGraphConv(args, blocks, data["n_vertex"], data["gso"]).to(device)
    else:
        model = STGCNGraphConv(args, blocks, data["n_vertex"], data["gso"]).to(device)
    
    # Binary classification loss function
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Set optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    logger.info(f"Model setup complete with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, criterion, optimizer, scheduler, early_stopping


def train_model(args, model, criterion, optimizer, scheduler, early_stopping, train_loader, val_loader):
    """Train the STGCN model."""
    logger.info("Starting model training...")
    
    # Get device
    device = next(model.parameters()).device
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch).squeeze()
            
            # Compute loss
            loss = criterion(outputs, y_batch.float())
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
        
        # Average training loss for the epoch
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Forward pass
                outputs = model(X_batch).squeeze()
                
                # Compute loss
                loss = criterion(outputs, y_batch.float())
                val_loss += loss.item() * X_batch.size(0)
                
                # Store predictions and labels for accuracy calculation
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        # Average validation loss for the epoch
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate validation accuracy
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_accuracy)
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch results
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.4f}, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break
    
    # Load the best model
    model.load_state_dict(early_stopping.best_model_state)
    
    # Return training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies
    }
    
    return model, history


def evaluate_model(model, criterion, test_loader):
    """Evaluate the trained model on the test set."""
    logger.info("Evaluating model on test set...")
    
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Forward pass
            outputs = model(X_batch).squeeze()
            
            # Compute loss
            loss = criterion(outputs, y_batch.float())
            test_loss += loss.item() * X_batch.size(0)
            
            # Store predictions, probabilities, and labels
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    # Average test loss
    test_loss = test_loss / len(test_loader.dataset)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }
    
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1-score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Calculate baseline accuracy (always predicting the majority class)
    pos_count = sum(all_labels)
    neg_count = len(all_labels) - pos_count
    baseline = max(pos_count, neg_count) / len(all_labels)
    logger.info(f"Baseline Accuracy (majority class): {baseline:.4f}")
    logger.info(f"Improvement over baseline: {(accuracy - baseline) / baseline * 100:.2f}%")
    
    return metrics


def plot_results(args, history, metrics):
    """Plot and save training curves and evaluation results."""
    logger.info("Plotting results...")
    
    # Create output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot validation accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy', color='green')
    plt.axhline(y=metrics['accuracy'], color='r', linestyle='--', 
                label=f'Test Accuracy: {metrics["accuracy"]:.4f}')
    
    # Add baseline
    pos_count = sum(metrics['labels'])
    neg_count = len(metrics['labels']) - pos_count
    baseline = max(pos_count, neg_count) / len(metrics['labels'])
    plt.axhline(y=baseline, color='grey', linestyle=':', 
                label=f'Baseline: {baseline:.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot confusion matrix
    plt.subplot(1, 3, 3)
    conf_mat = metrics['confusion_matrix']
    labels = ['Non-Work Hours', 'Work Hours']
    
    plt.imshow(conf_mat, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    
    # Add values in cells
    thresh = conf_mat.max() / 2
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            plt.text(j, i, conf_mat[i, j],
                     ha="center", va="center",
                     color="white" if conf_mat[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stgcn_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'stgcn_metrics.txt'), 'w') as f:
        f.write(f"Test Loss: {metrics['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Test Precision: {metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {metrics['recall']:.4f}\n")
        f.write(f"Test F1-score: {metrics['f1']:.4f}\n")
        f.write(f"Baseline Accuracy: {baseline:.4f}\n")
        f.write(f"Improvement over baseline: {(metrics['accuracy'] - baseline) / baseline * 100:.2f}%\n")
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main function to run the STGCN model for OfficeGraph classification."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Prepare data
    data = prepare_data(args)
    
    # Setup model
    model, criterion, optimizer, scheduler, early_stopping = setup_model(args, data)
    
    # Train model
    model, history = train_model(
        args, model, criterion, optimizer, scheduler, early_stopping,
        data['train_loader'], data['val_loader']
    )
    
    # Evaluate model
    metrics = evaluate_model(model, criterion, data['test_loader'])
    
    # Plot results
    plot_results(args, history, metrics)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'stgcn_model.pt'))
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
