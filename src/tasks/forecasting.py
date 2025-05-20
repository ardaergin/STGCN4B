# src/tasks/forecasting.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.sparse as sp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from ..config.args import parse_args
from ..data.load_and_split import load_and_split_data
from ..utils.graph_utils import calc_gso_edge
from ..models.stgcn import EarlyStopping

def setup_model(args, data):
    """Set up the STGCN model and training components."""
    logger.info("Setting up model for forecasting...")
    device   = data['device']
    n_vertex = data['n_vertex']
    
    if args.adjacency_type == "weighted" and args.gso_type not in ("rw_norm_adj", "rw_renorm_adj"):
        raise ValueError(
            f"When adjacency_type='weighted' you must pick gso_type "
            f"in {{'rw_norm_adj','rw_renorm_adj'}}, got '{args.gso_type}'."
        )

    # Build a single static GSO
    static_A = data["adjacency_matrix"]
    edge_index, edge_weight = dense_to_sparse(static_A)
    static_gso = calc_gso_edge(
        edge_index, edge_weight, 
        num_nodes           = n_vertex,
        gso_type            = args.gso_type,
        device              = device,
    )
    if args.gso_mode == "static":
        gso = static_gso

    # Build dynamic GSOs
    elif args.gso_mode == "dynamic":
        dynamic_adjacencies_dict = data.get("dynamic_adjacencies", {})
        dynamic_adjacencies = list(dynamic_adjacencies_dict.values())
        dynamic_adjacencies = dynamic_adjacencies[: args.stblock_num]
        dynamic_gsos = []
        for adjacency_matrix in dynamic_adjacencies:
            edge_index, edge_weight = dense_to_sparse(adjacency_matrix)
            G = calc_gso_edge(
                edge_index, edge_weight, 
                num_nodes           = n_vertex,
                gso_type            = args.gso_type,
                device              = device,
            )
            dynamic_gsos.append(G)

        # In case we've got fewer than stblock_num, pad with the static GSO
        while len(dynamic_gsos) < args.stblock_num:
            dynamic_gsos.append(static_gso)
        
        gso = dynamic_gsos
    
    else:
        raise ValueError(f"Unknown gso_mode: {args.gso_mode!r}. Must be 'static' or 'dynamic'.")

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
        
    # Output is a single continuous value
    blocks.append([1])  
    
    # Create model based on graph convolution type
    if args.graph_conv_type == 'cheb_graph_conv':
        from ..models.stgcn.models import STGCNChebGraphConv as Model
    else:
        from ..models.stgcn.models import STGCNGraphConv as Model

    model = Model(
        args     = args,
        blocks   = blocks,
        n_vertex = n_vertex,
        gso      = gso,
        task_type= 'forecasting',
    ).to(device)

    # Set loss function for regression
    criterion = torch.nn.MSELoss()
    
    # Set optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    logger.info(f"Model setup complete with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, criterion, optimizer, scheduler, early_stopping


def train_model(args, model, criterion, optimizer, scheduler, early_stopping, train_loader, val_loader):
    """Train the STGCN model for forecasting."""
    logger.info("Starting model training...")
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            
            # Ensure shapes match for loss calculation
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(1)  # [B] -> [B, 1]
            if len(y_batch.shape) == 1:
                y_batch = y_batch.unsqueeze(1)  # [B] -> [B, 1]
            
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
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                # Forward pass
                outputs = model(X_batch)
                
                # Ensure shapes match for loss calculation
                if len(outputs.shape) == 1:
                    outputs = outputs.unsqueeze(1)  # [B] -> [B, 1]
                if len(y_batch.shape) == 1:
                    y_batch = y_batch.unsqueeze(1)  # [B] -> [B, 1]
                
                # Compute loss
                loss = criterion(outputs, y_batch.float())
                val_loss += loss.item() * X_batch.size(0)
                
                # Store predictions and targets for metrics calculation
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        # Average validation loss for the epoch
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate validation R² score
        val_r2 = r2_score(all_targets, all_preds)
        val_r2_scores.append(val_r2)
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch results
        logger.info(f"Epoch [{epoch+1}/{args.epochs}] - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val R²: {val_r2:.4f}, "
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
        'val_r2': val_r2_scores
    }
    
    return model, history


def evaluate_model(model, criterion, test_loader):
    """Evaluate the trained model on the test set."""
    logger.info("Evaluating model on test set...")
    
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Forward pass
            outputs = model(X_batch)
            
            # Ensure shapes match for loss calculation
            if len(outputs.shape) == 1:
                outputs = outputs.unsqueeze(1)  # [B] -> [B, 1]
            if len(y_batch.shape) == 1:
                y_batch = y_batch.unsqueeze(1)  # [B] -> [B, 1]
            
            # Compute loss
            loss = criterion(outputs, y_batch.float())
            test_loss += loss.item() * X_batch.size(0)
            
            # Store predictions and targets for metrics calculation
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    # Reshape for metric calculations
    all_preds = np.array(all_preds).reshape(-1)
    all_targets = np.array(all_targets).reshape(-1)
    
    # Average test loss
    test_loss = test_loss / len(test_loader.dataset)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = all_targets != 0
    mape = np.mean(np.abs((all_targets[mask] - all_preds[mask]) / all_targets[mask])) * 100
    
    metrics = {
        'test_loss': test_loss,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'predictions': all_preds,
        'targets': all_targets
    }

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"R²: {r2:.4f}")
    logger.info(f"MAPE: {mape:.2f}%")
    
    return metrics


def plot_results(args, history, metrics):
    """Plot and save training curves and evaluation results."""
    logger.info("Plotting results...")
    
    # Create output directory
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot validation R² score
    plt.subplot(2, 2, 2)
    plt.plot(history['val_r2'], label='Validation R²', color='green')
    plt.axhline(y=metrics['r2'], color='r', linestyle='--', 
                label=f'Test R²: {metrics["r2"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('R² Score')
    plt.title('R² Score Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot actual vs predicted values
    plt.subplot(2, 2, 3)
    plt.scatter(metrics['targets'], metrics['predictions'], alpha=0.5)
    min_val = min(np.min(metrics['targets']), np.min(metrics['predictions']))
    max_val = max(np.max(metrics['targets']), np.max(metrics['predictions']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot prediction error histogram
    plt.subplot(2, 2, 4)
    errors = metrics['predictions'] - metrics['targets']
    plt.hist(errors, bins=20, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Error Distribution (RMSE: {metrics["rmse"]:.4f})')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stgcn_forecasting_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'stgcn_forecasting_metrics.txt'), 'w') as f:
        f.write(f"Test Loss (MSE): {metrics['test_loss']:.4f}\n")
        f.write(f"RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"MAE: {metrics['mae']:.4f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
        f.write(f"MAPE: {metrics['mape']:.2f}%\n")
    
    # Save time series plot of predictions vs actual for a sample
    plt.figure(figsize=(12, 6))
    sample_size = min(100, len(metrics['targets']))
    indices = np.arange(sample_size)
    plt.plot(indices, metrics['targets'][:sample_size], label='Actual', marker='o', markersize=4, alpha=0.7)
    plt.plot(indices, metrics['predictions'][:sample_size], label='Predicted', marker='x', markersize=4, alpha=0.7)
    plt.xlabel('Time Step')
    plt.ylabel('Consumption Value')
    plt.title('Predicted vs Actual Consumption (Sample)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'stgcn_forecasting_timeseries.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main function to run the STGCN model for forecasting."""
    # Parse arguments
    args = parse_args()
    args.task_type = 'forecasting'
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Ensure task type is set to forecasting
    args.task_type = 'forecasting'
    
    # Prepare data
    data = load_and_split_data(args)
    
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
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'stgcn_forecasting_model.pt'))
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
