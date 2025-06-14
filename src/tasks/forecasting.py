#!/usr/bin/env python

import os
import sys
import logging
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from ..config.args import parse_args
from ..data.Loader.graph_loader import load_data
from ..utils.graph_utils import calc_gso_edge
from ..utils.early_stopping import EarlyStopping

def setup_model(args, data):
    """Set up the STGCN model and training components."""
    logger.info("Setting up model for forecasting...")
    device   = data['device']
    logger.info(f"Device: {device}")
    n_nodes = data['n_nodes']
    logger.info(f"Number of nodes: {n_nodes}")
    n_features = data["n_features"]
    logger.info(f"Number of features: {n_features}")

    
    if args.adjacency_type == "weighted" and args.gso_type not in ("rw_norm_adj", "rw_renorm_adj"):
        raise ValueError(
            f"When adjacency_type='weighted' you must pick gso_type "
            f"in {{'rw_norm_adj','rw_renorm_adj'}}, got '{args.gso_type}'."
        )

    # Build a single static GSO
    static_A = data["adjacency_matrix"]
    edge_index, edge_weight = dense_to_sparse(static_A)
    logger.info("edge_index shape:", edge_index.shape)
    logger.info("edge_weight shape:", edge_weight.shape)

    static_gso = calc_gso_edge(
        edge_index, edge_weight, 
        num_nodes           = n_nodes,
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
                num_nodes           = n_nodes,
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
    
    batch_sample = next(iter(data['train_loader']))
    X_batch_list, y_batch = batch_sample[0], batch_sample[1] # Unpack first two elements, ignore the mask
    logger.info(f"Sample input shape (X[0]): {X_batch_list[0].shape}  # shape=(batch_size, n_nodes, n_features)")
    logger.info(f"Sample target shape: {y_batch.shape}")
    
    blocks = []
    blocks.append([n_features])  # Input features
    logger.info(f"Model first block input dimension: {blocks[0][0]}")

    # Add intermediate blocks
    for _ in range(args.stblock_num):
        blocks.append([64, 16, 64])
        
    # Add output blocks
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
        
    blocks.append([1])
    
    # Create model based on graph convolution type
    if args.graph_conv_type == 'cheb_graph_conv':
        from ..models.STGCN4B.homogeneous.models import STGCNChebGraphConv as Model
    else:
        from ..models.STGCN4B.homogeneous.models import STGCNGraphConv as Model
    
    model = Model(
        args     = args,
        blocks   = blocks,
        n_vertex = n_nodes,
        gso      = gso,
        task_type= args.task_type,
    ).to(device)
    
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
    
    return model, optimizer, scheduler, early_stopping


def train_model(args, model, optimizer, scheduler, early_stopping, train_loader, val_loader):
    """Train the STGCN model for forecasting."""
    logger.info("Starting model training...")
    
    train_losses = []
    val_losses = []
    val_r2_scores = []
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        total_valid_points_train = 0
        
        for X_batch, y_batch, mask_batch in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Convert X_batch: List[T × (B, R, F)] → (B, T, R, F) → (B, F, T, R)
            x = torch.stack(X_batch, dim=1).permute(0, 3, 1, 2)

            # Forward pass
            outputs = model(x)
            
            if args.task_type == "consumption_forecast":
                preds = outputs.view(-1)                # (B, 1, 1, 1) or (B, 1) -> (B)
                targets = y_batch.view(-1).float()      # (B, 1) -> (B)
                mask = mask_batch.view(-1)              # (B, 1) -> (B)
            else: # args.task_type == "measurement_forecast"
                preds   = outputs                       # (B, n_pred, N) -> No change
                targets = y_batch.float()               # (B, n_pred, N) -> No change, just ensuring float type
                mask = mask_batch                       # (B, n_pred, N) -> No change

            # Manually calculate squared error and apply mask
            error = preds - targets
            masked_squared_error = (error ** 2) * mask
            loss = torch.sum(masked_squared_error)
        
            # Update running loss and count of valid points
            running_loss += loss.item()
            total_valid_points_train += torch.sum(mask).item()

            # Backward pass
            loss.backward()
            optimizer.step()
                            
        # Average training loss for the epoch
        epoch_train_loss = running_loss / total_valid_points_train if total_valid_points_train > 0 else 0.0
        train_losses.append(epoch_train_loss)
                
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        total_valid_points_val = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch, mask_batch in val_loader:

                # Convert X_batch: List[T × (B, R, F)] → (B, T, R, F) → (B, F, T, R)
                x = torch.stack(X_batch, dim=1).permute(0, 3, 1, 2)

                # Forward pass
                outputs = model(x)
                
                if args.task_type == "consumption_forecast":
                    preds = outputs.view(-1)                # (B, 1, 1, 1) or (B, 1) -> (B)
                    targets = y_batch.view(-1).float()      # (B, 1) -> (B)
                    mask = mask_batch.view(-1)              # (B, 1) -> (B)
                else: # args.task_type == "measurement_forecast"
                    preds   = outputs                       # (B, n_pred, N) -> No change
                    targets = y_batch.float()               # (B, n_pred, N) -> No change, just ensuring float type
                    mask = mask_batch                       # (B, n_pred, N) -> No change

                error = preds - targets
                masked_squared_error = (error ** 2) * mask
                loss_val = torch.sum(masked_squared_error)

                running_val_loss += loss_val.item()
                total_valid_points_val += torch.sum(mask).item()

                # Collect only valid predictions and targets for R² score
                valid_preds = preds[mask_batch == 1]
                valid_targets = targets[mask_batch == 1]
                all_preds.extend(valid_preds.cpu().tolist())
                all_targets.extend(valid_targets.cpu().tolist())

        # Average validation loss (MSE) for the epoch
        epoch_val_loss = running_val_loss / total_valid_points_val if total_valid_points_val > 0 else 0.0
        val_losses.append(epoch_val_loss)
        
        # Calculate validation R² score on the collected (and correctly masked) data
        epoch_r2 = r2_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
        val_r2_scores.append(epoch_r2)
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch results
        logger.info(
            f"Epoch [{epoch+1}/{args.epochs}]  "
            f"Train Loss: {epoch_train_loss:.4f}  "
            f"Val Loss: {epoch_val_loss:.4f}  "
            f"Val R²: {epoch_r2:.4f}  "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Check early stopping
        early_stopping(epoch_val_loss, model)
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


def evaluate_model(args, model, test_loader):
    """Evaluate the trained model on the test set."""
    logger.info("Evaluating model on test set...")
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch, mask_batch in test_loader:

            # Convert X_batch: List[T × (B, R, F)] → (B, T, R, F) → (B, F, T, R)
            x = torch.stack(X_batch, dim=1).permute(0, 3, 1, 2)

            # Forward pass
            outputs = model(x)
            
            if args.task_type == "consumption_forecast":
                preds = outputs.view(-1)                # (B, 1, 1, 1) or (B, 1) -> (B)
                targets = y_batch.view(-1).float()      # (B, 1) -> (B)
                mask = mask_batch.view(-1)              # (B, 1) -> (B)
            else: # args.task_type == "measurement_forecast"
                preds   = outputs                       # (B, n_pred, N) -> No change
                targets = y_batch.float()               # (B, n_pred, N) -> No change, just ensuring float type
                mask = mask_batch                       # (B, n_pred, N) -> No change

            # Collect only valid predictions and targets for R² score
            valid_preds = preds[mask_batch == 1]
            valid_targets = targets[mask_batch == 1]
            all_preds.extend(valid_preds.cpu().tolist())
            all_targets.extend(valid_targets.cpu().tolist())


    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)

    if all_targets_np.size == 0:
        logger.warning("No valid targets in test set to evaluate.")
        return {"test_loss": 0, "rmse": 0, "mae": 0, "r2": 0, "mape": 0, 
                "predictions": np.array([]), "targets": np.array([])}

    # MSE
    mse = mean_squared_error(all_targets_np, all_preds_np)
    # RMSE
    rmse = np.sqrt(mse)
    # MAE
    mae = mean_absolute_error(all_targets_np, all_preds_np)
    # R²
    r2 = r2_score(all_targets_np, all_preds_np)
    # MAPE
    mape_mask = all_targets_np != 0
    mape = np.mean(np.abs((all_targets_np[mape_mask] - all_preds_np[mape_mask]) / all_targets_np[mape_mask])) * 100 if np.sum(mape_mask) > 0 else 0.0

    logger.info(
        f"Evaluation Metrics (on valid data): MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MAPE: {mape:.2f}%"
    )

    return {
        "test_loss": mse, 
        "rmse": rmse, 
        "mae": mae, 
        "r2": r2, 
        "mape": mape,
        "predictions": all_preds_np, 
        "targets": all_targets_np,
    }

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
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        
    # Prepare data
    data = load_data(args)
    
    # Setup model
    model, optimizer, scheduler, early_stopping = setup_model(args, data)
    
    # Train model
    model, history = train_model(
        args, model, optimizer, scheduler, early_stopping,
        data['train_loader'], data['val_loader']
    )
    
    # Evaluate model
    metrics = evaluate_model(args, model, data['test_loader'])
    
    # Plot results
    plot_results(args, history, metrics)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'stgcn_forecasting_model.pt'))
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
