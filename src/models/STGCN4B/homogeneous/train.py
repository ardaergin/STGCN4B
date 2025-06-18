#!/usr/bin/env python

import os
import sys
import logging
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
# Forecasting
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
# Classificaation
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, precision_recall_curve,
                             roc_auc_score, average_precision_score, balanced_accuracy_score)
from ....utils.train_utils import ResultHandler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

from ....config.args import parse_args
from ....data.Loader.graph_loader import load_data
from ....utils.graph_utils import calc_gso_edge
from ....utils.early_stopping import EarlyStopping

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
    logger.info(f"edge_index shape: {edge_index.shape}")
    logger.info(f"edge_weight shape: {edge_weight.shape}")

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
        from .models import STGCNChebGraphConv as Model
    else:
        from .models import STGCNGraphConv as Model
    
    model = Model(
        args     = args,
        blocks   = blocks,
        n_vertex = n_nodes,
        gso      = gso,
        task_type= args.task_type,
    ).to(device)
    
    # Loss function based on task type
    criterion = None
    if args.task_type == "workhour_classification":
        all_labels = []
        for _, labels, _ in data['train_loader']:
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
        n_samples = len(all_labels)
        n_work_hours = sum(all_labels)
        n_non_work_hours = n_samples - n_work_hours
        pos_weight = n_non_work_hours / n_work_hours if n_work_hours > 0 else 1.0
        logger.info(f"Class distribution: Work hours={n_work_hours}, Non-work hours={n_non_work_hours}")
        logger.info(f"Using positive class weight: {pos_weight:.4f}")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    else:
        criterion = None # For forecasting, the loss is calculated manually later
    
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


def train_model(args, model, criterion, optimizer, scheduler, early_stopping, train_loader, val_loader = None):
    """Train the STGCN model for forecasting."""
    logger.info("Starting model training...")
    
    train_losses = []
    # (Optional)
    val_losses = []
    val_metrics = []  # For R2 score or accuracy
    
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
            
            if args.task_type == "workhour_classification":
                loss = criterion(outputs.squeeze(), y_batch.squeeze().float())
                running_loss += loss.item() * x.size(0)
                total_valid_points_train += x.size(0)
            else: # Forecasting tasks
                preds, targets, mask = get_preds_targets_mask(outputs, y_batch, mask_batch, args.task_type)
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
                
        # Validation phase (Optional)
        if val_loader is not None:
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

                    if args.task_type == "workhour_classification":
                        loss_val = criterion(outputs.squeeze(), y_batch.squeeze().float())
                        running_val_loss += loss_val.item() * x.size(0)
                        total_valid_points_val += x.size(0)
                        preds = (torch.sigmoid(outputs.squeeze()) > 0.5).int()
                        all_preds.extend(preds.cpu().tolist())
                        all_targets.extend(y_batch.cpu().tolist())
                    else: # Forecasting tasks
                        preds, targets, mask = get_preds_targets_mask(outputs, y_batch, mask_batch, args.task_type)
                        # Manually calculate squared error and apply mask
                        error = preds - targets
                        masked_squared_error = (error ** 2) * mask
                        loss_val = torch.sum(masked_squared_error)
                        # Update running loss and count of valid points
                        running_val_loss += loss_val.item()
                        total_valid_points_val += torch.sum(mask).item()
                        # Collect only valid predictions and targets for R² score
                        valid_preds = preds[mask == 1]
                        valid_targets = targets[mask == 1]
                        all_preds.extend(valid_preds.cpu().tolist())
                        all_targets.extend(valid_targets.cpu().tolist())

            # Average validation loss (MSE) for the epoch
            epoch_val_loss = running_val_loss / total_valid_points_val if total_valid_points_val > 0 else 0.0
            val_losses.append(epoch_val_loss)
            
            if args.task_type == "workhour_classification":
                epoch_metric = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
                val_metrics.append(epoch_metric)
                metric_name, metric_val = "Val Accuracy", f"{epoch_metric:.4f}"
            else: # Forecasting tasks
                epoch_metric = r2_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
                val_metrics.append(epoch_metric)
                metric_name, metric_val = "Val R²", f"{epoch_metric:.4f}"

        else: # No validation
             pass
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch results
        if val_loader is not None:
            logger.info(
                f"Epoch [{epoch+1}/{args.epochs}]  "
                f"Train Loss: {epoch_train_loss:.4f}  "
                f"Val Loss: {epoch_val_loss:.4f}  "
                f"{metric_name}: {metric_val}  "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
        else:
            logger.info(
                f"Epoch [{epoch+1}/{args.epochs}]  "
                f"Train Loss: {epoch_train_loss:.4f}  "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        # Check early stopping
        if val_loader is not None:
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
        
    # Load the best model
    if val_loader is not None:
        model.load_state_dict(early_stopping.best_model_state)
    
    # Return training history
    if val_loader is not None:
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
        }
        if args.task_type == "workhour_classification":
            history['val_accuracy'] = val_metrics
        else:
            history['val_r2'] = val_metrics
    else:
        history = {
            'train_loss': train_losses,
        }
    
    return model, history


def evaluate_model(args, model, test_loader, threshold=0.5):
    """Evaluate the trained model on the test set."""
    logger.info("Evaluating model on test set...")
    
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, y_batch, mask_batch in test_loader:

            # Convert X_batch: List[T × (B, R, F)] → (B, T, R, F) → (B, F, T, R)
            x = torch.stack(X_batch, dim=1).permute(0, 3, 1, 2)

            # Forward pass
            outputs = model(x)
            
            if args.task_type == "workhour_classification":
                probs = torch.sigmoid(outputs.squeeze())
                all_probs.extend(probs.cpu().tolist())
                all_targets.extend(y_batch.squeeze().cpu().tolist())
            else:
                preds, targets, mask = get_preds_targets_mask(outputs, y_batch, mask_batch, args.task_type)
                # Collect only valid predictions and targets for R² score
                valid_preds = preds[mask == 1]
                valid_targets = targets[mask == 1]
                all_preds.extend(valid_preds.cpu().tolist())
                all_targets.extend(valid_targets.cpu().tolist())

    if args.task_type == "workhour_classification":
        all_preds = [1 if prob >= threshold else 0 for prob in all_probs]
        roc_auc = roc_auc_score(all_targets, all_probs)
        ap_score = average_precision_score(all_targets, all_probs)
        return {
            "test_loss": 0, # Loss not easily comparable, can be calculated if needed
            "accuracy": accuracy_score(all_targets, all_preds),
            "balanced_accuracy": balanced_accuracy_score(all_targets, all_preds),
            "precision": precision_score(all_targets, all_preds, zero_division=0),
            "recall": recall_score(all_targets, all_preds, zero_division=0),
            "f1": f1_score(all_targets, all_preds, zero_division=0),
            "roc_auc": roc_auc,
            "auc_pr": ap_score,
            "confusion_matrix": confusion_matrix(all_targets, all_preds),
            "predictions": all_preds,
            "labels": all_targets,
            "probabilities": all_probs,
            "threshold": threshold
        }
    else: # Forecasting tasks
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


# Helpers
def get_preds_targets_mask(outputs, y_batch, mask_batch, task_type):
    """Helper to get predictions, targets, and masks based on task type."""
    if task_type == "consumption_forecast":
        preds = outputs.view(-1)                # (B, 1, 1, 1) or (B, 1) -> (B)
        targets = y_batch.view(-1).float()      # (B, 1) -> (B)
        mask = mask_batch.view(-1)              # (B, 1) -> (B)
    else: # args.task_type == "measurement_forecast"
        preds   = outputs                       # (B, n_pred, N) -> No change
        targets = y_batch.float()               # (B, n_pred, N) -> No change, just ensuring float type
        mask = mask_batch                       # (B, n_pred, N) -> No change
    return preds, targets, mask

def find_optimal_threshold(model, val_loader):
    """Find the optimal classification threshold using validation data."""
    logger.info("Finding optimal threshold on validation set...")
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch, _ in val_loader:
            outputs = model(torch.stack(X_batch, dim=1).permute(0, 3, 1, 2)).squeeze()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
    optimal_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else -1
    optimal_threshold = thresholds[optimal_idx] if optimal_idx != -1 else 0.5
    logger.info(f"Optimal classification threshold: {optimal_threshold:.4f}")
    return optimal_threshold

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
    model, criterion, optimizer, scheduler, early_stopping = setup_model(args, data)
    
    # Train model
    model, history = train_model(
        args, model, criterion, optimizer, scheduler, early_stopping,
        data['train_loader'], data['val_loader']
    )
    
    # Evaluate model
    if args.task_type == "workhour_classification":
        optimal_threshold = find_optimal_threshold(model, data['val_loader'])
        metrics = evaluate_model(args, model, data['test_loader'], threshold=optimal_threshold)
    else:
        metrics = evaluate_model(args, model, data['test_loader'])
    
    # Plot results
    plotter = ResultHandler(args, history, metrics)
    plotter.process()

    # Save model
    if args.task_type == "measurement_forecast":
        fname = f"stgcn_model_{args.adjacency_type}_{args.interval}_{args.graph_type}_{args.measurement_type}.pt"
    else:
        fname = f"torch_input_{args.adjacency_type}_{args.interval}_{args.graph_type}.pt"

    torch.save(model.state_dict(), os.path.join(args.output_dir, fname))
    
    logger.info("Done!")


if __name__ == "__main__":
    main()