#!/usr/bin/env python

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import dense_to_sparse
# Forecasting metrics
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
# Classificaation metrics
from sklearn.metrics import (confusion_matrix, accuracy_score, 
                             precision_score, recall_score, f1_score, precision_recall_curve,
                             roc_auc_score, average_precision_score, balanced_accuracy_score)
import optuna

import logging
logger = logging.getLogger(__name__)

from ....utils.graph_utils import calc_gso_edge
from ....utils.early_stopping import EarlyStopping
from .normalizer import STGCNNormalizer

def setup_model(args, data):
    """Set up the STGCN model and training components."""
    logger.info(f"Setting up model for the task type '{args.task_type}'...")
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

    # Build masked GSOs for information propagation
    elif args.gso_mode == "dynamic":
        masked_adjacency_matrices_dict = data.get("masked_adjacency_matrices", {})
        masked_adjacency_matrices = list(masked_adjacency_matrices_dict.values())
        masked_adjacency_matrices = masked_adjacency_matrices[: args.stblock_num]
        masked_gsos = []
        for adjacency_matrix in masked_adjacency_matrices:
            edge_index, edge_weight = dense_to_sparse(adjacency_matrix)
            G = calc_gso_edge(
                edge_index, edge_weight, 
                num_nodes           = n_nodes,
                gso_type            = args.gso_type,
                device              = device,
            )
            masked_gsos.append(G)

        # In case we've got fewer than stblock_num, pad with the static GSO
        while len(masked_gsos) < args.stblock_num:
            masked_gsos.append(static_gso)
        
        gso = masked_gsos
    
    else:
        raise ValueError(f"Unknown gso_mode: {args.gso_mode!r}. Must be 'static' or 'dynamic'.")
    
    batch_sample = next(iter(data['train_loader']))
    X_batch_sample, y_batch = batch_sample[0], batch_sample[1]
    logger.info(f"Sample batch input shape (X): {X_batch_sample.shape}  # shape=(batch_size, n_his, n_nodes, n_features)")
    logger.info(f"Sample target shape: {y_batch.shape}")
    
    blocks = []
    blocks.append([n_features])  # Input features
    logger.info(f"Model first block input dimension: {blocks[0][0]}")

    # Add intermediate blocks
    for _ in range(args.stblock_num):
        intermediate_channels = [args.st_main_channels, args.st_bottleneck_channels, args.st_main_channels]
        blocks.append(intermediate_channels)
        
    # Add output blocks
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num
    if Ko == 0:
        blocks.append([args.output_channels])
    elif Ko > 0:
        blocks.append([args.output_channels, args.output_channels])
    
    # Output dimension determined by n_pred
    n_pred = len(args.forecast_horizons)
    blocks.append([n_pred])
    
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
        for _, labels, _, _ in data['train_loader']:
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
        n_samples = len(all_labels)
        n_work_hours = sum(all_labels)
        n_non_work_hours = n_samples - n_work_hours
        pos_weight = n_non_work_hours / n_work_hours if n_work_hours > 0 else 1.0
        logger.info(f"Class distribution: Work hours={n_work_hours}, Non-work hours={n_non_work_hours}")
        logger.info(f"Using positive class weight: {pos_weight:.4f}")
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    else:
        criterion = MaskedMSELoss()
    
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


def train_model(
        args, 
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        early_stopping: EarlyStopping,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        trial: optuna.trial.Trial = None, 
        epoch_offset: int = 0
        ):
    """Train the STGCN model for forecasting."""
    logger.info("Starting model training...")
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': {}
    }

    for epoch in range(args.epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        total_valid_points_train = 0
        
        for X_batch, y_batch, mask_batch, _ in train_loader:
            # NOTE: we are not using the reconstruction_batch here, so left it as "_"
            # Zero the gradients
            optimizer.zero_grad()

            # Convert X_batch: List[T × (B, R, F)] → (B, T, R, F) → (B, F, T, R)
            x = X_batch.permute(0, 3, 1, 2)

            # Forward pass
            outputs = model(x)
            
            if args.task_type == "workhour_classification":
                loss_train = criterion(outputs.squeeze(), y_batch.squeeze().float())
                running_train_loss += loss_train.item()
            else: # Forecasting tasks
                preds = outputs
                targets = y_batch.float()
                mask = mask_batch
                loss_train = criterion(preds, targets, mask)

                # Update running train loss: accumulating the per-batch MSE into running_train_loss
                running_train_loss += loss_train.item()
            
            # Backward pass
            loss_train.backward()
            optimizer.step()
                            
        # Average training loss for the epoch
        epoch_train_loss = running_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        history['train_loss'].append(epoch_train_loss)
                
        # Validation phase (Optional)
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
            total_valid_points_val = 0
            # Forecasting:
            all_preds, all_targets = [], []
            # Classification:
            all_probs, all_targets_class = [], [] 
            
            with torch.no_grad():
                for X_batch, y_batch, mask_batch, _ in val_loader:

                    # Convert X_batch: List[T × (B, R, F)] → (B, T, R, F) → (B, F, T, R)
                    x = X_batch.permute(0, 3, 1, 2)

                    # Forward pass
                    outputs = model(x)

                    if args.task_type == "workhour_classification":
                        loss_val = criterion(outputs.squeeze(), y_batch.squeeze().float())
                        probs = torch.sigmoid(outputs.squeeze())
                        all_probs.extend(probs.cpu().tolist())
                        all_targets_class.extend(y_batch.squeeze().cpu().tolist())
                    else: # Forecasting tasks
                        preds = outputs
                        targets = y_batch.float()
                        mask = mask_batch
                        loss_val = criterion(preds, targets, mask)
                        
                        # Collect only valid predictions and targets for R² score
                        valid_preds = preds[mask == 1]
                        valid_targets = targets[mask == 1]
                        all_preds.extend(valid_preds.cpu().tolist())
                        all_targets.extend(valid_targets.cpu().tolist())
                                            
                    running_val_loss += loss_val.item()
            
            # Average validation loss (MSE) for the epoch
            epoch_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            history['val_loss'].append(epoch_val_loss)
            
            # Append epoch metrics to the history dictionary
            epoch_metrics = {}
            if args.task_type == "workhour_classification":
                if len(all_targets_class) > 0:
                    # Check if more than one class is present to calculate AUC
                    if len(np.unique(all_targets_class)) > 1:
                        epoch_metrics['val_auc'] = roc_auc_score(all_targets_class, all_probs)
                    else:
                        epoch_metrics['val_auc'] = 0.5 # Neutral score if only one class is present

                    # For logging, we can still compute metrics at the 0.5 threshold
                    preds_at_half = [1 if p >= 0.5 else 0 for p in all_probs]
                    epoch_metrics['accuracy'] = accuracy_score(all_targets_class, preds_at_half)
                    epoch_metrics['f1'] = f1_score(all_targets_class, preds_at_half, zero_division=0)
            else: # Forecasting
                if len(all_targets) > 0:
                    epoch_metrics['r2'] = r2_score(all_targets, all_preds)
                    epoch_metrics['mae'] = mean_absolute_error(all_targets, all_preds)
            
            for metric_name, metric_val in epoch_metrics.items():
                history['val_metrics'].setdefault(metric_name, []).append(metric_val)
            
            # --- PRUNING LOGIC ---
            if trial:
                # For classification, we prune based on the primary metric (e.g., F1-score)
                if args.task_type == "workhour_classification":
                    metric_to_report = epoch_metrics.get('val_auc', 0.0)
                # For forecasting, we prune based on validation loss
                else:
                    metric_to_report = epoch_val_loss
                
                # Report the intermediate metric to Optuna
                trial.report(metric_to_report, epoch + epoch_offset)
                
                # Check if the trial should be pruned based on the pruner's decision
                if trial.should_prune():
                    logger.info(f"Pruning trial {trial.number} at epoch {epoch} due to poor performance.")
                    raise optuna.exceptions.TrialPruned()
            # --- END OF PRUNING LOGIC ---
            
            # Logging
            metrics_log_str = " | ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics.items()])
            logger.info(
                f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | {metrics_log_str} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            
            # Check early stopping
            if args.task_type == "workhour_classification":
                # We want to MAXIMIZE AUC, so we MINIMIZE (-AUC) for the early stopping class
                current_auc = epoch_metrics.get('val_auc', 0.0)
                early_stopping(-current_auc, model, epoch + 1)
            else:
                early_stopping(epoch_val_loss, model, epoch + 1)
            
            if early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break
        
        else: # No validation
            logger.info(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {epoch_train_loss:.4f}")
        
        # Update learning rate
        scheduler.step()
    
    # Load the best model
    if val_loader is not None and early_stopping.early_stop:
        logger.info(f"Loading best model from epoch {early_stopping.best_epoch}")
        model.load_state_dict(early_stopping.best_model_state)
        
        # Add best AUC to history for the objective function
        if args.task_type == "workhour_classification":
            best_epoch_idx = early_stopping.best_epoch - 1
            best_auc = history['val_metrics']['val_auc'][best_epoch_idx]
            history['best_val_auc'] = best_auc
    
    return model, history

def evaluate_model(
        args,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        processor: "STGCNNormalizer",
        *,
        threshold: float = 0.5,
        ):
    """
    Run the trained model on the test set.
    
    Returns:
    - overall metrics  (unchanged: rmse, mae, r2, …)
    - per-horizon metrics  in `per_horizon_metrics`
    - optionally also as a DataFrame in `per_horizon_df`
    """
    logger.info("Evaluating model on test set...")
    model.eval()
    
    ####################
    # Classification
    ####################
    
    if args.task_type == "workhour_classification":
        all_probs, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch, *_ in test_loader:
                # Convert X_batch: List[T × (B, R, F)] → (B, T, R, F) → (B, F, T, R)
                x = X_batch.permute(0, 3, 1, 2)
                # Forward pass
                outputs = model(x)
                preds = torch.sigmoid(outputs.squeeze())
                # Add probabilities and labels to lists
                all_probs.extend(preds.cpu().tolist())
                all_labels.extend(y_batch.squeeze().cpu().tolist())
        
        y_hat = np.array(all_probs)
        y_true = np.array(all_labels)
        y_pred = (y_hat >= threshold).astype(int)
        
        return {
            "test_loss":           0, # Not applicable for classification
            "accuracy":            accuracy_score(y_true, y_pred),
            "balanced_accuracy":   balanced_accuracy_score(y_true, y_pred),
            "precision":           precision_score(y_true, y_pred, zero_division=0),
            "recall":              recall_score(y_true, y_pred, zero_division=0),
            "f1":                  f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":             roc_auc_score(y_true, y_hat),
            "auc_pr":              average_precision_score(y_true, y_hat),
            "confusion_matrix":    confusion_matrix(y_true, y_pred),
            "threshold":           threshold,
            "probabilities":       y_hat,
            "predictions":         y_pred,
            "labels":              y_true,
        }
    
    ####################
    # Forecasting
    ####################
    
    # Horizon-related variables
    H = len(args.forecast_horizons)
    preds_norm_per_h = [[] for _ in range(H)]
    targets_norm_per_h = [[] for _ in range(H)]
    y_source_per_h = [[] for _ in range(H)] if args.prediction_type == "delta" else None
    
    with torch.no_grad():
        for X_batch, y_batch, mask_batch, y_source_batch in test_loader:
            # Convert X_batch: List[T × (B, R, F)] → (B, T, R, F) → (B, F, T, R)
            x = X_batch.permute(0, 3, 1, 2)
            # Forward pass
            preds_norm = model(x)
            # Get targets & masks
            targets_norm = y_batch.float()
            mask = mask_batch
            
            for h in range(H):
                valid = mask[:, h].bool()
                pred = preds_norm[:, h][valid]
                target = targets_norm[:, h][valid]
                
                preds_norm_per_h[h].extend(pred.reshape(-1).cpu().tolist())
                targets_norm_per_h[h].extend(target.reshape(-1).cpu().tolist())
                
                if args.prediction_type == "delta":
                    # The reconstruction batches has the same shape as y_batch and mask_batch
                    # We need to filter it with the same mask to keep them aligned
                    y_source = y_source_batch[valid]
                    y_source_per_h[h].extend(y_source.reshape(-1).cpu().tolist())
    
    # Calculate per-horizon metrics on original scale
    per_horizon_metrics = {}
    for h, horizon in enumerate(args.forecast_horizons):
        if not preds_norm_per_h[h]:          # no valid data → skip
            logger.warning(f"No valid targets for horizon {horizon}.")
            continue
        
        # Converting the lists of normalized values to NumPy arrays
        p_norm = np.array(preds_norm_per_h[h])
        t_norm = np.array(targets_norm_per_h[h])
        
        # Inverse-transform the normalized predictions and targets.
        # NOTE: - For 'delta', these are now deltas in their original scale.
        #       - For 'absolute', these are the final predictions.
        p = processor.inverse_transform_target(p_norm)
        t = processor.inverse_transform_target(t_norm)
        
        # Delta -> Absolute reconstruction
        if args.prediction_type == "delta":
            src = np.array(y_source_per_h[h])
            p   = src + p
            t   = src + t
        
        mse  = mean_squared_error(t, p)
        rmse = np.sqrt(mse)
        mae  = mean_absolute_error(t, p)
        r2   = r2_score(t, p)
        
        mape_mask = t != 0
        mape = (np.mean(np.abs((t[mape_mask] - p[mape_mask]) / t[mape_mask])) * 100
                if mape_mask.any() else 0.0)
        
        per_horizon_metrics[f"h_{horizon}"] = {
            "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape
        }
        logger.info(f"Horizon {horizon:>3}:  RMSE={rmse:.4f} | MAE={mae:.4f} | "
                    f"R²={r2:.4f} | MAPE={mape:.2f}%")
    
    if not per_horizon_metrics: # Nothing to evaluate
        logger.warning("No valid targets in the entire test set.")
        return {"test_loss": 0, "rmse": 0, "mae": 0, "r2": 0, "mape": 0, 
                "predictions": np.array([]), "targets": np.array([])}
    
    # Aggregate overall metrics (flattening all horizons)
    all_p = np.concatenate([
        processor.inverse_transform_target(np.array(preds_norm_per_h[h]))
        if args.prediction_type == "absolute"
        else np.array(y_source_per_h[h]) +            # recon for delta
             processor.inverse_transform_target(np.array(preds_norm_per_h[h]))
        for h in range(H) if preds_norm_per_h[h]
    ])
    all_t = np.concatenate([
        processor.inverse_transform_target(np.array(targets_norm_per_h[h]))
        if args.prediction_type == "absolute"
        else np.array(y_source_per_h[h]) +
             processor.inverse_transform_target(np.array(targets_norm_per_h[h]))
        for h in range(H) if targets_norm_per_h[h]
    ])
    
    overall_mse  = mean_squared_error(all_t, all_p)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae  = mean_absolute_error(all_t, all_p)
    overall_r2   = r2_score(all_t, all_p)
    mape_mask    = all_t != 0
    overall_mape = (np.mean(np.abs((all_t[mape_mask] - all_p[mape_mask]) / all_t[mape_mask])) * 100
                    if mape_mask.any() else 0.0)
    logger.info(f"Overall: RMSE={overall_rmse:.4f} | MAE={overall_mae:.4f} | "
                f"R²={overall_r2:.4f} | MAPE={overall_mape:.2f}%")
    
    return {
        # Overall metrics
        "test_loss": overall_mse,
        "rmse":      overall_rmse,
        "mae":       overall_mae,
        "r2":        overall_r2,
        "mape":      overall_mape,
        # Per horizon metrics
        "per_horizon_metrics": per_horizon_metrics,
    }


# Helpers
class MaskedMSELoss(nn.Module):
    def forward(self, preds, targets, mask):
        error = preds - targets
        masked_squared_error = (error ** 2) * mask
        num_valid_points = torch.sum(mask)
        if num_valid_points > 0:
            return torch.sum(masked_squared_error) / num_valid_points
        return torch.tensor(0.0, device=preds.device)

def find_optimal_threshold(model, val_loader):
    """Find the optimal classification threshold using validation data."""
    logger.info("Finding optimal threshold on validation set...")
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch, _, _ in val_loader:
            outputs = model(X_batch.permute(0, 3, 1, 2)).squeeze()
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
    optimal_idx = np.argmax(f1_scores[:-1]) if len(thresholds) > 0 else -1
    optimal_threshold = thresholds[optimal_idx] if optimal_idx != -1 else 0.5
    logger.info(f"Optimal classification threshold: {optimal_threshold:.4f}")
    return optimal_threshold