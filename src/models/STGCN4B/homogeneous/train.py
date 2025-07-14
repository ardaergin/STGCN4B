from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.utils import dense_to_sparse
import optuna

from ....utils.graph_utils import calc_gso_edge
from ....utils.early_stopping import EarlyStopping
from ....utils.tracking import TrainingHistory, TrainingResult
from ....utils.metrics import (regression_results, binary_classification_results, find_optimal_f1_threshold)
from .normalizer import STGCNNormalizer
from .models import HomogeneousSTGCN

import logging; logger = logging.getLogger(__name__)


def setup_model(
        args, data
    ) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
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
    else:
        raise ValueError(f"Problematic Ko={Ko}. Try a different combination of n_his, stblock_num, and Kt.")
    
    # Output dimension determined by n_pred
    n_pred = len(args.forecast_horizons)
    blocks.append([n_pred])
    
    # Create model based on graph convolution type
    model = HomogeneousSTGCN(
        args        = args,
        blocks      = blocks,
        n_vertex    = n_nodes,
        gso         = gso,
        conv_type   = args.graph_conv_type,
        task_type   = args.task_type,
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
        
    logger.info(f"Model setup complete with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, criterion, optimizer, scheduler



def train_model(
        args, 
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        trial: optuna.trial.Trial = None, 
        epoch_offset: int = 0
        ) -> Tuple[nn.Module, TrainingHistory, TrainingResult]:
    logger.info("Starting model training...")
    
    # Instantiate training history
    train_metric        = "logloss" if args.task_type == "workhour_classification" else "mse"
    train_objective     = "minimize"
    optuna_metric, optuna_objective = None, None
    if trial:
        optuna_metric       = "auc" if args.task_type == "workhour_classification" else "mse"
        optuna_objective    = "maximize" if args.task_type == "workhour_classification" else "minimize"
    history = TrainingHistory(
        train_metric    = train_metric, 
        train_objective = train_objective, 
        optuna_metric   = optuna_metric,
        optuna_objective = optuna_objective
    )
    
    # Early stopping
    if val_loader is not None:
        early_stopping = EarlyStopping(
            direction   = optuna_objective if trial else train_objective,
            patience    = args.es_patience, 
            delta       = args.es_delta,
            verbose     = True)
    
    # Training loop
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
        
        # Log training loss
        train_metrics = {train_metric: epoch_train_loss}
        history.log_epoch("train", **train_metrics)
          
        # Validation phase (Optional)
        if val_loader is not None:
            model.eval()
            running_val_loss = 0.0
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
            
            # Average validation loss for the epoch
            epoch_val_loss = running_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
            
            ########## LOGGING VALIDATION METRICS ##########
            valid_metrics = {train_metric: epoch_val_loss}
            
            if args.task_type == "workhour_classification":
                cls_results = binary_classification_results(all_targets_class, all_probs, threshold=.5)
                valid_metrics.update({
                    "auc":   cls_results["roc_auc"],
                    "accuracy": cls_results["accuracy"],
                    "f1":   cls_results["f1"],
                })
            else: # Forecasting
                reg_results = regression_results(np.array(all_targets), np.array(all_preds))
                valid_metrics.update({
                    "r2": reg_results["r2"], 
                    "mae": reg_results["mae"]
                })
            
            history.log_epoch("valid", **valid_metrics)
            ########## END OF LOGGING VALIDATION METRICS ##########
            
            ########## PRUNING LOGIC ##########
            if trial:
                optuna_metric_value = valid_metrics.get(optuna_metric, 0.0)
                trial.report(optuna_metric_value, epoch + epoch_offset)
                if trial.should_prune():
                    logger.info(f"Pruning trial {trial.number} at epoch {epoch} due to poor performance.")
                    raise optuna.exceptions.TrialPruned()
            ########## END OF PRUNING LOGIC ##########
            
            # Logging
            metrics_log_str = " | ".join([f"{k}: {v:.4f}" for k, v in valid_metrics.items()])
            logger.info(
                f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {epoch_train_loss:.4f} | "
                f"Val Loss: {epoch_val_loss:.4f} | {metrics_log_str} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )
            
            # Check early stopping
            metric_for_stopping = optuna_metric_value if trial else epoch_val_loss
            early_stopping(metric_for_stopping, model, epoch)
                        
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
    
    # Filling the TrainingResult
    training_result = None
    if val_loader is not None:
        best_score = history.get_best_valid_score()
        best_epoch = early_stopping.best_epoch
        
        optimal_threshold = None
        if args.task_type == "workhour_classification":
            model.eval()
            probs, labels = [], []
            with torch.no_grad():
                for X, y, *_ in val_loader:
                    outputs = model(X.permute(0, 3, 1, 2)).squeeze()
                    probs.extend(torch.sigmoid(outputs).cpu().numpy())
                    labels.extend(y.cpu().numpy())
            optimal_threshold = find_optimal_f1_threshold(labels, probs)
        
        training_result = TrainingResult(
            metric=best_score, 
            best_epoch=best_epoch, 
            optimal_threshold=optimal_threshold
        )
    
    return model, history, training_result



def evaluate_model(
        args,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        normalizer: "STGCNNormalizer",
        *,
        threshold: float = 0.5,
        ) -> Dict[str, Any]:
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
        
        metrics = binary_classification_results(all_labels, all_probs, threshold)
        model_outputs = {
            "probabilities": np.array(all_probs),
            "predictions": (np.array(all_probs) >= threshold).astype(int),
            "labels": np.array(all_labels),
        }
        return metrics, model_outputs
    
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
                    y_source = y_source_batch.squeeze(1)[valid]
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
        p = normalizer.inverse_transform_target(p_norm)
        t = normalizer.inverse_transform_target(t_norm)
        
        # Delta -> Absolute reconstruction
        if args.prediction_type == "delta":
            src = np.array(y_source_per_h[h])
            p   = src + p
            t   = src + t
        
        h_reg_results = regression_results(t, p)

        per_horizon_metrics[f"h_{horizon}"] = {
            "mse": h_reg_results["mse"], 
            "rmse": h_reg_results["rmse"], 
            "mae": h_reg_results["mae"], 
            "r2": h_reg_results["r2"], 
            "mape": h_reg_results["mape"]
        }
        logger.info(
            f"Horizon {horizon:>3}: "
            f"MSE={h_reg_results['mse']:.4f} | "
            f"RMSE={h_reg_results['rmse']:.4f} | "
            f"MAE={h_reg_results['mae']:.4f} | "
            f"R²={h_reg_results['r2']:.4f} | "
            f"MAPE={h_reg_results['mape']:.2f}%"
        )
    
    if not per_horizon_metrics: # Nothing to evaluate
        logger.warning("No valid targets in the entire test set.")
        return {"mse": 0, "rmse": 0, "mae": 0, "r2": 0, "mape": 0, 
                "predictions": np.array([]), "targets": np.array([])}
    
    # Aggregate overall metrics (flattening all horizons)
    if args.prediction_type == "absolute":
        all_p = np.concatenate([
            normalizer.inverse_transform_target(np.array(preds_norm_per_h[h]))
            for h in range(H) if preds_norm_per_h[h]
        ])
        all_t = np.concatenate([
            normalizer.inverse_transform_target(np.array(targets_norm_per_h[h]))
            for h in range(H) if targets_norm_per_h[h]
        ])
    else:  # delta → reconstruct absolute values
        all_p = np.concatenate([
            np.array(y_source_per_h[h]) +
            normalizer.inverse_transform_target(np.array(preds_norm_per_h[h]))
            for h in range(H) if preds_norm_per_h[h]
        ])
        all_t = np.concatenate([
            np.array(y_source_per_h[h]) +
            normalizer.inverse_transform_target(np.array(targets_norm_per_h[h]))
            for h in range(H) if targets_norm_per_h[h]
        ])

    reg_results = regression_results(all_t, all_p)
    
    logger.info(f"Overall: "
                f"MSE={reg_results['mse']:.4f} | "
                f"RMSE={reg_results['rmse']:.4f} | "
                f"MAE={reg_results['mae']:.4f} | "
                f"R²={reg_results['r2']:.4f} | "
                f"MAPE={reg_results['mape']:.2f}%")
    
    metrics = {**reg_results, "per_horizon_metrics": per_horizon_metrics}
    
    model_output = {
        "predictions": all_p,
        "targets": all_t,
    }
    return metrics, model_output


# Helpers
class MaskedMSELoss(nn.Module):
    def forward(self, preds, targets, mask):
        error = preds - targets
        masked_squared_error = (error ** 2) * mask
        num_valid_points = torch.sum(mask)
        if num_valid_points > 0:
            return torch.sum(masked_squared_error) / num_valid_points
        return torch.tensor(0.0, device=preds.device)