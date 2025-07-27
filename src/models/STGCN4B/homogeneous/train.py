import time
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import numpy as np
import optuna

from ....utils.early_stopping import EarlyStopping
from ....utils.tracking import TrainingHistory, TrainingResult
from ....utils.metrics import (regression_results, binary_classification_results, find_optimal_f1_threshold)
from .normalizer import STGCNNormalizer

import logging; logger = logging.getLogger(__name__)


def train_model(
        args,
        device: torch.device,
        model: nn.Module,
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        trial: optuna.trial.Trial = None, 
        epoch_offset: int = 0,
        ) -> Tuple[nn.Module, TrainingHistory, TrainingResult]:
    logger.info("Starting model training...")

    use_amp   = getattr(args, 'amp', False)
    amp_dtype = (torch.bfloat16 
                if getattr(args, 'amp_dtype', 'bf16') == 'bf16' 
                else torch.float16)
    scaler = torch.cuda.amp.GradScaler(
        enabled = use_amp and amp_dtype == torch.float16
    )
    logger.info(
        f"AMP: {use_amp} (dtype={amp_dtype}), "
        f"GradScaler enabled: {scaler.is_enabled()}, "
        f"TF32: {torch.backends.cuda.matmul.allow_tf32}"
    )
    
    # Instantiate criterion
    criterion = get_criterion(args, train_loader, device)
    
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
        epoch_start_time = time.time()

        # Training phase
        model.train()
        running_train_loss = 0.0
                
        for X_batch, y_batch, mask_batch, _ in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            # NOTE: we are not using the reconstruction_batch here, so left it as "_"
            
            # Zero the gradients
            optimizer.zero_grad()
                        
            # Forward pass (with Automatic Mixed Precision)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(X_batch)
                if args.task_type == "workhour_classification":
                    loss_train = criterion(outputs.squeeze(), y_batch.squeeze().float())
                else: # Forecasting tasks
                    loss_train = criterion(outputs, y_batch, mask_batch)

            # Running loss (calculated outside the autocast context)
            running_train_loss += loss_train.item()

            # Backward pass (with GradScaler)
            if scaler.is_enabled():
                scaler.scale(loss_train).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
            
            with torch.inference_mode():
                for X_batch, y_batch, mask_batch, _ in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    mask_batch = mask_batch.to(device, non_blocking=True)

                    # Forward pass
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        outputs = model(X_batch)
                        if args.task_type == "workhour_classification":
                            if outputs.dim() == 1:
                                outputs = outputs.unsqueeze(1)  # (B,1)
                            y_flat = y_batch.float().view(outputs.size(0), -1)  # (B,1) if binary
                            loss_val = criterion(outputs, y_flat)
                        else:
                            loss_val = criterion(outputs, y_batch, mask_batch)
                    
                    if args.task_type == "workhour_classification":
                        probs = torch.sigmoid(outputs).view(-1)
                        all_probs.extend(probs.cpu().tolist())
                        all_targets_class.extend(y_flat.view(-1).cpu().tolist())
                    else: # Forecasting tasks
                        preds = outputs
                        targets = y_batch.float()
                        mask = mask_batch
                        
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
            
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            ########## Pruning: extremely slow trials ##########
            if trial and epoch > 0 and epoch_duration > args.max_epoch_duration:
                # NOTE: Skipping first epoch due to model.compile()
                logger.warning(
                    f"Pruning trial {trial.number} for exceeding time limit. "
                    f"Epoch duration ({epoch_duration:.2f}s) > "
                    f"limit ({args.max_epoch_duration}s)."
                )
                # This stops the trial and tells Optuna it was pruned
                raise optuna.exceptions.TrialPruned()
            ########## End of Pruning: extremely slow trials ##########
            
            ########## Pruning: bad trials ##########
            if trial:
                optuna_metric_value = valid_metrics.get(optuna_metric, 0.0)
                trial.report(optuna_metric_value, epoch + epoch_offset)
                if trial.should_prune():
                    logger.info(f"Pruning trial {trial.number} at epoch {epoch} due to poor performance.")
                    raise optuna.exceptions.TrialPruned()
            ########## End of Pruning: bad trials ##########
            
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
            with torch.inference_mode():
                for X_batch, y_batch, *_ in val_loader:
                    X_batch = X_batch.to(device, non_blocking=True)
                    y_batch = y_batch.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        outputs = model(X_batch).squeeze()
                    probs.extend(torch.sigmoid(outputs).cpu().numpy())
                    labels.extend(y_batch.cpu().numpy())
            optimal_threshold = find_optimal_f1_threshold(labels, probs)
        
        training_result = TrainingResult(
            metric=best_score, 
            best_epoch=best_epoch, 
            optimal_threshold=optimal_threshold
        )
    
    return model, history, training_result



def evaluate_model(
        args,
        device: torch.device,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        normalizer: "STGCNNormalizer",
        *,
        threshold: float = 0.5,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Run the trained model on the test set.
    
    Returns:
    - overall metrics  (unchanged: rmse, mae, r2, …)
    - per-horizon metrics  in `per_horizon_metrics`
    - optionally also as a DataFrame in `per_horizon_df`
    """
    logger.info("Evaluating model on test set...")
    use_amp   = getattr(args, 'amp', False)
    amp_dtype = (torch.bfloat16 
                if getattr(args, 'amp_dtype', 'bf16') == 'bf16' 
                else torch.float16)
    logger.info(
        f"AMP: {use_amp} (dtype={amp_dtype}), "
        f"TF32: {torch.backends.cuda.matmul.allow_tf32}"
    )

    model.eval()
    
    ####################
    # Classification
    ####################
    
    if args.task_type == "workhour_classification":
        all_probs, all_labels = [], []
        with torch.inference_mode():
            for X_batch, y_batch, *_ in test_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                # Forward pass
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    outputs = model(X_batch)
                
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
    
    with torch.inference_mode():
        for X_batch, y_batch, mask_batch, y_source_batch in test_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            y_source_batch = y_source_batch.to(device, non_blocking=True)
            
            # Forward pass
            with torch.autocast(device_type=device.type, dtype=(torch.bfloat16 
                  if getattr(args,'amp_dtype','bf16')=='bf16' else torch.float16),
                  enabled=getattr(args,'amp',False)):
                preds_norm = model(X_batch)
            
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


# Criterion helpers
def get_criterion(args: Any, train_loader: torch.utils.data.DataLoader, device: torch.device) -> nn.Module:
    """Creates the loss function based on the task type.
    
    For binary classification tasks, calculate positive class weight for imbalanced data.
    """
    if args.task_type == "workhour_classification":
        # Calculate positive class weight for imbalanced data
        all_labels = []
        for _, labels, _, _ in train_loader:
            all_labels.extend(labels.cpu().numpy().flatten().tolist())
        n_pos = np.sum(all_labels)
        n_neg = len(all_labels) - n_pos
        pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, device=device)
        logger.info(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight.item():.2f}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        return MaskedMSELoss()

class MaskedMSELoss(nn.Module):
    def forward(self, preds, targets, mask):
        error = preds - targets
        masked_squared_error = (error ** 2) * mask
        num_valid_points = torch.sum(mask)
        if num_valid_points > 0:
            return torch.sum(masked_squared_error) / num_valid_points
        return torch.sum(preds * 0.0)