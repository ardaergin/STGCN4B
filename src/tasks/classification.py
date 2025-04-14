#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, precision_recall_curve
from torch.utils.data import DataLoader, TensorDataset
import scipy.sparse as sp

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import argument parser
from ..config import parse_args

# Import models
from ..models.stgcn import STGCNChebGraphConv, STGCNGraphConv, EarlyStopping

# Import other modules
from ..data import prepare_data

def setup_model(args, data, train_loader=None):
    """Set up the STGCN model and training components with class imbalance handling."""
    logger.info("Setting up model with class imbalance handling...")
    
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
        model = STGCNChebGraphConv(
            args, blocks, data["n_vertex"], data["gso"], 
            task_type='classification', num_classes=1
        ).to(device)
    else:
        model = STGCNGraphConv(
            args, blocks, data["n_vertex"], data["gso"], 
            task_type='classification', num_classes=1
        ).to(device)
    
    # Calculate class weights if train_loader is provided
    if train_loader is not None:
        # Extract all labels from training set
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.cpu().numpy())
        
        # Count class occurrences
        n_samples = len(all_labels)
        n_work_hours = sum(all_labels)  # Class 1 (work hours)
        n_non_work_hours = n_samples - n_work_hours  # Class 0 (non-work hours)
        
        # Calculate positive class weight (for work hours - minority class)
        # Higher weight means the model pays more attention to this class
        pos_weight = n_non_work_hours / n_work_hours if n_work_hours > 0 else 1.0
        
        logger.info(f"Class distribution in training set: Work hours={n_work_hours}, Non-work hours={n_non_work_hours}")
        logger.info(f"Using positive class weight: {pos_weight:.4f}")
        
        # Binary classification loss function with class weight
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    else:
        # Default loss without weighting
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


def find_optimal_threshold(model, val_loader):
    """Find the optimal classification threshold using validation data."""
    logger.info("Finding optimal threshold on validation set...")
    
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            # Forward pass
            outputs = model(X_batch).squeeze()
            
            # Store probabilities and labels
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    # Find optimal threshold on validation set
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)
    
    # Compute F1 scores for each threshold
    f1_scores = []
    for p, r in zip(precisions, recalls):
        if p + r == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * p * r / (p + r))
    
    # Find threshold that maximizes F1 score
    if len(thresholds) > 0:
        optimal_idx = np.argmax(f1_scores[:-1])  # Last element doesn't have a threshold
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5  # Default if no threshold found
    
    logger.info(f"Optimal classification threshold: {optimal_threshold:.4f}")
    return optimal_threshold


def evaluate_model(model, criterion, test_loader, threshold=0.5):
    """Evaluate the trained model on the test set with a pre-determined threshold."""
    logger.info(f"Evaluating model on test set using threshold: {threshold:.4f}...")
    
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
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    # Average test loss
    test_loss = test_loss / len(test_loader.dataset)
    
    # AUC scores
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
        ap_score = average_precision_score(all_labels, all_probs)
    except ValueError:
        # This can happen if there is only one class in y_true
        roc_auc = float('nan')
        ap_score = float('nan')

    logger.info(f"AUC-ROC: {roc_auc:.4f}")
    logger.info(f"AUC-PR (Average Precision): {ap_score:.4f}")

    # Apply the pre-determined threshold to get predictions
    all_preds = [1 if prob >= threshold else 0 for prob in all_probs]
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'auc_pr': ap_score,
        'confusion_matrix': conf_matrix,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'threshold': threshold
    }

    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
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
    
    # Calculate specificity (true negative rate)
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        logger.info(f"Specificity (True Negative Rate): {specificity:.4f}")
    
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
        f.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"Test Precision: {metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {metrics['recall']:.4f}\n")
        f.write(f"Test F1-score: {metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC: {metrics['roc_auc']:.4f}\n")
        f.write(f"AUC-PR (Average Precision): {metrics['auc_pr']:.4f}\n")
        f.write(f"Threshold: {metrics['threshold']:.4f}\n")
        f.write(f"Baseline Accuracy: {baseline:.4f}\n")
        f.write(f"Improvement over baseline: {(metrics['accuracy'] - baseline) / baseline * 100:.2f}%\n")
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Main function to run the STGCN model for OfficeGraph classification with class imbalance handling."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Prepare data
    data = prepare_data(args)
    
    # Setup model with class weighting
    model, criterion, optimizer, scheduler, early_stopping = setup_model(
        args, data, train_loader=data['train_loader']
    )
    
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
