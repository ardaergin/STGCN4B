import os
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import roc_curve, auc, precision_recall_curve

logger = logging.getLogger(__name__)

class ResultHandler:
    """
    A class to handle plotting and saving of model training and evaluation results.
    It dispatches to the correct plotting methods based on the task type.
    """
    def __init__(self, output_dir: str, task_type: str, history: dict, metrics: dict):
        """Initializes the ResultPlotter with all necessary data."""
        self.output_dir = output_dir
        self.task_type = task_type
        self.history = history
        self.metrics = metrics

    def process(self):
        """
        Public method to generate and save all plots and metrics.
        This acts as a dispatcher to the appropriate private methods.
        """
        logger.info("Plotting and saving results...")
        self._setup_output_dir()

        if self.task_type in ["measurement_forecast", "consumption_forecast"]:
            self._plot_forecasting_main()
            self._plot_forecasting_timeseries()
            self._save_forecasting_metrics()
        elif self.task_type == "workhour_classification":
            self._plot_classification()
            self._save_classification_metrics()
        else:
            raise ValueError(f"Unknown task type for plotting: {self.task_type}")

        logger.info(f"Results saved to {self.output_dir}")

    def _setup_output_dir(self):
        """Creates the output directory if it doesn't exist."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def _plot_forecasting_main(self):
        """Generates the main 4-panel plot for forecasting results."""
        plt.figure(figsize=(15, 10))
        
        # Loss Curves
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss', color='blue')
        val_loss = self.history.get('val_loss')
        if val_loss is not None:
            plt.plot(val_loss, label='Validation Loss', color='red')
        plt.xlabel('Epoch'); plt.ylabel('Loss (MSE)'); plt.title('Loss Curves')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        
        # R² Score Curve
        plt.subplot(2, 2, 2)
        if self.history.get('val_metrics', {}).get('r2'):
            plt.plot(self.history['val_metrics']['r2'], label='Validation R²', color='green')
        plt.axhline(y=self.metrics['r2'], color='r', linestyle='--', label=f'Test R²: {self.metrics["r2"]:.4f}')
        plt.xlabel('Epoch'); plt.ylabel('R² Score'); plt.title('R² Score Curve')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        
        # Actual vs. Predicted
        plt.subplot(2, 2, 3)
        plt.scatter(self.metrics['targets'], self.metrics['predictions'], alpha=0.5)
        min_val = min(np.min(self.metrics['targets']), np.min(self.metrics['predictions']))
        max_val = max(np.max(self.metrics['targets']), np.max(self.metrics['predictions']))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel('Actual Values'); plt.ylabel('Predicted Values'); plt.title('Actual vs Predicted')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Error Distribution
        plt.subplot(2, 2, 4)
        errors = self.metrics['predictions'] - self.metrics['targets']
        plt.hist(errors, bins=20, alpha=0.7)
        plt.xlabel('Prediction Error'); plt.ylabel('Frequency'); plt.title(f'Error Distribution (RMSE: {self.metrics["rmse"]:.4f})')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'stgcn_forecasting_results.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_forecasting_timeseries(self):
        """Generates the time series comparison plot for forecasting."""
        plt.figure(figsize=(12, 6))
        sample_size = min(100, len(self.metrics['targets']))
        indices = np.arange(sample_size)
        plt.plot(indices, self.metrics['targets'][:sample_size], label='Actual', marker='o', markersize=4, alpha=0.7)
        plt.plot(indices, self.metrics['predictions'][:sample_size], label='Predicted', marker='x', markersize=4, alpha=0.7)
        plt.xlabel('Time Step'); plt.ylabel('Value'); plt.title('Predicted vs Actual (Sample)')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(self.output_dir, 'stgcn_forecasting_timeseries.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _save_forecasting_metrics(self):
        """Saves forecasting metrics to a text file."""
        with open(os.path.join(self.output_dir, 'stgcn_forecasting_metrics.txt'), 'w') as f:
            f.write(f"Test Loss (MSE): {self.metrics['test_loss']:.4f}\n")
            f.write(f"RMSE: {self.metrics['rmse']:.4f}\n")
            f.write(f"MAE: {self.metrics['mae']:.4f}\n")
            f.write(f"R²: {self.metrics['r2']:.4f}\n")
            f.write(f"MAPE: {self.metrics['mape']:.2f}%\n")

    def _plot_classification(self):
        """
        Generates a revised, more informative 2x2 plot for classification results.
        Includes Loss curve(s), Confusion Matrix, ROC Curve, PR Curve.
        """
        plt.figure(figsize=(14, 12))
        
        # --- 1. Loss curves (training, and valid if avaiable) ---
        plt.subplot(2, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss', color='blue')
        val_loss = self.history.get('val_loss')
        if val_loss is not None:
            plt.plot(val_loss, label='Validation Loss', color='red')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves')
        plt.legend(); plt.grid(True, linestyle='--', alpha=0.6)

        # --- 2. Confusion Matrix ---
        plt.subplot(2, 2, 2)
        conf_mat = self.metrics['confusion_matrix']
        labels = ['Non-Work', 'Work']
        im = plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45, ha="right")
        plt.yticks(tick_marks, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Calculate row-wise normalization (percentages of each true class)
        # Add a small epsilon (1e-6) to avoid division by zero for rows with no samples
        conf_mat_norm = conf_mat.astype('float') / (conf_mat.sum(axis=1)[:, np.newaxis] + 1e-6)

        # Loop over data dimensions and create text annotations.
        thresh = conf_mat.max() / 2.
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                count = conf_mat[i, j]
                percent = conf_mat_norm[i, j]
                # Format text string to include count and percentage
                text_label = f"{count:d}\n({percent:.1%})"
                plt.text(j, i, text_label,
                         ha="center", va="center",
                         color="white" if count > thresh else "black",
                         fontsize=11)

        # --- 3. ROC Curve ---
        plt.subplot(2, 2, 3)
        fpr, tpr, _ = roc_curve(self.metrics['labels'], self.metrics['probabilities'])
        roc_auc = self.metrics['roc_auc']
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)

        # --- 4. Precision-Recall Curve ---
        plt.subplot(2, 2, 4)
        precision, recall, _ = precision_recall_curve(self.metrics['labels'], self.metrics['probabilities'])
        avg_precision = self.metrics['auc_pr']
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.output_dir, 'stgcn_classification_results_revised.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def _save_classification_metrics(self):
        """Saves classification metrics to a text file."""
        baseline = max(sum(self.metrics['labels']), len(self.metrics['labels']) - sum(self.metrics['labels'])) / len(self.metrics['labels'])
        with open(os.path.join(self.output_dir, 'stgcn_classification_metrics.txt'), 'w') as f:
            f.write(f"Test Loss: {self.metrics['test_loss']:.4f}\n")
            f.write(f"Test Accuracy: {self.metrics['accuracy']:.4f}\n")
            f.write(f"Balanced Accuracy: {self.metrics['balanced_accuracy']:.4f}\n")
            f.write(f"Test Precision: {self.metrics['precision']:.4f}\n")
            f.write(f"Test Recall: {self.metrics['recall']:.4f}\n")
            f.write(f"Test F1-score: {self.metrics['f1']:.4f}\n")
            f.write(f"AUC-ROC: {self.metrics['roc_auc']:.4f}\n")
            f.write(f"AUC-PR (Average Precision): {self.metrics['auc_pr']:.4f}\n")
            f.write(f"Threshold: {self.metrics['threshold']:.4f}\n")
            f.write(f"Baseline Accuracy: {baseline:.4f}\n")
            f.write(f"Improvement over baseline: {(self.metrics['accuracy'] - baseline) / baseline * 100:.2f}%\n")