from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                         confusion_matrix, roc_curve, auc, precision_recall_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List
import json
import pandas as pd

class ModelEvaluator:
    def __init__(self):
        """Initialize ModelEvaluator class"""
        self.metrics = {}
        self.confusion_matrix = None
        self.roc_curve_data = None
        self.pr_curve_data = None
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict[str, float]:
        """
        Compute classification metrics
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_prob (np.ndarray): Predicted probabilities for positive class
        Returns:
            Dict[str, float]: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred)
        }
        
        # Compute confusion matrix
        self.confusion_matrix = confusion_matrix(y_true, y_pred)
        
        # Compute ROC and PR curves if probabilities are provided
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            self.roc_curve_data = {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}
            metrics['auc_roc'] = self.roc_curve_data['auc']
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            self.pr_curve_data = {'precision': precision, 'recall': recall}
        
        self.metrics = metrics
        return metrics
    
    def plot_metrics(self, output_dir: str):
        """
        Plot metrics visualization
        Args:
            output_dir (str): Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        
        # Plot ROC curve if available
        if self.roc_curve_data is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(self.roc_curve_data['fpr'], self.roc_curve_data['tpr'], 
                    label=f"AUC = {self.roc_curve_data['auc']:.4f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
            plt.close()
        
        # Plot PR curve if available
        if self.pr_curve_data is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(self.pr_curve_data['recall'], self.pr_curve_data['precision'])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
            plt.close()
        
        # Bar plot of metrics
        plt.figure(figsize=(10, 6))
        metrics_to_plot = {k: v for k, v in self.metrics.items() if k != 'auc_roc'}
        sns.barplot(x=list(metrics_to_plot.keys()), y=list(metrics_to_plot.values()))
        plt.title('Model Performance Metrics')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'metrics.png'))
        plt.close()
    
    def save_metrics(self, output_dir: str):
        """
        Save metrics to JSON file
        Args:
            output_dir (str): Directory to save metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save confusion matrix
        np.save(os.path.join(output_dir, 'confusion_matrix.npy'), self.confusion_matrix)
        
        # Save curve data if available
        if self.roc_curve_data is not None:
            np.savez(os.path.join(output_dir, 'roc_curve_data.npz'), **self.roc_curve_data)
        if self.pr_curve_data is not None:
            np.savez(os.path.join(output_dir, 'pr_curve_data.npz'), **self.pr_curve_data)
    
    def print_metrics(self):
        """Print metrics to console"""
        print("\nModel Performance Metrics:")
        print("-" * 30)
        for metric, value in self.metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
        
        # Print confusion matrix statistics
        tn, fp, fn, tp = self.confusion_matrix.ravel()
        print("\nConfusion Matrix Statistics:")
        print(f"True Negatives: {tn:,}")
        print(f"False Positives: {fp:,}")
        print(f"False Negatives: {fn:,}")
        print(f"True Positives: {tp:,}")
        print(f"False Positive Rate: {fp/(fp+tn):.4f}")
        print(f"False Negative Rate: {fn/(fn+tp):.4f}") 