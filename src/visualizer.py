"""
Visualization module for model evaluation and analysis.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import ensure_dir, setup_logger


class ModelVisualizer:
    """
    Handles visualization of model predictions, feature importance, and residuals.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize ModelVisualizer.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)

        # Visualization settings
        viz_config = config.get('visualization', {})
        self.figsize: Tuple[int, int] = tuple(viz_config.get('figsize', [10, 6]))
        self.dpi = viz_config.get('dpi', 300)
        self.style = viz_config.get('style', 'seaborn-v0_8-whitegrid')
        self.color_palette = viz_config.get('color_palette', 'viridis')

        # Set matplotlib style
        try:
            plt.style.use(self.style)
        except OSError:
            plt.style.use('default')

        self.output_dir = Path(config['paths']['output_dir'])
        ensure_dir(self.output_dir)

    def plot_predictions_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Create scatter plot of predicted vs actual values.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            dataset_name: Name of dataset (e.g., "Train", "Test")
            save_path: Optional path to save the figure
            metrics: Optional metrics dictionary to display on plot
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

        # ±20% error bands
        ax.plot([min_val, max_val], [min_val * 0.8, max_val * 0.8], 'g:', lw=1, alpha=0.5, label='±20% Error')
        ax.plot([min_val, max_val], [min_val * 1.2, max_val * 1.2], 'g:', lw=1, alpha=0.5)

        ax.set_xlabel('Actual Nexp (kN)', fontsize=12)
        ax.set_ylabel('Predicted Nexp (kN)', fontsize=12)
        ax.set_title(f'Predicted vs Actual - {dataset_name} Set', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add metrics text box if provided
        if metrics:
            textstr = '\n'.join([
                f"R² = {metrics['R2']:.4f}",
                f"RMSE = {metrics['RMSE']:.2f}",
                f"MAPE = {metrics['MAPE']:.2f}%",
                f"COV = {metrics['COV']:.4f}"
            ])
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.8, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved prediction plot to: {save_path}")

        plt.close()

    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_values: np.ndarray,
        title: str = "Feature Importance",
        top_n: int = 20,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot feature importance.

        Args:
            feature_names: List of feature names
            importance_values: Array of importance values
            title: Plot title
            top_n: Number of top features to show
            save_path: Optional path to save the figure
        """
        # Sort features by importance
        indices = np.argsort(importance_values)[::-1]

        # Select top N
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importance = importance_values[top_indices]

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_importance, color=colors)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_importance)):
            ax.text(val + 0.01 * max(top_importance), i, f'{val:.3f}',
                   va='center', fontsize=9)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved feature importance plot to: {save_path}")

        plt.close()

    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create residual analysis plots.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            dataset_name: Name of dataset
            save_path: Optional path to save the figure
        """
        residuals = y_pred - y_true
        relative_errors = residuals / y_true * 100  # Percentage

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)

        # 1. Residuals vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)

        # 2. Residuals vs Actual
        ax2 = axes[0, 1]
        ax2.scatter(y_true, residuals, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Actual')
        ax2.grid(True, alpha=0.3)

        # 3. Residual distribution (histogram)
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax3.axvline(x=0, color='r', linestyle='--', lw=2)
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Residual Distribution')
        ax3.grid(True, alpha=0.3)

        # Add statistics
        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        ax3.text(0.05, 0.95, f'Mean: {mean_res:.4f}\nStd: {std_res:.4f}',
                transform=ax3.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 4. Relative error distribution
        ax4 = axes[1, 1]
        ax4.hist(relative_errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax4.axvline(x=0, color='r', linestyle='--', lw=2)
        ax4.set_xlabel('Relative Error (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Relative Error Distribution')
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f'Residual Analysis - {dataset_name} Set', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved residual plot to: {save_path}")

        plt.close()

    def plot_training_history(
        self,
        evals_result: Dict[str, Dict[str, List[float]]],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot training history (for XGBoost evals_result).

        Args:
            evals_result: XGBoost evaluation results dictionary
            save_path: Optional path to save the figure
        """
        if not evals_result:
            self.logger.warning("No evaluation results to plot")
            return

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        for dataset_name, metrics in evals_result.items():
            for metric_name, values in metrics.items():
                ax.plot(values, label=f"{dataset_name}-{metric_name}")

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_title('Training History', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved training history plot to: {save_path}")

        plt.close()

    def plot_feature_selection_results(
        self,
        n_features: List[int],
        metrics: Dict[str, List[float]],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot feature selection results showing metrics vs number of features.

        Args:
            n_features: List of number of features tested
            metrics: Dictionary of metric name to list of values
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        axes = axes.flatten()

        metric_names = ['R2', 'RMSE', 'MAE', 'COV']

        for idx, metric_name in enumerate(metric_names):
            ax = axes[idx]
            if metric_name in metrics:
                ax.plot(n_features, metrics[metric_name], 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Number of Features', fontsize=11)
                ax.set_ylabel(metric_name, fontsize=11)
                ax.set_title(f'{metric_name} vs Number of Features', fontsize=12)
                ax.grid(True, alpha=0.3)

                # Find best value
                if metric_name == 'R2':
                    best_idx = np.argmax(metrics[metric_name])
                else:
                    best_idx = np.argmin(metrics[metric_name])

                best_n = n_features[best_idx]
                best_val = metrics[metric_name][best_idx]
                ax.axvline(x=best_n, color='r', linestyle='--', alpha=0.5)
                ax.text(0.05, 0.95, f'Best: {best_n} features\n{metric_name}={best_val:.4f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.suptitle('Feature Selection Results', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved feature selection plot to: {save_path}")

        plt.close()

    def plot_correlation_matrix(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot correlation matrix of features.

        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            save_path: Optional path to save the figure
        """
        corr_matrix = df[feature_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', square=True, linewidths=0.5, ax=ax)

        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved correlation matrix to: {save_path}")

        plt.close()
