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

from utils import ensure_dir, setup_logger


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
            lines = [
                f"R² = {metrics['R2']:.4f}",
                f"RMSE = {metrics['RMSE']:.2f}",
                f"MAPE = {metrics['MAPE']:.2f}%",
                f"COV = {metrics['COV']:.4f}",
                f"μ_ξ = {metrics.get('mu_xi', 0):.4f}",
                f"±10% = {metrics.get('within_10pct', 0):.1f}%",
                f"±20% = {metrics.get('within_20pct', 0):.1f}%"
            ]
            textstr = '\n'.join(lines)
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.75, textstr, transform=ax.transAxes, fontsize=10,
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

    def plot_ratio_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create ratio analysis plots (ξ = y_pred / y_true).

        This visualization helps identify systematic bias and is more
        suitable for data with wide dynamic ranges.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            dataset_name: Name of dataset
            save_path: Optional path to save the figure
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Calculate ratio, avoiding division by zero
        mask = y_true != 0
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        xi = y_pred_filtered / y_true_filtered

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)

        # 1. Ratio distribution histogram
        ax1 = axes[0, 0]
        ax1.hist(xi, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=1.0, color='r', linestyle='--', lw=2, label='Perfect (ξ=1)')
        ax1.axvline(x=np.mean(xi), color='g', linestyle='--', lw=2, label=f'Mean ξ={np.mean(xi):.3f}')
        ax1.set_xlabel('Ratio ξ = Predicted / Actual')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Ratio Distribution (Should Center at 1.0)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add text with statistics
        mean_xi = np.mean(xi)
        std_xi = np.std(xi)
        within_10pct = np.sum((xi >= 0.90) & (xi <= 1.10)) / len(xi) * 100
        within_20pct = np.sum((xi >= 0.80) & (xi <= 1.20)) / len(xi) * 100
        ax1.text(0.05, 0.95, f'μ_ξ={mean_xi:.4f}\nσ_ξ={std_xi:.4f}\nWithin ±10%: {within_10pct:.1f}%\nWithin ±20%: {within_20pct:.1f}%',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 2. Ratio vs Actual (check for systematic bias across value range)
        ax2 = axes[0, 1]
        ax2.scatter(y_true_filtered, xi, alpha=0.6, edgecolors='black', linewidth=0.5)
        ax2.axhline(y=1.0, color='r', linestyle='--', lw=2, label='Perfect')
        ax2.axhline(y=1.2, color='orange', linestyle=':', lw=1, alpha=0.7, label='±20% bounds')
        ax2.axhline(y=0.8, color='orange', linestyle=':', lw=1, alpha=0.7)
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Ratio ξ = Predicted / Actual')
        ax2.set_title('Ratio vs Actual (Check for Systematic Bias)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative distribution of ratio
        ax3 = axes[1, 0]
        xi_sorted = np.sort(xi)
        cumulative = np.arange(1, len(xi_sorted) + 1) / len(xi_sorted) * 100
        ax3.plot(xi_sorted, cumulative, 'b-', linewidth=2)
        ax3.axvline(x=1.0, color='r', linestyle='--', lw=2)
        ax3.axhline(y=50, color='g', linestyle='--', lw=1, alpha=0.7, label='50%')
        ax3.set_xlabel('Ratio ξ = Predicted / Actual')
        ax3.set_ylabel('Cumulative Percentage (%)')
        ax3.set_title('Cumulative Distribution of Ratio')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Error bands visualization
        ax4 = axes[1, 1]
        relative_errors = (xi - 1) * 100  # Convert to percentage
        ax4.hist(relative_errors, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax4.axvline(x=0, color='r', linestyle='--', lw=2, label='Perfect (0%)')
        ax4.axvline(x=10, color='g', linestyle=':', lw=1, alpha=0.7, label='±10%')
        ax4.axvline(x=-10, color='g', linestyle=':', lw=1, alpha=0.7)
        ax4.axvline(x=20, color='orange', linestyle=':', lw=1, alpha=0.7, label='±20%')
        ax4.axvline(x=-20, color='orange', linestyle=':', lw=1, alpha=0.7)
        ax4.set_xlabel('Relative Error (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Relative Error Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        fig.suptitle(f'Ratio Analysis (ξ = Predicted/Actual) - {dataset_name} Set',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved ratio analysis plot to: {save_path}")

        plt.close()

    def plot_residual_distribution(
        self,
        residuals_train: np.ndarray,
        residuals_test: np.ndarray,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot residual distribution for train and test sets.

        Args:
            residuals_train: Training residuals
            residuals_test: Test residuals
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)

        # Training set
        ax1 = axes[0]
        ax1.hist(residuals_train, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(x=0, color='r', linestyle='--', lw=2)
        ax1.axvline(x=np.mean(residuals_train), color='g', linestyle='--', lw=2)
        ax1.set_xlabel('Residuals (ε = y - ŷ)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Training Set Residual Distribution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Add statistics
        mean_train = np.mean(residuals_train)
        std_train = np.std(residuals_train)
        ax1.text(0.05, 0.95, f'Mean: {mean_train:.4f}\nStd: {std_train:.4f}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Test set
        ax2 = axes[1]
        ax2.hist(residuals_test, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)
        ax2.axvline(x=np.mean(residuals_test), color='g', linestyle='--', lw=2)
        ax2.set_xlabel('Residuals (ε = y - ŷ)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Test Set Residual Distribution', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add statistics
        mean_test = np.mean(residuals_test)
        std_test = np.std(residuals_test)
        ax2.text(0.05, 0.95, f'Mean: {mean_test:.4f}\nStd: {std_test:.4f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.suptitle('Residual Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved residual distribution plot to: {save_path}")

        plt.close()

    def plot_prediction_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        confidence: float = 0.97,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot prediction intervals vs actual values.

        Args:
            y_true: Ground truth values
            y_pred: Point predictions
            lower_bound: Lower bounds of prediction intervals
            upper_bound: Upper bounds of prediction intervals
            confidence: Confidence level
            save_path: Optional path to save the figure
        """
        # Sort by actual values for better visualization
        sort_idx = np.argsort(y_true)
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        lower_sorted = lower_bound[sort_idx]
        upper_sorted = upper_bound[sort_idx]

        # Create sample index for x-axis
        x_idx = np.arange(len(y_true))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot prediction interval as shaded area
        ax.fill_between(x_idx, lower_sorted, upper_sorted, alpha=0.3,
                        color='blue', label=f'{confidence:.0%} Prediction Interval')

        # Plot point predictions
        ax.plot(x_idx, y_pred_sorted, 'b-', linewidth=1, label='Point Prediction')

        # Plot actual values
        ax.scatter(x_idx, y_true_sorted, c='red', s=20, alpha=0.6,
                   label='Actual Values', zorder=5)

        ax.set_xlabel('Sample Index (sorted by actual value)', fontsize=12)
        ax.set_ylabel('Nexp (kN)', fontsize=12)
        ax.set_title(f'{confidence:.0%} Prediction Intervals', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Calculate coverage
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound))

        # Add statistics
        interval_width = upper_bound - lower_bound
        mean_width_pct = np.mean(interval_width / y_true) * 100
        ax.text(0.02, 0.98, f'Actual Coverage: {coverage:.2%}\n'
                           f'Target Coverage: {confidence:.0%}\n'
                           f'Mean Interval Width: {mean_width_pct:.2f}% of actual',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved prediction intervals plot to: {save_path}")

        plt.close()

    def plot_calibration_curve(
        self,
        ngboost_trainer: Any,
        X: np.ndarray,
        y_true: np.ndarray,
        xgb_pred: np.ndarray,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Plot calibration curve showing predicted vs actual coverage rates.

        Args:
            ngboost_trainer: Trained NGBoost trainer
            X: Feature matrix
            y_true: True values (original scale)
            xgb_pred: XGBoost predictions (original scale)
            save_path: Optional path to save the figure
        """
        confidence_levels = np.arange(0.1, 1.0, 0.05)
        actual_coverages = []

        for conf in confidence_levels:
            epsilon_lower, epsilon_upper = ngboost_trainer.predict_interval(X, conf)
            y_lower = xgb_pred + epsilon_lower
            y_upper = xgb_pred + epsilon_upper
            coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
            actual_coverages.append(coverage)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Plot calibration curve
        ax.plot(confidence_levels * 100, np.array(actual_coverages) * 100,
                'o-', linewidth=2, markersize=8, label='Actual Coverage')

        # Plot perfect calibration line
        ax.plot(confidence_levels * 100, confidence_levels * 100,
                'k--', linewidth=1.5, alpha=0.7, label='Perfect Calibration')

        # Highlight 97% target
        ax.axvline(x=97, color='r', linestyle=':', alpha=0.7)
        ax.axhline(y=97, color='r', linestyle=':', alpha=0.7)

        ax.set_xlabel('Predicted Confidence Level (%)', fontsize=12)
        ax.set_ylabel('Actual Coverage Rate (%)', fontsize=12)
        ax.set_title('Prediction Interval Calibration Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])

        # Add text annotation
        idx_97 = np.argmin(np.abs(confidence_levels - 0.97))
        actual_97 = actual_coverages[idx_97]
        ax.text(0.05, 0.95, f'At 97% confidence:\nActual coverage: {actual_97:.2%}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved calibration curve to: {save_path}")

        plt.close()

    # ------------------------------------------------------------------
    # NGBoost Stacking-specific visualizations
    # ------------------------------------------------------------------

    @staticmethod
    def _lowess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """Simple LOWESS-like smoothing using ordered bin means."""
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        order = np.argsort(x)
        x_s, y_s = x[order], y[order]
        n = len(x_s)
        window = max(5, int(frac * n))
        # Convolution with constant window (same padding)
        pad = window // 2
        y_padded = np.concatenate([np.full(pad, y_s[0]), y_s, np.full(pad, y_s[-1])])
        kernel = np.ones(window) / window
        y_smooth = np.convolve(y_padded, kernel, mode='same')[pad:pad + n]
        return x_s, y_smooth

    def plot_stacking_comparison(
        self,
        y_true: np.ndarray,
        y_pred_xgb: np.ndarray,
        y_pred_final: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        真实值 vs 预测值散点图（加残差修正对比）。
        直观对比 XGBoost 单独预测与堆叠修正后的差异。
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        min_val = min(np.min(y_true), np.min(y_pred_xgb), np.min(y_pred_final))
        max_val = max(np.max(y_true), np.max(y_pred_xgb), np.max(y_pred_final))

        # Perfect prediction line
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction (y=x)', zorder=1)

        # XGBoost predictions
        ax.scatter(y_true, y_pred_xgb, c='steelblue', s=40, alpha=0.6,
                   edgecolors='black', linewidth=0.3, label='XGBoost Only', zorder=2)

        # Stacked predictions
        ax.scatter(y_true, y_pred_final, c='crimson', s=40, alpha=0.6,
                   edgecolors='black', linewidth=0.3, label='XGBoost + NGBoost (Final)', zorder=3)

        ax.set_xlabel('Actual Nexp (kN)', fontsize=12)
        ax.set_ylabel('Predicted Nexp (kN)', fontsize=12)
        ax.set_title(f'Actual vs Predicted – {dataset_name} Set\n(Stacking Correction Comparison)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Metrics text
        from sklearn.metrics import r2_score, mean_squared_error
        r2_xgb = r2_score(y_true, y_pred_xgb)
        r2_final = r2_score(y_true, y_pred_final)
        rmse_xgb = np.sqrt(mean_squared_error(y_true, y_pred_xgb))
        rmse_final = np.sqrt(mean_squared_error(y_true, y_pred_final))
        textstr = (f"XGBoost   R²={r2_xgb:.4f}  RMSE={rmse_xgb:.2f}\n"
                   f"Stacked   R²={r2_final:.4f}  RMSE={rmse_final:.2f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax.text(0.05, 0.7, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved stacking comparison plot to: {save_path}")
        plt.close()

    def plot_residual_diagnostics(
        self,
        y_true: np.ndarray,
        y_pred_xgb: np.ndarray,
        y_pred_final: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        残差诊断图（核心灵魂）。
        包括残差 vs 预测值散点图（Lowess 平滑）、残差分布直方图、Q-Q 图。
        """
        residuals_xgb = y_true - y_pred_xgb
        residuals_stack = y_true - y_pred_final

        fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=self.dpi)

        # ---------- Row 0: Residuals vs Predicted ----------
        # Left: XGBoost residuals
        ax = axes[0, 0]
        ax.scatter(y_pred_xgb, residuals_xgb, alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.3)
        x_smooth, y_smooth = self._lowess_smooth(y_pred_xgb, residuals_xgb, frac=0.3)
        ax.plot(x_smooth, y_smooth, color='gold', lw=2.5, label='Lowess trend')
        ax.axhline(0, color='r', linestyle='--', lw=1.5)
        ax.set_xlabel('XGBoost Predicted Nexp (kN)')
        ax.set_ylabel('Residuals (y - ŷ)')
        ax.set_title('Residuals vs Predicted (XGBoost)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Middle: Stacked residuals
        ax = axes[0, 1]
        ax.scatter(y_pred_xgb, residuals_stack, alpha=0.5, s=30, c='crimson', edgecolors='black', linewidth=0.3)
        x_smooth, y_smooth = self._lowess_smooth(y_pred_xgb, residuals_stack, frac=0.3)
        ax.plot(x_smooth, y_smooth, color='gold', lw=2.5, label='Lowess trend')
        ax.axhline(0, color='r', linestyle='--', lw=1.5)
        ax.set_xlabel('XGBoost Predicted Nexp (kN)')
        ax.set_ylabel('Residuals (y - ŷ_final)')
        ax.set_title('Residuals vs Predicted (Stacked)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Right: Residual histogram comparison
        ax = axes[0, 2]
        bins = np.linspace(min(residuals_xgb.min(), residuals_stack.min()),
                           max(residuals_xgb.max(), residuals_stack.max()), 35)
        ax.hist(residuals_xgb, bins=bins, alpha=0.5, color='steelblue', edgecolor='black', label='XGBoost')
        ax.hist(residuals_stack, bins=bins, alpha=0.5, color='crimson', edgecolor='black', label='Stacked')
        ax.axvline(0, color='k', linestyle='--', lw=1.5)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Residual Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ---------- Row 1: Normality diagnostics ----------
        from scipy import stats

        # Left: XGBoost Q-Q
        ax = axes[1, 0]
        stats.probplot(residuals_xgb, dist="norm", sparams=(residuals_xgb.mean(), residuals_xgb.std()), plot=ax)
        ax.get_lines()[0].set_markerfacecolor('steelblue')
        ax.get_lines()[0].set_markeredgecolor('black')
        ax.get_lines()[0].set_markersize(5)
        ax.get_lines()[1].set_color('red')
        ax.set_title('Q-Q Plot (XGBoost Residuals)')
        ax.grid(True, alpha=0.3)

        # Middle: Stacked Q-Q
        ax = axes[1, 1]
        stats.probplot(residuals_stack, dist="norm", sparams=(residuals_stack.mean(), residuals_stack.std()), plot=ax)
        ax.get_lines()[0].set_markerfacecolor('crimson')
        ax.get_lines()[0].set_markeredgecolor('black')
        ax.get_lines()[0].set_markersize(5)
        ax.get_lines()[1].set_color('red')
        ax.set_title('Q-Q Plot (Stacked Residuals)')
        ax.grid(True, alpha=0.3)

        # Right: Box plot comparison
        ax = axes[1, 2]
        bp = ax.boxplot([residuals_xgb, residuals_stack], labels=['XGBoost', 'Stacked'],
                        patch_artist=True, showmeans=True)
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('crimson')
        ax.axhline(0, color='k', linestyle='--', lw=1.5)
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Spread Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(f'Residual Diagnostics – {dataset_name} Set', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved residual diagnostics plot to: {save_path}")
        plt.close()

    def plot_uncertainty_calibration(
        self,
        ngboost_trainer: Any,
        X: np.ndarray,
        residuals_true: np.ndarray,
        y_true: np.ndarray,
        y_pred_xgb: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        不确定性校准图（NGBoost 独特价值）。
        包含 PIT 直方图 和 置信区间覆盖图。
        """
        from scipy import stats

        mu, sigma = ngboost_trainer.predict_params(X)
        mu = np.array(mu).flatten()
        sigma = np.array(sigma).flatten()
        residuals_true = np.array(residuals_true).flatten()
        y_true = np.array(y_true).flatten()
        y_pred_xgb = np.array(y_pred_xgb).flatten()

        # PIT values
        pit = stats.norm.cdf(residuals_true, loc=mu, scale=sigma + 1e-8)

        # Prediction intervals for final predictions
        epsilon_lower, epsilon_upper = ngboost_trainer.predict_interval(X)
        y_lower = y_pred_xgb + epsilon_lower
        y_upper = y_pred_xgb + epsilon_upper
        y_pred_final = y_pred_xgb + mu

        # Sort by predicted final for coverage ribbon plot
        sort_idx = np.argsort(y_pred_final)
        y_pred_final_s = y_pred_final[sort_idx]
        y_lower_s = y_lower[sort_idx]
        y_upper_s = y_upper[sort_idx]
        y_true_s = y_true[sort_idx]
        x_idx = np.arange(len(y_true))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)

        # ---------- PIT Histogram ----------
        ax = axes[0]
        ax.hist(pit, bins=15, color='teal', edgecolor='black', alpha=0.7)
        ax.axhline(len(pit) / 15, color='red', linestyle='--', lw=2, label='Uniform (perfect calibration)')
        ax.set_xlabel('PIT Value (CDF of residual under predicted distribution)')
        ax.set_ylabel('Frequency')
        ax.set_title('PIT Histogram (Uncertainty Calibration)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add KS test info
        ks_stat, ks_p = stats.kstest(pit, 'uniform')
        ax.text(0.05, 0.95, f'KS vs Uniform:\nstat={ks_stat:.3f}, p={ks_p:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # ---------- Interval Coverage Ribbon ----------
        ax = axes[1]
        conf_level = ngboost_trainer.confidence_level
        ax.fill_between(x_idx, y_lower_s, y_upper_s, alpha=0.25, color='blue',
                        label=f'{conf_level:.0%} Prediction Interval')
        ax.plot(x_idx, y_pred_final_s, 'b-', linewidth=1, label='Final Prediction', zorder=2)
        ax.scatter(x_idx, y_true_s, c='red', s=15, alpha=0.6, label='Actual', zorder=3)

        coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
        mean_width = np.mean(y_upper - y_lower)
        ax.set_xlabel('Sample Index (sorted by final prediction)')
        ax.set_ylabel('Nexp (kN)')
        ax.set_title(f'Prediction Interval Coverage ({dataset_name})')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        ax.text(0.02, 0.7, f'Actual Coverage: {coverage:.2%}\nMean Width: {mean_width:.2f} kN',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        fig.suptitle('Uncertainty Calibration Assessment', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved uncertainty calibration plot to: {save_path}")
        plt.close()

    def plot_ngboost_interpretation(
        self,
        ngboost_trainer: Any,
        X: np.ndarray,
        feature_names: List[str],
        top_n: int = 5,
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        特征重要性与残差敏感性图（业务解释）。
        理解哪些特征导致 XGBoost 预测不准，从而触发 NGBoost 的大幅修正。
        """
        model = ngboost_trainer.model
        if model is None:
            self.logger.warning("NGBoost model not trained; skipping interpretation plot.")
            return

        # Feature importances: NGBoost returns (n_params, n_features) for distributional models
        raw_importance = np.array(model.feature_importances_)
        if raw_importance.ndim == 2:
            # Average across distribution parameters (e.g., loc and scale for Normal)
            importance = raw_importance.mean(axis=0)
            param_labels = getattr(model, 'param_labels', [f'Param{i}' for i in range(raw_importance.shape[0])])
        elif raw_importance.ndim == 1:
            importance = raw_importance
            param_labels = []
        else:
            self.logger.warning(f"Unexpected feature_importances_ shape {raw_importance.shape}; skipping interpretation plot.")
            return

        if len(importance) != len(feature_names):
            self.logger.warning(f"Feature importance length mismatch ({len(importance)} vs {len(feature_names)}); skipping interpretation plot.")
            return

        # Predict residuals (NGBoost output)
        residual_pred = model.predict(X).flatten()
        X_arr = np.array(X)

        # Sort by importance
        imp_order = np.argsort(importance)[::-1]
        top_indices = imp_order[:top_n]

        n_cols = 3
        n_rows = int(np.ceil((top_n + 1) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * 3.5), dpi=self.dpi)
        axes = np.atleast_1d(axes).flatten()

        # First subplot: feature importance bar chart
        ax = axes[0]
        top_features = [feature_names[i] for i in top_indices]
        top_imp = importance[top_indices]
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
        bars = ax.barh(range(len(top_features)), top_imp, color=colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('NGBoost Feature Importance')
        ax.grid(True, axis='x', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, top_imp)):
            ax.text(val + 0.01 * max(top_imp), i, f'{val:.3f}', va='center', fontsize=8)

        # Remaining subplots: feature value vs predicted residual
        for idx, feat_idx in enumerate(top_indices, start=1):
            ax = axes[idx]
            feat_vals = X_arr[:, feat_idx]
            ax.scatter(feat_vals, residual_pred, alpha=0.5, s=20, c='darkgreen', edgecolors='black', linewidth=0.3)
            # Add lowess trend
            if len(feat_vals) > 5:
                x_s, y_s = self._lowess_smooth(feat_vals, residual_pred, frac=0.3)
                ax.plot(x_s, y_s, color='gold', lw=2, label='Lowess trend')
            ax.axhline(0, color='r', linestyle='--', lw=1)
            ax.set_xlabel(feature_names[feat_idx], fontsize=9)
            ax.set_ylabel('Predicted Residual')
            ax.set_title(f'{feature_names[feat_idx]}\nvs Predicted Residual', fontsize=10)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(top_n + 1, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('NGBoost Interpretation: Feature Importance & Residual Sensitivity', fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved NGBoost interpretation plot to: {save_path}")
        plt.close()

    # ------------------------------------------------------------------
    # Engineering-oriented presentations (Direction 2: XGBoost point + NGBoost uncertainty)
    # ------------------------------------------------------------------

    def plot_prediction_reliability_map(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        预测可靠性地图。
        用颜色/形状区分落在预测区间内/外的样本，直观展示 NGBoost 区间覆盖的实用价值。
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_lower = np.array(y_lower).flatten()
        y_upper = np.array(y_upper).flatten()

        inside = (y_true >= y_lower) & (y_true <= y_upper)
        within_20pct = ((y_pred / y_true) >= 0.8) & ((y_pred / y_true) <= 1.2)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction', zorder=1)
        ax.plot([min_val, max_val], [min_val * 0.8, max_val * 0.8], 'g:', lw=1, alpha=0.5, zorder=1)
        ax.plot([min_val, max_val], [min_val * 1.2, max_val * 1.2], 'g:', lw=1, alpha=0.5, label='±20% bounds', zorder=1)

        # Green: inside interval
        ax.scatter(y_true[inside], y_pred[inside], c='limegreen', s=50, alpha=0.7,
                   edgecolors='black', linewidth=0.3, label='Inside PI (Reliable)', zorder=3)

        # Orange: outside interval but within ±20%
        mask_warn = (~inside) & within_20pct
        ax.scatter(y_true[mask_warn], y_pred[mask_warn], c='orange', s=80, alpha=0.8,
                   marker='^', edgecolors='black', linewidth=0.5, label='Outside PI but ±20%', zorder=4)

        # Red: outside interval and outside ±20%
        mask_risk = (~inside) & (~within_20pct)
        ax.scatter(y_true[mask_risk], y_pred[mask_risk], c='crimson', s=100, alpha=0.9,
                   marker='X', edgecolors='black', linewidth=0.5, label='High Risk', zorder=5)

        coverage = np.mean(inside)
        ax.set_xlabel('Actual Nexp (kN)', fontsize=12)
        ax.set_ylabel('Predicted Nexp (kN)', fontsize=12)
        ax.set_title(f'Prediction Reliability Map – {dataset_name}\n(XGBoost point prediction + NGBoost interval coverage)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        n_warn = np.sum(mask_warn)
        n_risk = np.sum(mask_risk)
        ax.text(0.05, 0.6, f'Coverage: {coverage:.1%}\nWarning: {n_warn}\nHigh Risk: {n_risk}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved prediction reliability map to: {save_path}")
        plt.close()

    def plot_uncertainty_error_correlation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        区间宽度 vs 实际绝对误差。
        证明 NGBoost 的预测区间是"有信息的"——区间越宽，实际误差往往越大。
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        interval_width = np.array(y_upper).flatten() - np.array(y_lower).flatten()
        abs_error = np.abs(y_true - y_pred)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        ax.scatter(interval_width, abs_error, alpha=0.5, s=40, c='steelblue',
                   edgecolors='black', linewidth=0.3)

        # Lowess trend
        if len(interval_width) > 5:
            x_s, y_s = self._lowess_smooth(interval_width, abs_error, frac=0.3)
            ax.plot(x_s, y_s, color='gold', lw=2.5, label='Lowess trend')

        # Pearson correlation
        from scipy import stats
        r, p = stats.pearsonr(interval_width, abs_error)

        ax.set_xlabel('Prediction Interval Width (kN)', fontsize=12)
        ax.set_ylabel('Actual Absolute Error (kN)', fontsize=12)
        ax.set_title(f'Uncertainty-Error Correlation – {dataset_name}\n(Does wider interval mean larger error?)', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.text(0.05, 0.98, f'Pearson r = {r:.3f}\np-value = {p:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved uncertainty-error correlation plot to: {save_path}")
        plt.close()

    def plot_correction_benefit(
        self,
        y_true: np.ndarray,
        y_pred_xgb: np.ndarray,
        y_pred_final: np.ndarray,
        residual_pred: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        NGBoost 修正效益图。
        展示 NGBoost 在哪些样本上有效修正了点预测（尤其是大误差样本）。
        """
        y_true = np.array(y_true).flatten()
        y_pred_xgb = np.array(y_pred_xgb).flatten()
        y_pred_final = np.array(y_pred_final).flatten()
        residual_pred = np.array(residual_pred).flatten()

        xgb_err = np.abs(y_true - y_pred_xgb)
        final_err = np.abs(y_true - y_pred_final)
        improvement = xgb_err - final_err  # positive = corrected effectively

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        scatter = ax.scatter(xgb_err, improvement, c=np.abs(residual_pred), s=50,
                             cmap='RdYlGn_r', alpha=0.7, edgecolors='black', linewidth=0.3)
        plt.colorbar(scatter, ax=ax, label='|NGBoost Predicted Residual|')

        ax.axhline(0, color='k', linestyle='--', lw=1.5)
        ax.set_xlabel('XGBoost Absolute Error (kN)', fontsize=12)
        ax.set_ylabel('Error Improvement after Stacking (kN)', fontsize=12)
        ax.set_title(f'Correction Benefit – {dataset_name}\n(Positive = NGBoost pulled prediction closer)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

        n_helped = np.sum(improvement > 0)
        n_hurt = np.sum(improvement < 0)
        mean_improve_large = np.mean(improvement[xgb_err > np.percentile(xgb_err, 75)]) if np.any(xgb_err > 0) else 0

        ax.text(0.98, 0.98, f'Helped: {n_helped}\nHurt: {n_hurt}\nMean improve (top 25% error): {mean_improve_large:.1f} kN',
                transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved correction benefit plot to: {save_path}")
        plt.close()

    def plot_safety_margin(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_lower: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        工程安全裕度图。
        展示用 NGBoost 下界作为保守设计值时，相比 XGBoost 点预测提供了多少安全缓冲。
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_lower = np.array(y_lower).flatten()

        sort_idx = np.argsort(y_pred)
        y_pred_s = y_pred[sort_idx]
        y_lower_s = y_lower[sort_idx]
        y_true_s = y_true[sort_idx]
        x_idx = np.arange(len(y_true))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Fill safety margin area
        ax.fill_between(x_idx, y_lower_s, y_pred_s, alpha=0.3, color='blue',
                        label='Safety Margin (XGBoost – Lower Bound)')
        ax.plot(x_idx, y_pred_s, 'b-', linewidth=1.5, label='XGBoost Point Prediction', zorder=2)
        ax.plot(x_idx, y_lower_s, 'b--', linewidth=1.5, label='Conservative Estimate (Lower Bound)', zorder=2)
        ax.scatter(x_idx, y_true_s, c='red', s=15, alpha=0.6, label='Actual', zorder=3)

        safety_rate = np.mean(y_true >= y_lower_s)
        mean_margin = np.mean(y_pred_s - y_lower_s)
        mean_margin_pct = np.mean((y_pred_s - y_lower_s) / y_pred_s) * 100

        ax.set_xlabel('Sample Index (sorted by XGBoost prediction)', fontsize=12)
        ax.set_ylabel('Nexp (kN)', fontsize=12)
        ax.set_title(f'Engineering Safety Margin – {dataset_name}\n(Using lower bound as conservative design value)', fontsize=13, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        ax.text(0.02, 0.6, f'Actual ≥ Lower Bound: {safety_rate:.1%}\nMean Margin: {mean_margin:.1f} kN ({mean_margin_pct:.1f}%)',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved safety margin plot to: {save_path}")
        plt.close()

    def plot_3d_prediction_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
        dataset_name: str = "Test",
        save_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        3D散点图可视化预测点、预测区间和真实值的偏差。

        X轴: 真实值 (Actual)
        Y轴: 预测值 (Predicted)
        Z轴: 绝对偏差 (|Actual - Predicted|)
        点大小: 预测区间宽度 (反映不确定性)
        颜色: 是否被预测区间覆盖 (绿色=覆盖, 红色=未覆盖)
        同时绘制完美预测平面 (y=x, z=0) 作为参照。
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        y_lower = np.array(y_lower).flatten()
        y_upper = np.array(y_upper).flatten()

        abs_error = np.abs(y_true - y_pred)
        interval_width = y_upper - y_lower
        inside = (y_true >= y_lower) & (y_true <= y_upper)

        fig = plt.figure(figsize=(14, 10), dpi=self.dpi)
        ax = fig.add_subplot(111, projection='3d')

        # Color map: green for inside PI, red for outside
        colors = np.where(inside, 'limegreen', 'crimson')

        # Scale point size by interval width (clip to reasonable range)
        sizes = np.clip(interval_width / (np.median(interval_width) + 1e-8) * 40, 20, 300)

        # Scatter plot: actual vs predicted vs absolute error
        ax.scatter(y_true, y_pred, abs_error, c=colors, s=sizes,
                   alpha=0.7, edgecolors='black', linewidth=0.3, depthshade=True)

        # Draw perfect prediction plane: y=x, z=0
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        xx, yy = np.meshgrid(
            np.linspace(min_val, max_val, 10),
            np.linspace(min_val, max_val, 10)
        )
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.15, color='gray', zorder=1)
        ax.plot([min_val, max_val], [min_val, max_val], [0, 0],
                'k--', linewidth=2, label='Perfect Prediction (y=x, error=0)', zorder=2)

        # Add drop lines from points to the perfect prediction plane for a few extreme points
        n_show_lines = min(15, len(y_true))
        line_idx = np.argsort(abs_error)[-n_show_lines:]
        for i in line_idx:
            ax.plot([y_true[i], y_true[i]],
                    [y_pred[i], y_true[i]],
                    [abs_error[i], 0],
                    'k-', alpha=0.15, linewidth=0.5, zorder=1)

        ax.set_xlabel('Actual Nexp (kN)', fontsize=11, labelpad=10)
        ax.set_ylabel('Predicted Nexp (kN)', fontsize=11, labelpad=10)
        ax.set_zlabel('Absolute Error (kN)', fontsize=11, labelpad=10)
        ax.set_title(f'3D Prediction Interval Visualization – {dataset_name}\n'
                     f'(Size ∝ Interval Width, Color = Coverage)',
                     fontsize=13, fontweight='bold')

        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                   markersize=10, label=f'Inside PI ({np.sum(inside)} pts)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',
                   markersize=10, label=f'Outside PI ({np.sum(~inside)} pts)'),
            Line2D([0], [0], color='k', linestyle='--', linewidth=2,
                   label='Perfect Prediction'),
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

        # Add statistics text
        coverage = np.mean(inside)
        mean_width = np.mean(interval_width)
        mean_err = np.mean(abs_error)
        textstr = (f'Coverage: {coverage:.1%}\n'
                   f'Mean Width: {mean_width:.1f} kN\n'
                   f'Mean Abs Error: {mean_err:.1f} kN')
        fig.text(0.15, 0.12, textstr, fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            ensure_dir(save_path.parent)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"Saved 3D prediction interval plot to: {save_path}")
        plt.close()
