"""
Model evaluation module with standard metrics and COV (Coefficient of Variation).

COV is a critical engineering metric for CFDST column strength prediction:
COV = σ_ξ / μ_ξ, where ξ = y_pred / y_true
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from .utils import setup_logger


class ModelEvaluator:
    """
    Comprehensive model evaluator with focus on COV metric.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize ModelEvaluator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or setup_logger(__name__)

    def calculate_cov(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Coefficient of Variation (COV).

        COV = σ_ξ / μ_ξ, where ξ = y_pred / y_true

        This is the key engineering metric for CFDST column strength prediction.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            COV value (lower is better, ideally < 0.1)

        Raises:
            ValueError: If y_true contains zeros
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Avoid division by zero
        if np.any(y_true == 0):
            # Add small epsilon to zero values
            mask = y_true == 0
            self.logger.warning(f"Found {np.sum(mask)} zero values in y_true, adding epsilon")
            y_true = y_true.copy()
            y_true[mask] = 1e-10

        # Calculate ξ = y_pred / y_true
        xi = y_pred / y_true

        # Calculate statistics
        mu_xi = np.mean(xi)
        sigma_xi = np.std(xi, ddof=1)  # Sample standard deviation

        # Calculate COV
        cov = sigma_xi / mu_xi if mu_xi != 0 else np.inf

        return float(cov)

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary containing all metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Handle edge cases
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Empty arrays provided")

        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

        metrics = {
            'R2': r2_score(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,  # As percentage
            'COV': self.calculate_cov(y_true, y_pred),
        }

        # Additional statistics
        metrics['mean_prediction'] = np.mean(y_pred)
        metrics['mean_actual'] = np.mean(y_true)
        metrics['std_prediction'] = np.std(y_pred)
        metrics['std_actual'] = np.std(y_true)

        return metrics

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dataset_name: str = "Test"
    ) -> Dict[str, float]:
        """
        Evaluate model and log results.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            dataset_name: Name of dataset (e.g., "Train", "Test")

        Returns:
            Dictionary of metrics
        """
        metrics = self.calculate_all_metrics(y_true, y_pred)

        self.logger.info(f"\n{'='*50}")
        self.logger.info(f"Evaluation Results - {dataset_name} Set")
        self.logger.info(f"{'='*50}")
        self.logger.info(f"R² Score:        {metrics['R2']:.6f}")
        self.logger.info(f"RMSE:            {metrics['RMSE']:.4f}")
        self.logger.info(f"MAE:             {metrics['MAE']:.4f}")
        self.logger.info(f"MAPE:            {metrics['MAPE']:.4f}%")
        self.logger.info(f"COV:             {metrics['COV']:.6f}")
        self.logger.info(f"{'='*50}")

        return metrics

    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: Any
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation and return metrics for each fold.

        Args:
            model: Model to evaluate
            X: Feature matrix
            y: Target vector
            cv: Cross-validation splitter

        Returns:
            Dictionary with metrics for each fold
        """
        from sklearn.model_selection import cross_val_predict

        # Get predictions for each fold
        y_pred = cross_val_predict(model, X, y, cv=cv)

        # Calculate metrics
        metrics = self.calculate_all_metrics(y, y_pred)

        self.logger.info(f"\nCross-Validation Results ({cv.get_n_splits()} folds)")
        self.logger.info(f"R²:   {metrics['R2']:.6f}")
        self.logger.info(f"RMSE: {metrics['RMSE']:.4f}")
        self.logger.info(f"COV:  {metrics['COV']:.6f}")

        return metrics

    def get_metric_summary(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for multiple metric dictionaries.

        Args:
            metrics_list: List of metric dictionaries

        Returns:
            Dictionary with mean and std for each metric
        """
        if not metrics_list:
            return {}

        summary = {}
        metric_names = metrics_list[0].keys()

        for metric_name in metric_names:
            values = [m[metric_name] for m in metrics_list if metric_name in m]
            summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        return summary

    def is_better_metric(
        self,
        current: float,
        best: float,
        metric_name: str
    ) -> bool:
        """
        Determine if current metric is better than best.

        Args:
            current: Current metric value
            best: Best metric value so far
            metric_name: Name of the metric

        Returns:
            True if current is better
        """
        # Metrics where higher is better
        higher_is_better = ['R2']

        # Metrics where lower is better
        lower_is_better = ['RMSE', 'MAE', 'MAPE', 'COV']

        if metric_name in higher_is_better:
            return current > best
        elif metric_name in lower_is_better:
            return current < best
        else:
            # Default: lower is better
            return current < best

    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Format metrics as a readable string.

        Args:
            metrics: Dictionary of metrics

        Returns:
            Formatted string
        """
        lines = [
            f"R² = {metrics['R2']:.4f}",
            f"RMSE = {metrics['RMSE']:.4f}",
            f"MAE = {metrics['MAE']:.4f}",
            f"MAPE = {metrics['MAPE']:.2f}%",
            f"COV = {metrics['COV']:.6f}"
        ]
        return " | ".join(lines)
