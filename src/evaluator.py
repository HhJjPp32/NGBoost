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

from utils import setup_logger


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

    def calculate_ratio_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate ratio-based metrics (ξ = y_pred / y_true).

        These metrics are independent of the magnitude of values and are
        more suitable for engineering applications with wide dynamic ranges.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Dictionary containing ratio-based metrics
        """
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        # Avoid division by zero
        mask = y_true != 0
        if not np.all(mask):
            self.logger.warning(f"Found {np.sum(~mask)} zero values in y_true, excluding from ratio metrics")
            y_true = y_true[mask]
            y_pred = y_pred[mask]

        # Calculate ξ = y_pred / y_true
        xi = y_pred / y_true

        # Basic statistics
        mu_xi = np.mean(xi)  # Mean ratio (should be close to 1)
        sigma_xi = np.std(xi, ddof=1)  # Sample standard deviation
        median_xi = np.median(xi)

        # Ratio-based RMSE (relative to perfect prediction of 1)
        rmse_ratio = np.sqrt(np.mean((xi - 1.0) ** 2))

        # Ratio-based MAE
        mae_ratio = np.mean(np.abs(xi - 1.0))

        # Error intervals (percentage of predictions within certain bounds)
        within_5pct = np.sum((xi >= 0.95) & (xi <= 1.05)) / len(xi) * 100
        within_10pct = np.sum((xi >= 0.90) & (xi <= 1.10)) / len(xi) * 100
        within_20pct = np.sum((xi >= 0.80) & (xi <= 1.20)) / len(xi) * 100

        return {
            'mu_xi': mu_xi,  # Mean of ratio (ideal: 1.0)
            'rati_mean': mu_xi,  # 预测/试验的均值 (same as mu_xi)
            'sigma_xi': sigma_xi,  # Std of ratio
            'median_xi': median_xi,  # Median of ratio
            'RMSE_ratio': rmse_ratio,  # Ratio-based RMSE (ideal: 0)
            'MAE_ratio': mae_ratio,  # Ratio-based MAE (ideal: 0)
            'within_5pct': within_5pct,  # % within ±5%
            'within_10pct': within_10pct,  # % within ±10%
            'within_20pct': within_20pct,  # % within ±20%
        }

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
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,  # As percentage
            'COV': self.calculate_cov(y_true, y_pred),
        }

        # Add ratio-based metrics
        ratio_metrics = self.calculate_ratio_metrics(y_true, y_pred)
        metrics.update(ratio_metrics)

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
        self.logger.info(f"R2 Score:        {metrics['R2']:.6f}")
        self.logger.info(f"MSE:             {metrics['MSE']:.4f}")
        self.logger.info(f"RMSE:            {metrics['RMSE']:.4f}")
        self.logger.info(f"MAE:             {metrics['MAE']:.4f}")
        self.logger.info(f"MAPE:            {metrics['MAPE']:.4f}%")
        self.logger.info(f"COV:             {metrics['COV']:.6f}")
        self.logger.info(f"{'-'*50}")
        self.logger.info(f"Ratio Metrics (xi = pred/actual):")
        self.logger.info(f"  mu_xi (mean):  {metrics['mu_xi']:.6f}  (ideal: 1.0)")
        self.logger.info(f"  rati_mean:     {metrics['rati_mean']:.6f}  (预测/试验的均值, ideal: 1.0)")
        self.logger.info(f"  sigma_xi (std): {metrics['sigma_xi']:.6f}")
        self.logger.info(f"  RMSE_ratio:    {metrics['RMSE_ratio']:.6f}")
        self.logger.info(f"  Within +/-10%:   {metrics['within_10pct']:.2f}%")
        self.logger.info(f"  Within +/-20%:   {metrics['within_20pct']:.2f}%")
        self.logger.info(f"{'='*50}")

        return metrics

    def calculate_overfitting_metrics(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate overfitting indicators by comparing train vs test metrics.

        Args:
            train_metrics: Metrics from training set
            test_metrics: Metrics from test set

        Returns:
            Dictionary containing overfitting indicators:
            - r2_gap: R2 difference (train - test), positive indicates overfitting
            - rmse_ratio: Test RMSE / Train RMSE ratio
            - mape_ratio: Test MAPE / Train MAPE ratio
            - overfitting_score: Composite score (0=perfect, >0.1=overfitting)
            - generalization_score: Overall generalization ability
        """
        r2_gap = train_metrics['R2'] - test_metrics['R2']
        rmse_ratio = test_metrics['RMSE'] / train_metrics['RMSE'] if train_metrics['RMSE'] > 0 else 1.0
        mape_ratio = test_metrics['MAPE'] / train_metrics['MAPE'] if train_metrics['MAPE'] > 0 else 1.0
        mse_ratio = test_metrics['MSE'] / train_metrics['MSE'] if train_metrics['MSE'] > 0 else 1.0

        # Composite overfitting score
        # R2 gap > 0.05 indicates overfitting
        # RMSE ratio > 1.5 indicates overfitting
        overfitting_score = max(0, r2_gap) + max(0, rmse_ratio - 1.0) * 0.5 + max(0, mape_ratio - 1.0) * 0.1

        # Generalization score (higher is better, max 100)
        generalization_score = max(0, 100 - overfitting_score * 200)

        # Severity assessment
        if r2_gap > 0.05 or rmse_ratio > 1.5:
            severity = "HIGH" if r2_gap > 0.1 or rmse_ratio > 2.0 else "MODERATE"
        else:
            severity = "LOW"

        return {
            'r2_gap': r2_gap,
            'rmse_ratio': rmse_ratio,
            'mse_ratio': mse_ratio,
            'mape_ratio': mape_ratio,
            'overfitting_score': overfitting_score,
            'generalization_score': generalization_score,
            'severity': severity
        }

    def evaluate_overfitting(
        self,
        train_metrics: Dict[str, float],
        test_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Evaluate and log overfitting metrics.

        Args:
            train_metrics: Training set metrics
            test_metrics: Test set metrics

        Returns:
            Overfitting metrics dictionary
        """
        overfit = self.calculate_overfitting_metrics(train_metrics, test_metrics)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Overfitting Analysis")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"R2 Gap (Train - Test):     {overfit['r2_gap']:.6f}")
        self.logger.info(f"  - Threshold: >0.05 indicates overfitting")
        self.logger.info(f"RMSE Ratio (Test/Train):   {overfit['rmse_ratio']:.4f}")
        self.logger.info(f"  - Threshold: >1.5 indicates overfitting")
        self.logger.info(f"MSE Ratio (Test/Train):    {overfit['mse_ratio']:.4f}")
        self.logger.info(f"MAPE Ratio (Test/Train):   {overfit['mape_ratio']:.4f}")
        self.logger.info(f"{'-'*60}")
        self.logger.info(f"Overfitting Score:         {overfit['overfitting_score']:.4f}")
        self.logger.info(f"Generalization Score:      {overfit['generalization_score']:.2f}/100")
        self.logger.info(f"Severity:                  {overfit['severity']}")
        self.logger.info(f"{'='*60}")

        # Recommendations
        if overfit['severity'] == 'HIGH':
            self.logger.warning("Recommendations to reduce overfitting:")
            self.logger.warning("  1. Increase regularization (minibatch_frac, col_sample)")
            self.logger.warning("  2. Reduce n_estimators or learning_rate")
            self.logger.warning("  3. Add early stopping with validation set")
        elif overfit['severity'] == 'MODERATE':
            self.logger.info("Suggestions for improvement:")
            self.logger.info("  1. Fine-tune col_sample and minibatch_frac")
            self.logger.info("  2. Slightly reduce learning_rate")
        else:
            self.logger.info("Model generalizes well - no significant overfitting detected.")

        return overfit

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
            f"COV = {metrics['COV']:.4f}",
            f"μ_ξ = {metrics.get('mu_xi', 0):.4f}",
            f"±10% = {metrics.get('within_10pct', 0):.1f}%"
        ]
        return " | ".join(lines)
