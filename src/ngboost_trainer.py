"""
NGBoost model training module for residual distribution modeling.

This module provides NGBoostTrainer class for training probabilistic models
that capture the uncertainty in XGBoost predictions by modeling residuals.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import optuna
from sklearn.model_selection import cross_val_score

from evaluator import ModelEvaluator
from utils import ensure_dir, setup_logger

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# NGBoost imports
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal, Laplace, LogNormal
    NGBOOST_AVAILABLE = True
except ImportError:
    NGBOOST_AVAILABLE = False
    NGBRegressor = None
    Normal = None
    Laplace = None


class NGBoostTrainer:
    """
    NGBoost model trainer for residual distribution modeling.

    NGBoost models the conditional distribution of residuals given features,
    enabling prediction intervals with calibrated uncertainty estimates.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize NGBoostTrainer.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance

        Raises:
            ImportError: If ngboost is not installed
        """
        if not NGBOOST_AVAILABLE:
            raise ImportError(
                "ngboost is required but not installed. "
                "Install it with: pip install ngboost>=0.5.0"
            )

        self.config = config
        self.logger = logger or setup_logger(__name__)

        self.model: Optional['NGBRegressor'] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.evaluator = ModelEvaluator(self.logger)

        # Paths
        self.logs_dir = Path(config['paths']['logs_dir'])
        self.output_dir = Path(config['paths']['output_dir'])
        self.optuna_db_path = Path(config['paths'].get('optuna_db', 'logs/ngboost_optuna.db'))
        self.best_params_path = Path(config['paths'].get('best_params', 'logs/ngboost_best_params.json'))

        ensure_dir(self.logs_dir)
        ensure_dir(self.output_dir)

        # Optuna settings
        self.use_optuna = config.get('optuna', {}).get('use_optuna', False)
        self.n_trials = config.get('optuna', {}).get('n_trials', 100)
        self.timeout = config.get('optuna', {}).get('timeout', 3600)
        self.study_name = config.get('optuna', {}).get('study_name', 'ngboost_residual_optimization')
        self.direction = config.get('optuna', {}).get('direction', 'minimize')

        # Confidence level for prediction intervals
        self.confidence_level = config.get('confidence', {}).get('level', 0.97)

        # Distribution type
        self.distribution = config.get('distribution', 'Normal')

        # Calibration margin for split conformal prediction
        self.calib_margin: Optional[float] = None

    def _get_dist(self):
        """Get distribution class from config string."""
        dist_map = {
            'Normal': Normal,
            'Laplace': Laplace,
            'LogNormal': LogNormal
        }
        return dist_map.get(self.distribution, Normal)

    def _get_base_params(self) -> Dict[str, Any]:
        """Get base NGBoost parameters from config."""
        params = self.config.get('ngboost_params', {}).copy()
        params['Dist'] = self._get_dist()
        params['verbose'] = params.get('verbose', False)
        return params

    def _get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define Optuna search space for hyperparameters.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters
        """
        search_space = self.config.get('optuna_search_space', {})

        params = {
            'n_estimators': trial.suggest_int(
                'n_estimators',
                search_space.get('n_estimators', {}).get('low', 100),
                search_space.get('n_estimators', {}).get('high', 1000),
                step=search_space.get('n_estimators', {}).get('step', 50)
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate',
                search_space.get('learning_rate', {}).get('low', 0.001),
                search_space.get('learning_rate', {}).get('high', 0.1),
                log=search_space.get('learning_rate', {}).get('log', True)
            ),
            'minibatch_frac': trial.suggest_float(
                'minibatch_frac',
                search_space.get('minibatch_frac', {}).get('low', 0.5),
                search_space.get('minibatch_frac', {}).get('high', 1.0)
            ),
            'col_sample': trial.suggest_float(
                'col_sample',
                search_space.get('col_sample', {}).get('low', 0.5),
                search_space.get('col_sample', {}).get('high', 1.0)
            ),
            'Dist': self._get_dist(),
            'verbose': False
        }

        return params

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, cv: Any) -> float:
        """
        Optuna objective function for hyperparameter optimization.
        Optimizes for prediction interval coverage with width penalty.

        Args:
            trial: Optuna trial object
            X: Feature matrix
            y: Target vector (residuals)
            cv: Cross-validation splitter

        Returns:
            Mean CV objective score (to minimize)
        """
        params = self._get_search_space(trial)
        target_coverage = self.confidence_level

        scores = []
        for train_idx, val_idx in cv.split(X):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            try:
                model = NGBRegressor(**params)
                model.fit(X_train_fold, y_train_fold)

                # Predict interval on validation set
                dist = model.pred_dist(X_val_fold)
                alpha = (1 - target_coverage) / 2
                lower = dist.ppf(alpha)
                upper = dist.ppf(1 - alpha)

                coverage_rate = np.mean((y_val_fold >= lower) & (y_val_fold <= upper))
                interval_width = upper - lower
                mean_width_pct = np.mean(interval_width / (np.abs(y_val_fold) + 1e-8))

                # Objective: prioritize coverage target, then penalize width
                coverage_gap = max(0, target_coverage - coverage_rate)
                if coverage_gap > 0:
                    # Heavy penalty for not meeting coverage
                    score = coverage_gap * 100.0 + mean_width_pct * 0.1
                else:
                    # Once coverage is met, lightly penalize width to keep intervals tight
                    score = -coverage_rate * 1.0 + mean_width_pct * 2.0

                scores.append(score)
            except Exception as e:
                self.logger.debug(f"Trial failed in CV fold: {e}")
                return float('inf')

        return np.mean(scores)

    def optimize_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: Any,
        n_trials: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run Optuna hyperparameter optimization.

        Args:
            X: Feature matrix
            y: Target vector (residuals)
            cv: Cross-validation splitter
            n_trials: Number of trials (overrides config)

        Returns:
            Dictionary of best hyperparameters
        """
        n_trials = n_trials or self.n_trials

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Starting NGBoost Hyperparameter Optimization")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Number of trials: {n_trials}")
        self.logger.info(f"Timeout: {self.timeout} seconds")
        self.logger.info(f"Study name: {self.study_name}")

        # Create or load study
        storage_url = f"sqlite:///{self.optuna_db_path}"

        study = optuna.create_study(
            study_name=self.study_name,
            storage=storage_url,
            direction=self.direction,
            load_if_exists=True
        )

        # Run optimization
        study.optimize(
            lambda trial: self._objective(trial, X, y, cv),
            n_trials=n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        # Get best parameters
        self.best_params = study.best_params.copy()
        self.best_params.update({
            'Dist': self._get_dist(),
            'verbose': False
        })

        # Save best parameters
        self._save_best_params(self.best_params)

        # Log results
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Optimization Complete")
        self.logger.info(f"Best NLL: {study.best_value:.6f}")
        self.logger.info(f"Best Parameters:")
        for key, value in study.best_params.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info(f"{'='*60}")

        return self.best_params

    def _save_best_params(self, params: Dict[str, Any]) -> None:
        """Save best parameters to JSON file."""
        ensure_dir(self.best_params_path.parent)
        # Remove non-serializable objects (like Dist class)
        save_params = {k: v for k, v in params.items() if k != 'Dist'}
        with open(self.best_params_path, 'w') as f:
            json.dump(save_params, f, indent=2)
        self.logger.info(f"Saved best parameters to: {self.best_params_path}")

    def load_best_params(self) -> Optional[Dict[str, Any]]:
        """Load best parameters from JSON file if it exists."""
        if self.best_params_path.exists():
            with open(self.best_params_path, 'r') as f:
                params = json.load(f)
            params['Dist'] = self._get_dist()
            params['verbose'] = False
            self.logger.info(f"Loaded best parameters from: {self.best_params_path}")
            return params
        return None

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters based on configuration.

        Returns:
            Dictionary of model parameters
        """
        if self.use_optuna:
            loaded_params = self.load_best_params()
            if loaded_params:
                self.logger.info("Using previously optimized parameters")
                return loaded_params
            else:
                self.logger.warning("No optimized parameters found. Using defaults.")
                return self._get_base_params()
        else:
            loaded_params = self.load_best_params()
            if loaded_params:
                self.logger.info("Using best parameters from file")
                return loaded_params
            self.logger.info("Using default NGBoost parameters")
            return self._get_base_params()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> 'NGBRegressor':
        """
        Train NGBoost model on residuals.

        Args:
            X_train: Training features
            y_train: Training targets (residuals)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            params: Model parameters (optional)

        Returns:
            Trained NGBoost model
        """
        if params is None:
            params = self.get_model_params()

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Training NGBoost Model on Residuals")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Training samples: {len(X_train)}")
        self.logger.info(f"Parameters: {params}")

        # Create model
        self.model = NGBRegressor(**params)

        # Check for early stopping configuration
        early_stopping_config = self.config.get('early_stopping', {})
        use_early_stopping = early_stopping_config.get('use_early_stopping', False)

        # Train model
        if X_val is not None and y_val is not None:
            if use_early_stopping:
                n_early = early_stopping_config.get('n_early_stopping_rounds', 50)
                self.model.fit(X_train, y_train, X_val=X_val, Y_val=y_val, early_stopping_rounds=n_early)
                self.logger.info(f"Training completed with early stopping (patience={n_early})")
            else:
                self.model.fit(X_train, y_train, X_val=X_val, Y_val=y_val)
                self.logger.info("Training completed with validation set")
        else:
            self.model.fit(X_train, y_train)
            self.logger.info("Training completed")

        return self.model

    def calibrate_intervals(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        confidence: Optional[float] = None
    ) -> float:
        """
        Calibrate prediction intervals using split conformal prediction.

        Computes a symmetric margin to add to both sides of the predicted
        interval such that the empirical coverage on the validation set
        matches the target confidence level.

        Args:
            X_val: Validation features
            y_val: Validation targets (residuals)
            confidence: Target confidence level (default: self.confidence_level)

        Returns:
            Calibration margin
        """
        confidence = confidence or self.confidence_level

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Calibrating Prediction Intervals (Split Conformal)")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Calibration samples: {len(y_val)}")
        self.logger.info(f"Target confidence: {confidence:.2%}")

        # Get uncalibrated intervals
        lower, upper = self.predict_interval(X_val, confidence)

        # Non-conformity scores: distance outside the interval
        scores = np.maximum(lower - y_val, y_val - upper)

        # Compute the required margin (quantile of scores)
        # Using higher method ensures conservative coverage
        q = float(np.quantile(scores, confidence, method='higher'))
        self.calib_margin = q

        # Evaluate calibrated coverage on validation set
        lower_cal = lower - q
        upper_cal = upper + q
        coverage = np.mean((y_val >= lower_cal) & (y_val <= upper_cal))
        mean_width = np.mean(upper_cal - lower_cal)

        self.logger.info(f"Calibration margin: {q:.4f}")
        self.logger.info(f"Calibrated validation coverage: {coverage:.2%}")
        self.logger.info(f"Calibrated mean width: {mean_width:.4f}")
        self.logger.info(f"{'='*60}")

        return q

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make point predictions (mean of predicted distribution).

        Args:
            X: Feature matrix

        Returns:
            Mean predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def predict_params(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict distribution parameters (μ, σ) for residuals.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (mu, sigma) arrays
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        dist = self.model.pred_dist(X)
        mu = dist.params['loc']
        sigma = dist.params['scale']
        return mu, sigma

    def predict_interval(
        self,
        X: np.ndarray,
        confidence: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict confidence interval for residuals.

        Args:
            X: Feature matrix
            confidence: Confidence level (default: self.confidence_level)

        Returns:
            Tuple of (lower_bound, upper_bound) arrays
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        confidence = confidence or self.confidence_level
        alpha = (1 - confidence) / 2

        dist = self.model.pred_dist(X)
        lower = dist.ppf(alpha)
        upper = dist.ppf(1 - alpha)

        # Apply split conformal calibration margin if available
        if self.calib_margin is not None:
            lower = lower - self.calib_margin
            upper = upper + self.calib_margin

        return lower, upper

    def calculate_coverage(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        confidence: Optional[float] = None
    ) -> float:
        """
        Calculate coverage rate of prediction intervals.

        Args:
            X: Feature matrix
            y_true: True values (residuals)
            confidence: Confidence level (default: self.confidence_level)

        Returns:
            Coverage rate (0 to 1)
        """
        lower, upper = self.predict_interval(X, confidence)
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        return float(coverage)

    def save_model(self, filepath: Optional[str] = None) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save model. If None, uses config path.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if filepath is None:
            filepath = self.config['paths'].get('ngboost_model', 'output/ngboost_residual_model.pkl')

        filepath = Path(filepath)
        ensure_dir(filepath.parent)

        # Save model and config
        model_package = {
            'model': self.model,
            'config': self.config,
            'confidence_level': self.confidence_level,
            'calib_margin': self.calib_margin
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)

        self.logger.info(f"Saved NGBoost model to: {filepath}")

    def load_model(self, filepath: Optional[str] = None) -> 'NGBRegressor':
        """
        Load model from file.

        Args:
            filepath: Path to model file. If None, uses config path.

        Returns:
            Loaded NGBoost model
        """
        if filepath is None:
            filepath = self.config['paths'].get('ngboost_model', 'output/ngboost_residual_model.pkl')

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)

        self.model = model_package['model']
        loaded_config = model_package.get('config', self.config)
        self.confidence_level = model_package.get('confidence_level', 0.97)
        self.calib_margin = model_package.get('calib_margin', None)
        if self.calib_margin is not None:
            self.logger.info(f"Loaded calibration margin: {self.calib_margin:.4f}")

        self.logger.info(f"Loaded NGBoost model from: {filepath}")
        return self.model
