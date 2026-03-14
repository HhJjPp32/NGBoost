"""
XGBoost model training module with Optuna hyperparameter optimization.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score

from .evaluator import ModelEvaluator
from .utils import ensure_dir, setup_logger

# Suppress Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBoostTrainer:
    """
    XGBoost model trainer with Optuna hyperparameter optimization support.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize XGBoostTrainer.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)

        self.model: Optional[xgb.XGBRegressor] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.evaluator = ModelEvaluator(self.logger)

        # Paths
        self.logs_dir = Path(config['paths']['logs_dir'])
        self.output_dir = Path(config['paths']['output_dir'])
        self.optuna_db_path = Path(config['paths']['optuna_db'])
        self.best_params_path = Path(config['paths']['best_params'])

        ensure_dir(self.logs_dir)
        ensure_dir(self.output_dir)

        # Optuna settings
        self.use_optuna = config['optuna']['use_optuna']
        self.n_trials = config['optuna']['n_trials']
        self.timeout = config['optuna']['timeout']
        self.study_name = config['optuna']['study_name']
        self.direction = config['optuna']['direction']

    def _get_base_params(self) -> Dict[str, Any]:
        """Get base XGBoost parameters from config."""
        return self.config['xgboost_params'].copy()

    def _get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define Optuna search space for hyperparameters.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of hyperparameters
        """
        search_space = self.config['optuna_search_space']

        params = {
            'n_estimators': trial.suggest_int(
                'n_estimators',
                search_space['n_estimators']['low'],
                search_space['n_estimators']['high'],
                step=search_space['n_estimators'].get('step', 1)
            ),
            'max_depth': trial.suggest_int(
                'max_depth',
                search_space['max_depth']['low'],
                search_space['max_depth']['high']
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate',
                search_space['learning_rate']['low'],
                search_space['learning_rate']['high'],
                log=search_space['learning_rate'].get('log', False)
            ),
            'subsample': trial.suggest_float(
                'subsample',
                search_space['subsample']['low'],
                search_space['subsample']['high']
            ),
            'colsample_bytree': trial.suggest_float(
                'colsample_bytree',
                search_space['colsample_bytree']['low'],
                search_space['colsample_bytree']['high']
            ),
            'min_child_weight': trial.suggest_int(
                'min_child_weight',
                search_space['min_child_weight']['low'],
                search_space['min_child_weight']['high']
            ),
            'gamma': trial.suggest_float(
                'gamma',
                search_space['gamma']['low'],
                search_space['gamma']['high']
            ),
            'reg_alpha': trial.suggest_float(
                'reg_alpha',
                search_space['reg_alpha']['low'],
                search_space['reg_alpha']['high'],
                log=search_space['reg_alpha'].get('log', False)
            ),
            'reg_lambda': trial.suggest_float(
                'reg_lambda',
                search_space['reg_lambda']['low'],
                search_space['reg_lambda']['high'],
                log=search_space['reg_lambda'].get('log', False)
            ),
            'objective': 'reg:squarederror',
            'random_state': self.config['xgboost_params']['random_state'],
            'n_jobs': -1,
            'verbosity': 0
        }

        return params

    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray, cv: Any) -> float:
        """
        Optuna objective function for hyperparameter optimization.

        Args:
            trial: Optuna trial object
            X: Feature matrix
            y: Target vector
            cv: Cross-validation splitter

        Returns:
            Mean CV RMSE (to minimize)
        """
        params = self._get_search_space(trial)

        model = xgb.XGBRegressor(**params)

        # Use negative RMSE for scoring (sklearn convention)
        scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )

        # Return mean RMSE (positive, for minimization)
        return -scores.mean()

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
            y: Target vector
            cv: Cross-validation splitter
            n_trials: Number of trials (overrides config)

        Returns:
            Dictionary of best hyperparameters
        """
        n_trials = n_trials or self.n_trials

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Starting Optuna Hyperparameter Optimization")
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
            'objective': 'reg:squarederror',
            'random_state': self.config['xgboost_params']['random_state'],
            'n_jobs': -1,
            'verbosity': 0
        })

        # Save best parameters
        self._save_best_params(self.best_params)

        # Log results
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Optimization Complete")
        self.logger.info(f"Best RMSE: {study.best_value:.6f}")
        self.logger.info(f"Best Parameters:")
        for key, value in study.best_params.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info(f"{'='*60}")

        return self.best_params

    def _save_best_params(self, params: Dict[str, Any]) -> None:
        """Save best parameters to JSON file."""
        ensure_dir(self.best_params_path.parent)
        with open(self.best_params_path, 'w') as f:
            json.dump(params, f, indent=2)
        self.logger.info(f"Saved best parameters to: {self.best_params_path}")

    def load_best_params(self) -> Optional[Dict[str, Any]]:
        """Load best parameters from JSON file if it exists."""
        if self.best_params_path.exists():
            with open(self.best_params_path, 'r') as f:
                params = json.load(f)
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
            # Try to load existing best params
            loaded_params = self.load_best_params()
            if loaded_params:
                self.logger.info("Using previously optimized parameters")
                return loaded_params
            else:
                self.logger.warning("No optimized parameters found. Run optimization first.")
                self.logger.info("Falling back to default parameters")
                return self._get_base_params()
        else:
            # Try to load best params, otherwise use defaults
            loaded_params = self.load_best_params()
            if loaded_params:
                self.logger.info("Using best parameters from file")
                return loaded_params
            self.logger.info("Using default XGBoost parameters")
            return self._get_base_params()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> xgb.XGBRegressor:
        """
        Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            params: Model parameters (optional, uses config if not provided)

        Returns:
            Trained XGBoost model
        """
        if params is None:
            params = self.get_model_params()

        self.logger.info(f"\n{'='*60}")
        self.logger.info("Training XGBoost Model")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Training samples: {len(X_train)}")
        self.logger.info(f"Parameters: {params}")

        # Create model
        self.model = xgb.XGBRegressor(**params)

        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))

        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )

        self.logger.info("Training completed")

        return self.model

    def train_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: Any,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[xgb.XGBRegressor, Dict[str, Any]]:
        """
        Train model on full dataset after cross-validation.

        Args:
            X: Feature matrix
            y: Target vector
            cv: Cross-validation splitter
            params: Model parameters (optional)

        Returns:
            Tuple of (trained model, CV metrics)
        """
        if params is None:
            params = self.get_model_params()

        # Run cross-validation
        self.logger.info("\nRunning cross-validation...")
        cv_results = self.evaluator.cross_validate(
            xgb.XGBRegressor(**params), X, y, cv
        )

        # Train on full dataset
        model = self.train(X, y, params=params)

        return model, cv_results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with trained model.

        Args:
            X: Feature matrix

        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def get_feature_importance(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Get feature importance from trained model.

        Args:
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.feature_importances_

        return {name: float(imp) for name, imp in zip(feature_names, importance)}

    def save_model(self, filepath: Optional[str] = None, preprocessor_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Save trained model to file.

        Args:
            filepath: Path to save model. If None, uses config path.
            preprocessor_state: Optional preprocessor state dict (for inverse transforms)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if filepath is None:
            filepath = self.config['paths']['model_file']

        filepath = Path(filepath)
        ensure_dir(filepath.parent)

        # Save both model and preprocessor state
        model_package = {
            'model': self.model,
            'config': self.config,
            'preprocessor_state': preprocessor_state
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_package, f)

        self.logger.info(f"Saved model to: {filepath}")

    def load_model(self, filepath: Optional[str] = None) -> Tuple[xgb.XGBRegressor, Optional[Dict[str, Any]]]:
        """
        Load model from file.

        Args:
            filepath: Path to model file. If None, uses config path.

        Returns:
            Tuple of (loaded XGBoost model, preprocessor_state dict or None)
        """
        if filepath is None:
            filepath = self.config['paths']['model_file']

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, 'rb') as f:
            model_package = pickle.load(f)

        # Handle both old format (just model) and new format (dict with model + state)
        if isinstance(model_package, dict):
            self.model = model_package['model']
            preprocessor_state = model_package.get('preprocessor_state')
            loaded_config = model_package.get('config', self.config)
            self.logger.info(f"Loaded model (with preprocessor state) from: {filepath}")
            # Update config if loaded from package
            if 'preprocessing' in loaded_config and 'log_transform_target' in loaded_config['preprocessing']:
                self.logger.info(f"Model uses log_transform_target: {loaded_config['preprocessing']['log_transform_target']}")
            return self.model, preprocessor_state
        else:
            # Old format - just the model
            self.model = model_package
            self.logger.info(f"Loaded model (legacy format) from: {filepath}")
            return self.model, None
