"""
Prediction module for making inference with trained models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .data_loader import DataLoader
from .model_trainer import XGBoostTrainer
from .preprocessor import DataPreprocessor
from .utils import setup_logger


class Predictor:
    """
    Handles model inference/prediction on new data.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize Predictor.

        Args:
            config: Configuration dictionary
            model_path: Path to saved model. If None, uses config path.
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)

        self.model_path = model_path or config['paths']['model_file']
        self.target_col = config['target']['name']

        # Initialize components
        self.trainer = XGBoostTrainer(config, self.logger)
        self.preprocessor = DataPreprocessor(config, self.logger)
        self.data_loader = DataLoader(config, self.logger)

        # Load model and preprocessor state
        self.preprocessor_state: Optional[Dict[str, Any]] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load trained model and preprocessor state."""
        try:
            _, preprocessor_state = self.trainer.load_model(self.model_path)
            self.preprocessor_state = preprocessor_state

            # Restore preprocessor state if available
            if preprocessor_state:
                self.preprocessor.set_preprocessor_state(preprocessor_state)
                self.logger.info("Preprocessor state restored (includes log transform settings)")

            self.logger.info(f"Model loaded successfully from: {self.model_path}")
        except FileNotFoundError:
            self.logger.error(f"Model file not found: {self.model_path}")
            raise

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Make predictions on input data.

        Automatically applies inverse transform if model was trained with log transform.

        Args:
            X: Input features (numpy array or DataFrame)
            feature_names: List of feature names (required if X is numpy array)

        Returns:
            Predictions as numpy array (in original scale)
        """
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.logger.info(f"Making predictions on {len(X)} samples")

        # Make predictions (in log space if log transform was used)
        predictions_log = self.trainer.predict(X)

        # Apply inverse transform if needed
        if self.preprocessor.log_transform_target:
            predictions = self.preprocessor.inverse_transform_target(predictions_log)
            self.logger.info("Applied inverse log transform to predictions")
        else:
            predictions = predictions_log

        return predictions

    def predict_from_csv(
        self,
        csv_path: str,
        output_path: Optional[str] = None,
        preprocess: bool = True
    ) -> pd.DataFrame:
        """
        Make predictions from CSV file.

        Args:
            csv_path: Path to input CSV file
            output_path: Optional path to save predictions
            preprocess: Whether to apply preprocessing

        Returns:
            DataFrame with predictions
        """
        # Load data
        df, feature_cols = self.data_loader.load_and_prepare(csv_path)

        # Preprocess if needed
        if preprocess:
            df, feature_cols = self.preprocessor.preprocess(df, feature_cols, fit=False)

        # Get features
        X = df[feature_cols].values

        # Make predictions
        predictions = self.predict(X, feature_cols)

        # Add predictions to DataFrame
        df['predicted_Nexp'] = predictions

        # Save if path provided
        if output_path:
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"Predictions saved to: {output_path}")

        return df

    def predict_with_confidence(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        feature_names: Optional[List[str]] = None,
        confidence_method: str = "interval"
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence estimates.

        Args:
            X: Input features
            feature_names: List of feature names
            confidence_method: Method for confidence estimation

        Returns:
            Dictionary with predictions and confidence intervals
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get base predictions
        predictions = self.trainer.predict(X)

        result = {
            'predictions': predictions,
            'lower_bound': None,
            'upper_bound': None
        }

        # Simple confidence interval based on MAPE from config if available
        # This is a placeholder - in practice, you might use quantile regression
        # or bootstrap methods
        mape = self.config.get('evaluation', {}).get('mape', 10.0)  # Default 10%

        if confidence_method == "interval":
            margin = predictions * (mape / 100) * 1.96  # 95% CI approximation
            result['lower_bound'] = predictions - margin
            result['upper_bound'] = predictions + margin

        return result

    def batch_predict(
        self,
        data_list: List[Union[np.ndarray, pd.DataFrame]],
        batch_size: int = 1000
    ) -> List[np.ndarray]:
        """
        Make predictions on multiple batches of data.

        Args:
            data_list: List of data batches
            batch_size: Size of each batch (for progress reporting)

        Returns:
            List of prediction arrays
        """
        results = []
        total_samples = sum(len(d) for d in data_list)

        self.logger.info(f"Batch prediction on {total_samples} total samples")

        for i, batch in enumerate(data_list):
            preds = self.predict(batch)
            results.append(preds)
            self.logger.debug(f"Processed batch {i+1}/{len(data_list)}")

        return results

    def evaluate_on_new_data(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on new labeled data.

        Args:
            X: Feature matrix
            y_true: Ground truth values
            sample_weights: Optional sample weights

        Returns:
            Dictionary of evaluation metrics
        """
        from .evaluator import ModelEvaluator

        evaluator = ModelEvaluator(self.logger)

        # Make predictions
        y_pred = self.predict(X)

        # Calculate metrics
        metrics = evaluator.evaluate(y_true, y_pred, "New Data")

        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model information
        """
        if self.trainer.model is None:
            return {"status": "No model loaded"}

        model = self.trainer.model

        info = {
            "status": "Model loaded",
            "model_type": type(model).__name__,
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "learning_rate": model.learning_rate,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "min_child_weight": model.min_child_weight,
            "reg_alpha": model.reg_alpha,
            "reg_lambda": model.reg_lambda,
        }

        return info
