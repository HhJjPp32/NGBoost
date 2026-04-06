#!/usr/bin/env python3
"""
Prediction script with confidence intervals using XGBoost + NGBoost ensemble.

This script combines:
- XGBoost for point prediction (mean)
- NGBoost for uncertainty quantification (residual distribution)

Output includes:
- Point prediction
- 97% prediction interval [lower, upper]
- Residual distribution parameters (μ, σ)

Usage:
    python predict_with_interval.py --input data/new_data.csv --output predictions.csv
    python predict_with_interval.py --input data/new_data.csv --confidence 0.99
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import XGBoostTrainer
from src.ngboost_trainer import NGBoostTrainer
from src.utils import load_config, setup_logger, ensure_dir


class EnsemblePredictor:
    """
    Two-stage ensemble predictor combining XGBoost and NGBoost.

    XGBoost provides point predictions, NGBoost provides uncertainty estimates.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        xgb_model_path: Optional[str] = None,
        ngboost_model_path: Optional[str] = None,
        logger=None
    ):
        """
        Initialize ensemble predictor.

        Args:
            config: Configuration dictionary
            xgb_model_path: Path to XGBoost model (optional)
            ngboost_model_path: Path to NGBoost model (optional)
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or setup_logger("ensemble_predictor")

        # Initialize trainers
        self.xgb_trainer = XGBoostTrainer(config, self.logger)
        self.ngboost_trainer = NGBoostTrainer(config, self.logger)

        # Load models
        self._load_models(xgb_model_path, ngboost_model_path)

        # Get confidence level
        self.confidence_level = config.get('confidence', {}).get('level', 0.97)

    def _load_models(
        self,
        xgb_model_path: Optional[str] = None,
        ngboost_model_path: Optional[str] = None
    ):
        """Load both XGBoost and NGBoost models."""
        # Load XGBoost model
        xgb_path = xgb_model_path or self.config['paths']['xgboost_model']
        self.logger.info(f"Loading XGBoost model from: {xgb_path}")
        self.xgb_model, self.preprocessor_state = self.xgb_trainer.load_model(xgb_path)

        # Load NGBoost model
        ngboost_path = ngboost_model_path or self.config['paths']['ngboost_model']
        self.logger.info(f"Loading NGBoost model from: {ngboost_path}")
        self.ngboost_model = self.ngboost_trainer.load_model(ngboost_path)

        self.logger.info("Models loaded successfully")

    def predict_with_interval(
        self,
        X: np.ndarray,
        confidence: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence intervals.

        Args:
            X: Feature matrix
            confidence: Confidence level (default: self.confidence_level)

        Returns:
            Dictionary with keys:
            - 'point_prediction': XGBoost point prediction
            - 'mean_prediction': XGBoost + NGBoost mean adjustment
            - 'lower_bound': Lower bound of prediction interval
            - 'upper_bound': Upper bound of prediction interval
            - 'residual_mu': Mean of residual distribution
            - 'residual_sigma': Std of residual distribution
            - 'interval_width': Width of prediction interval
        """
        confidence = confidence or self.confidence_level

        # Stage 1: XGBoost point prediction
        y_pred_log = self.xgb_trainer.predict(X)

        # Stage 2: NGBoost residual distribution
        mu_epsilon, sigma_epsilon = self.ngboost_trainer.predict_params(X)
        epsilon_lower, epsilon_upper = self.ngboost_trainer.predict_interval(X, confidence)

        # Check if log transform was used
        use_log_transform = self.config['preprocessing'].get('log_transform_target', False)

        if use_log_transform:
            # Transform from log space to original scale
            epsilon = self.config['preprocessing'].get('log_transform_epsilon', 1.0)

            # XGBoost prediction in original scale
            y_pred_orig = np.exp(y_pred_log) - epsilon

            # Final mean prediction (with NGBoost adjustment)
            y_mean_orig = y_pred_orig + mu_epsilon

            # Prediction intervals in original scale
            y_lower = y_pred_orig + epsilon_lower
            y_upper = y_pred_orig + epsilon_upper
        else:
            # Original scale predictions
            y_pred_orig = y_pred_log
            y_mean_orig = y_pred_orig + mu_epsilon
            y_lower = y_pred_orig + epsilon_lower
            y_upper = y_pred_orig + epsilon_upper

        # Calculate interval width
        interval_width = y_upper - y_lower

        return {
            'point_prediction': y_pred_orig,
            'mean_prediction': y_mean_orig,
            'lower_bound': y_lower,
            'upper_bound': y_upper,
            'residual_mu': mu_epsilon,
            'residual_sigma': sigma_epsilon,
            'interval_width': interval_width,
            'confidence_level': confidence
        }

    def predict_from_csv(
        self,
        csv_path: str,
        output_path: Optional[str] = None,
        preprocess: bool = True,
        confidence: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Make predictions from CSV file.

        Args:
            csv_path: Path to input CSV
            output_path: Path to save results (optional)
            preprocess: Whether to apply preprocessing
            confidence: Confidence level

        Returns:
            DataFrame with predictions and intervals
        """
        # Load data
        data_loader = DataLoader(self.config, self.logger)
        df = data_loader.load_csv(csv_path)
        df = data_loader.clean_column_names(df)

        # Get feature columns (same as training)
        feature_cols = [
            col for col in df.columns
            if col != self.config['target']['name']
            and col not in self.config.get('drop_columns', [])
        ]

        # Preprocess if needed
        if preprocess:
            preprocessor = DataPreprocessor(self.config, self.logger)
            df_processed, feature_cols = preprocessor.preprocess(df, feature_cols, fit=False)
            # Use preprocessor state from XGBoost model if available
            if self.preprocessor_state:
                preprocessor.load_preprocessor_state(self.preprocessor_state)
            X = df_processed[feature_cols].values
        else:
            X = df[feature_cols].values

        # Make predictions
        results = self.predict_with_interval(X, confidence)

        # Create results DataFrame
        results_df = pd.DataFrame({
            'predicted_Nexp': results['mean_prediction'],
            'lower_bound_97': results['lower_bound'],
            'upper_bound_97': results['upper_bound'],
            'xgboost_prediction': results['point_prediction'],
            'residual_mu': results['residual_mu'],
            'residual_sigma': results['residual_sigma'],
            'interval_width': results['interval_width']
        })

        # Add input features for reference
        for col in feature_cols:
            if col in df.columns:
                results_df[f'input_{col}'] = df[col].values

        # Save to CSV if path provided
        if output_path:
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.logger.info(f"Saved predictions to: {output_path}")

        return results_df


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions with confidence intervals"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input CSV file with features"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="predictions_with_intervals.csv",
        help="Path to save predictions CSV"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config_ngboost.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence level (default: 0.97 from config)"
    )
    parser.add_argument(
        "--xgboost-model",
        type=str,
        default=None,
        help="Path to XGBoost model (overrides config)"
    )
    parser.add_argument(
        "--ngboost-model",
        type=str,
        default=None,
        help="Path to NGBoost model (overrides config)"
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Skip preprocessing (data already preprocessed)"
    )
    return parser.parse_args()


def main():
    """Main prediction pipeline."""
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    logger = setup_logger("predict", level="INFO")

    logger.info("=" * 60)
    logger.info("Ensemble Prediction with Confidence Intervals")
    logger.info("=" * 60)

    # Initialize predictor
    try:
        predictor = EnsemblePredictor(
            config,
            xgb_model_path=args.xgboost_model,
            ngboost_model_path=args.ngboost_model,
            logger=logger
        )
    except FileNotFoundError as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Please train models first:")
        logger.error("  1. python train.py (for XGBoost)")
        logger.error("  2. python train_ngboost_residual.py (for NGBoost)")
        return 1

    # Get confidence level
    confidence = args.confidence or config.get('confidence', {}).get('level', 0.97)
    logger.info(f"\nConfidence level: {confidence:.0%}")

    # Make predictions
    logger.info(f"\nLoading input data from: {args.input}")

    try:
        results_df = predictor.predict_from_csv(
            csv_path=args.input,
            output_path=args.output,
            preprocess=not args.no_preprocess,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    # Display summary
    logger.info("\n" + "-" * 40)
    logger.info("Prediction Summary")
    logger.info("-" * 40)
    logger.info(f"Total samples: {len(results_df)}")
    logger.info(f"\nPoint Predictions:")
    logger.info(f"  Mean: {results_df['predicted_Nexp'].mean():.4f}")
    logger.info(f"  Min:  {results_df['predicted_Nexp'].min():.4f}")
    logger.info(f"  Max:  {results_df['predicted_Nexp'].max():.4f}")
    logger.info(f"\n{confidence:.0%} Prediction Intervals:")
    logger.info(f"  Mean Width: {results_df['interval_width'].mean():.4f}")
    logger.info(f"  Mean Width / Prediction: "
                f"{np.mean(results_df['interval_width'] / results_df['predicted_Nexp']) * 100:.2f}%")

    logger.info("\n" + "=" * 60)
    logger.info("Prediction Complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
