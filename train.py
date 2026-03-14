#!/usr/bin/env python3
"""
Main training script for XGBoost model on CFDST data.

Usage:
    python train.py --config config/config.yaml
    python train.py --config config/config.yaml --optimize
    python train.py --config config/config.yaml --no-optimize
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
from sklearn.model_selection import KFold

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import XGBoostTrainer
from src.evaluator import ModelEvaluator
from src.visualizer import ModelVisualizer
from src.utils import load_config, setup_logger, ensure_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train XGBoost model for CFDST column strength prediction"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip Optuna optimization (use default or saved params)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides config)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data CSV file (overrides config)"
    )
    return parser.parse_args()


def main():
    """Main training pipeline."""
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    log_file = config.get('logging', {}).get('file', 'logs/training.log')
    logger = setup_logger(
        "train",
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=log_file
    )

    logger.info("=" * 60)
    logger.info("XGBoost Training Pipeline for CFDST Column Strength Prediction")
    logger.info("=" * 60)

    # Override config with command line args
    if args.optimize:
        config['optuna']['use_optuna'] = True
    if args.no_optimize:
        config['optuna']['use_optuna'] = False

    # Create output directories
    ensure_dir(config['paths']['logs_dir'])
    ensure_dir(config['paths']['output_dir'])

    # Step 1: Load Data
    logger.info("\n" + "-" * 40)
    logger.info("Step 1: Loading Data")
    logger.info("-" * 40)

    data_loader = DataLoader(config, logger)
    data_path = args.data or config['paths']['raw_data']
    df, feature_cols = data_loader.load_and_prepare(data_path)

    # Step 2: Preprocess Data
    logger.info("\n" + "-" * 40)
    logger.info("Step 2: Preprocessing Data")
    logger.info("-" * 40)

    preprocessor = DataPreprocessor(config, logger)
    df_processed, feature_cols = preprocessor.preprocess(df, feature_cols, fit=True)

    # Save processed data
    data_loader.save_processed_data(df_processed)

    # Step 3: Split Data
    logger.info("\n" + "-" * 40)
    logger.info("Step 3: Splitting Data")
    logger.info("-" * 40)

    # Check if log transform is enabled
    use_log_transform = config['preprocessing'].get('log_transform_target', False)
    if use_log_transform:
        logger.info("Log transform enabled for target variable (Nexp)")

    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df_processed, feature_cols, apply_log_transform=True
    )

    # Step 4: Hyperparameter Optimization (Optuna)
    trainer = XGBoostTrainer(config, logger)

    if config['optuna']['use_optuna']:
        logger.info("\n" + "-" * 40)
        logger.info("Step 4: Hyperparameter Optimization")
        logger.info("-" * 40)

        cv = KFold(
            n_splits=config['cross_validation']['n_splits'],
            shuffle=config['cross_validation']['shuffle'],
            random_state=config['cross_validation']['random_state']
        )

        n_trials = args.n_trials or config['optuna']['n_trials']
        best_params = trainer.optimize_hyperparameters(X_train, y_train, cv, n_trials)
    else:
        logger.info("\n" + "-" * 40)
        logger.info("Step 4: Using Existing/Pre-configured Parameters")
        logger.info("-" * 40)
        best_params = trainer.get_model_params()

    # Step 5: Train Final Model
    logger.info("\n" + "-" * 40)
    logger.info("Step 5: Training Final Model")
    logger.info("-" * 40)

    model = trainer.train(X_train, y_train, X_test, y_test, params=best_params)

    # Step 6: Evaluate Model
    logger.info("\n" + "-" * 40)
    logger.info("Step 6: Evaluating Model")
    logger.info("-" * 40)

    evaluator = ModelEvaluator(logger)
    visualizer = ModelVisualizer(config, logger)

    # Train predictions (in log space if log transform enabled)
    y_train_pred_log = trainer.predict(X_train)

    # Test predictions (in log space if log transform enabled)
    y_test_pred_log = trainer.predict(X_test)

    # Inverse transform predictions back to original scale if log transform was used
    if use_log_transform:
        logger.info("Converting predictions from log space to original scale")
        y_train_pred = preprocessor.inverse_transform_target(y_train_pred_log)
        y_test_pred = preprocessor.inverse_transform_target(y_test_pred_log)
        # Also need to inverse transform the actual y values for evaluation
        y_train_orig = preprocessor.inverse_transform_target(y_train)
        y_test_orig = preprocessor.inverse_transform_target(y_test)
    else:
        y_train_pred = y_train_pred_log
        y_test_pred = y_test_pred_log
        y_train_orig = y_train
        y_test_orig = y_test

    train_metrics = evaluator.evaluate(y_train_orig, y_train_pred, "Train")
    test_metrics = evaluator.evaluate(y_test_orig, y_test_pred, "Test")

    # Step 7: Generate Visualizations
    logger.info("\n" + "-" * 40)
    logger.info("Step 7: Generating Visualizations")
    logger.info("-" * 40)

    output_dir = Path(config['paths']['output_dir'])

    # Prediction scatter plots
    visualizer.plot_predictions_vs_actual(
        y_train_orig, y_train_pred, "Train",
        save_path=output_dir / "predictions_train.png",
        metrics=train_metrics
    )
    visualizer.plot_predictions_vs_actual(
        y_test_orig, y_test_pred, "Test",
        save_path=output_dir / "predictions_test.png",
        metrics=test_metrics
    )

    # Feature importance
    feature_importance = trainer.get_feature_importance(feature_cols)
    visualizer.plot_feature_importance(
        list(feature_importance.keys()),
        np.array(list(feature_importance.values())),
        title="XGBoost Feature Importance",
        save_path=output_dir / "feature_importance.png"
    )

    # Residual analysis
    visualizer.plot_residuals(
        y_test_orig, y_test_pred, "Test",
        save_path=output_dir / "residuals_test.png"
    )

    # Ratio analysis (new visualization)
    visualizer.plot_ratio_analysis(
        y_test_orig, y_test_pred, "Test",
        save_path=output_dir / "ratio_analysis_test.png"
    )
    visualizer.plot_ratio_analysis(
        y_train_orig, y_train_pred, "Train",
        save_path=output_dir / "ratio_analysis_train.png"
    )

    # Correlation matrix
    visualizer.plot_correlation_matrix(
        df_processed, feature_cols,
        save_path=output_dir / "correlation_matrix.png"
    )

    # Step 8: Save Model
    logger.info("\n" + "-" * 40)
    logger.info("Step 8: Saving Model")
    logger.info("-" * 40)

    # Get preprocessor state for saving (needed for inverse transform in prediction)
    preprocessor_state = preprocessor.get_preprocessor_state()
    trainer.save_model(preprocessor_state=preprocessor_state)

    # Save feature list
    feature_file = output_dir / "feature_names.txt"
    with open(feature_file, 'w') as f:
        f.write("\n".join(feature_cols))
    logger.info(f"Saved feature names to: {feature_file}")

    # Save metrics summary
    metrics_file = output_dir / "metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Training Metrics\n")
        f.write("=" * 50 + "\n")
        for key, value in train_metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("=" * 50 + "\n")
        f.write("Test Metrics\n")
        f.write("=" * 50 + "\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

        # Add log transform info if used
        if use_log_transform:
            f.write("\n")
            f.write("=" * 50 + "\n")
            f.write("Log Transform Configuration\n")
            f.write("=" * 50 + "\n")
            f.write(f"log_transform_target: {use_log_transform}\n")
            f.write(f"log_transform_epsilon: {config['preprocessing'].get('log_transform_epsilon', 1.0)}\n")

    logger.info(f"Saved metrics to: {metrics_file}")

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {config['paths']['model_file']}")
    logger.info(f"Outputs saved to: {config['paths']['output_dir']}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
