#!/usr/bin/env python3
"""
Feature Selection Pipeline using Recursive Feature Elimination (RFE).

This script implements an iterative feature elimination process to find
the optimal subset of features that balances model performance and simplicity.

Usage:
    python feature_selection.py --config config/config.yaml
    python feature_selection.py --config config/config.yaml --min-features 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
import xgboost as xgb

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
        description="Feature Selection using Recursive Feature Elimination"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=None,
        help="Minimum number of features to keep (overrides config)"
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Number of features to eliminate at each iteration"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results"
    )
    return parser.parse_args()


def evaluate_feature_subset(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: List[int],
    feature_names: List[str],
    params: Dict[str, Any],
    cv: Any,
    logger
) -> Dict[str, Any]:
    """
    Evaluate model performance with a subset of features.

    Args:
        X: Full feature matrix
        y: Target vector
        feature_indices: Indices of features to use
        feature_names: List of all feature names
        params: XGBoost parameters
        cv: Cross-validation splitter
        logger: Logger instance

    Returns:
        Dictionary with evaluation results
    """
    # Select features
    X_subset = X[:, feature_indices]
    selected_names = [feature_names[i] for i in feature_indices]

    # Create model
    model = xgb.XGBRegressor(**params)

    # Cross-validation
    evaluator = ModelEvaluator(logger)
    metrics = evaluator.cross_validate(model, X_subset, y, cv)

    return {
        'n_features': len(feature_indices),
        'feature_names': selected_names,
        'feature_indices': feature_indices,
        'metrics': metrics
    }


def recursive_feature_elimination(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    params: Dict[str, Any],
    config: Dict[str, Any],
    logger
) -> List[Dict[str, Any]]:
    """
    Perform recursive feature elimination.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        params: XGBoost parameters
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        List of results for each iteration
    """
    min_features = config['feature_selection']['min_features']
    step = config['feature_selection'].get('step', 1)

    # Cross-validation setup
    cv = KFold(
        n_splits=config['cross_validation']['n_splits'],
        shuffle=config['cross_validation']['shuffle'],
        random_state=config['cross_validation']['random_state']
    )

    results = []
    current_features = list(range(len(feature_names)))

    iteration = 0

    while len(current_features) >= min_features:
        iteration += 1
        n_current = len(current_features)

        logger.info(f"\n{'-'*50}")
        logger.info(f"Iteration {iteration}: Evaluating {n_current} features")
        logger.info(f"{'-'*50}")

        # Evaluate current feature set
        result = evaluate_feature_subset(
            X, y, current_features, feature_names,
            params, cv, logger
        )
        results.append(result)

        logger.info(f"  R²:   {result['metrics']['R2']:.6f}")
        logger.info(f"  RMSE: {result['metrics']['RMSE']:.4f}")
        logger.info(f"  MAE:  {result['metrics']['MAE']:.4f}")
        logger.info(f"  COV:  {result['metrics']['COV']:.6f}")

        # Determine which features to eliminate
        if len(current_features) <= min_features:
            break

        # Train model to get feature importance
        X_current = X[:, current_features]
        model = xgb.XGBRegressor(**params)
        model.fit(X_current, y, verbose=False)

        # Get feature importance for current subset
        importance = model.feature_importances_

        # Number of features to eliminate
        n_eliminate = min(step, len(current_features) - min_features)

        # Find least important features
        least_important_indices = np.argsort(importance)[:n_eliminate]

        # Remove from current features (in reverse order to maintain indices)
        for idx in sorted(least_important_indices, reverse=True):
            removed_feature = current_features[idx]
            logger.info(f"  Removing: {feature_names[removed_feature]} (importance: {importance[idx]:.4f})")
            current_features.pop(idx)

    return results


def find_optimal_features(
    results: List[Dict[str, Any]],
    metric: str = "COV",
    logger = None
) -> Dict[str, Any]:
    """
    Find the optimal number of features based on metrics.

    Args:
        results: List of results from each iteration
        metric: Metric to optimize ('COV', 'R2', 'RMSE', 'MAE')
        logger: Logger instance

    Returns:
        Best result dictionary
    """
    if metric == 'R2':
        # Higher is better
        best_idx = max(range(len(results)),
                      key=lambda i: results[i]['metrics'][metric])
    else:
        # Lower is better for COV, RMSE, MAE
        best_idx = min(range(len(results)),
                      key=lambda i: results[i]['metrics'][metric])

    best_result = results[best_idx]

    if logger:
        logger.info(f"\n{'='*50}")
        logger.info(f"Optimal Feature Set (based on {metric})")
        logger.info(f"{'='*50}")
        logger.info(f"Number of features: {best_result['n_features']}")
        logger.info(f"R²:   {best_result['metrics']['R2']:.6f}")
        logger.info(f"RMSE: {best_result['metrics']['RMSE']:.4f}")
        logger.info(f"MAE:  {best_result['metrics']['MAE']:.4f}")
        logger.info(f"COV:  {best_result['metrics']['COV']:.6f}")
        logger.info(f"\nSelected features:")
        for i, name in enumerate(best_result['feature_names'], 1):
            logger.info(f"  {i}. {name}")

    return best_result


def main():
    """Main feature selection pipeline."""
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Override config with command line args
    if args.min_features is not None:
        config['feature_selection']['min_features'] = args.min_features
    if args.step is not None:
        config['feature_selection']['step'] = args.step

    # Setup logging
    logger = setup_logger("feature_selection", level="INFO")

    logger.info("=" * 60)
    logger.info("Feature Selection Pipeline - Recursive Feature Elimination")
    logger.info("=" * 60)

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(config['paths']['output_dir']) / "feature_selection"
    ensure_dir(output_dir)

    # Step 1: Load Data
    logger.info("\n" + "-" * 40)
    logger.info("Step 1: Loading and Preprocessing Data")
    logger.info("-" * 40)

    data_loader = DataLoader(config, logger)
    df, feature_cols = data_loader.load_and_prepare()

    preprocessor = DataPreprocessor(config, logger)
    df_processed, feature_cols = preprocessor.preprocess(df, feature_cols, fit=True)

    X = df_processed[feature_cols].values
    y = df_processed[config['target']['name']].values

    logger.info(f"Data shape: {X.shape}")
    logger.info(f"Features: {feature_cols}")

    # Step 2: Get Model Parameters
    logger.info("\n" + "-" * 40)
    logger.info("Step 2: Loading Model Parameters")
    logger.info("-" * 40)

    trainer = XGBoostTrainer(config, logger)
    params = trainer.get_model_params()

    logger.info(f"Parameters: {params}")

    # Step 3: Recursive Feature Elimination
    logger.info("\n" + "-" * 40)
    logger.info("Step 3: Running Recursive Feature Elimination")
    logger.info("-" * 40)

    results = recursive_feature_elimination(
        X, y, feature_cols, params, config, logger
    )

    # Step 4: Find Optimal Feature Set
    logger.info("\n" + "-" * 40)
    logger.info("Step 4: Finding Optimal Feature Set")
    logger.info("-" * 40)

    best_cov = find_optimal_features(results, metric="COV", logger=logger)
    best_r2 = find_optimal_features(results, metric="R2", logger=logger)

    # Step 5: Generate Visualizations
    logger.info("\n" + "-" * 40)
    logger.info("Step 5: Generating Visualizations")
    logger.info("-" * 40)

    visualizer = ModelVisualizer(config, logger)

    # Prepare data for plotting
    n_features_list = [r['n_features'] for r in results]
    metrics_data = {
        'R2': [r['metrics']['R2'] for r in results],
        'RMSE': [r['metrics']['RMSE'] for r in results],
        'MAE': [r['metrics']['MAE'] for r in results],
        'COV': [r['metrics']['COV'] for r in results]
    }

    visualizer.plot_feature_selection_results(
        n_features_list, metrics_data,
        save_path=output_dir / "feature_selection_results.png"
    )

    # Step 6: Save Results
    logger.info("\n" + "-" * 40)
    logger.info("Step 6: Saving Results")
    logger.info("-" * 40)

    # Save detailed results
    results_file = output_dir / "feature_selection_results.json"
    with open(results_file, 'w') as f:
        # Convert to serializable format
        serializable_results = []
        for r in results:
            sr = {
                'n_features': r['n_features'],
                'feature_names': r['feature_names'],
                'metrics': {k: float(v) for k, v in r['metrics'].items()}
            }
            serializable_results.append(sr)
        json.dump(serializable_results, f, indent=2)
    logger.info(f"Saved results to: {results_file}")

    # Save best feature sets
    best_features_file = output_dir / "best_features_cov.txt"
    with open(best_features_file, 'w') as f:
        f.write(f"Optimal features based on COV ({best_cov['n_features']} features):\n")
        f.write("=" * 50 + "\n")
        for name in best_cov['feature_names']:
            f.write(f"{name}\n")
        f.write("\n")
        f.write(f"COV:  {best_cov['metrics']['COV']:.6f}\n")
        f.write(f"R²:   {best_cov['metrics']['R2']:.6f}\n")
        f.write(f"RMSE: {best_cov['metrics']['RMSE']:.4f}\n")
    logger.info(f"Saved best features (COV) to: {best_features_file}")

    best_features_r2_file = output_dir / "best_features_r2.txt"
    with open(best_features_r2_file, 'w') as f:
        f.write(f"Optimal features based on R² ({best_r2['n_features']} features):\n")
        f.write("=" * 50 + "\n")
        for name in best_r2['feature_names']:
            f.write(f"{name}\n")
        f.write("\n")
        f.write(f"R²:   {best_r2['metrics']['R2']:.6f}\n")
        f.write(f"COV:  {best_r2['metrics']['COV']:.6f}\n")
        f.write(f"RMSE: {best_r2['metrics']['RMSE']:.4f}\n")
    logger.info(f"Saved best features (R²) to: {best_features_r2_file}")

    # Save summary report
    report_file = output_dir / "feature_selection_report.txt"
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Feature Selection Report - Recursive Feature Elimination\n")
        f.write("=" * 60 + "\n\n")

        f.write("Summary by Number of Features:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'N_Features':<12} {'R2':<10} {'RMSE':<10} {'MAE':<10} {'COV':<10}\n")
        f.write("-" * 60 + "\n")

        for r in results:
            m = r['metrics']
            f.write(f"{r['n_features']:<12} {m['R2']:<10.4f} {m['RMSE']:<10.2f} "
                   f"{m['MAE']:<10.2f} {m['COV']:<10.4f}\n")

        f.write("\n")
        f.write("=" * 60 + "\n")
        f.write("Recommendations:\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"1. For best COV ({best_cov['metrics']['COV']:.4f}):\n")
        f.write(f"   Use {best_cov['n_features']} features\n\n")

        f.write(f"2. For best R² ({best_r2['metrics']['R2']:.4f}):\n")
        f.write(f"   Use {best_r2['n_features']} features\n\n")

        f.write("=" * 60 + "\n")

    logger.info(f"Saved report to: {report_file}")

    logger.info("\n" + "=" * 60)
    logger.info("Feature Selection Complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
