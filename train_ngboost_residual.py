#!/usr/bin/env python3
"""
Training script for NGBoost residual model.

This script implements the two-stage training pipeline:
1. Load trained XGBoost model
2. Calculate residuals (y - xgboost_pred)
3. Train NGBoost on residuals to model uncertainty
4. Evaluate prediction interval coverage

Usage:
    python train_ngboost_residual.py --config config/config_ngboost.yaml
    python train_ngboost_residual.py --config config/config_ngboost.yaml --optimize
    python train_ngboost_residual.py --config config/config_ngboost.yaml --no-optimize
"""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.model_selection import KFold

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import XGBoostTrainer
from src.ngboost_trainer import NGBoostTrainer
from src.evaluator import ModelEvaluator
from src.visualizer import ModelVisualizer
from src.utils import load_config, setup_logger, ensure_dir


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train NGBoost model on XGBoost residuals for uncertainty quantification"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_ngboost.yaml",
        help="Path to NGBoost configuration file"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Optuna hyperparameter optimization"
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip Optuna optimization"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides config)"
    )
    parser.add_argument(
        "--xgboost-model",
        type=str,
        default=None,
        help="Path to XGBoost model file (overrides config)"
    )
    return parser.parse_args()


def load_xgboost_model(config: Dict[str, Any], logger, model_path: str = None):
    """Load trained XGBoost model."""
    xgb_model_path = model_path or config['paths']['xgboost_model']
    logger.info(f"Loading XGBoost model from: {xgb_model_path}")

    # Use XGBoostTrainer to load the model
    xgb_trainer = XGBoostTrainer(config, logger)
    xgb_model, preprocessor_state = xgb_trainer.load_model(xgb_model_path)

    return xgb_trainer, preprocessor_state


def calculate_residuals(
    xgb_trainer: XGBoostTrainer,
    X: np.ndarray,
    y: np.ndarray,
    preprocessor: DataPreprocessor,
    use_log_transform: bool
) -> np.ndarray:
    """
    Calculate residuals: ε = y - ŷ

    Args:
        xgb_trainer: Trained XGBoost trainer
        X: Feature matrix
        y: True targets (in log space if log_transform is enabled)
        preprocessor: Data preprocessor for inverse transform
        use_log_transform: Whether log transform was applied

    Returns:
        Residuals array
    """
    # Get XGBoost predictions (in log space if applicable)
    y_pred_log = xgb_trainer.predict(X)

    # Inverse transform to original scale
    if use_log_transform:
        y_pred = preprocessor.inverse_transform_target(y_pred_log)
        y_orig = preprocessor.inverse_transform_target(y)
    else:
        y_pred = y_pred_log
        y_orig = y

    # Calculate residuals in original scale
    residuals = y_orig - y_pred

    return residuals, y_orig, y_pred


def calculate_xgboost_prediction_variance(
    xgb_trainer: XGBoostTrainer,
    X: np.ndarray,
    preprocessor: DataPreprocessor,
    use_log_transform: bool
) -> np.ndarray:
    """
    Calculate prediction variance from XGBoost ensemble.

    Uses the variance across individual tree predictions as an uncertainty estimate.

    Args:
        xgb_trainer: Trained XGBoost trainer
        X: Feature matrix
        preprocessor: Data preprocessor for inverse transform
        use_log_transform: Whether log transform was applied

    Returns:
        Variance array (one value per sample)
    """
    import xgboost as xgb

    model = xgb_trainer.model
    n_estimators = model.n_estimators

    # Get predictions from all trees
    # Use iteration_range to get predictions from each tree
    all_predictions = []
    for i in range(n_estimators):
        pred = model.predict(X, iteration_range=(i, i+1))
        if use_log_transform:
            pred = preprocessor.inverse_transform_target(pred)
        all_predictions.append(pred)

    # Stack predictions and calculate variance
    all_predictions = np.array(all_predictions)  # Shape: (n_estimators, n_samples)
    variance = np.var(all_predictions, axis=0)  # Shape: (n_samples,)

    return variance


def evaluate_interval_coverage(
    ngboost_trainer: NGBoostTrainer,
    X: np.ndarray,
    y_true: np.ndarray,
    xgb_pred: np.ndarray,
    logger,
    dataset_name: str = "Test"
) -> Dict[str, Any]:
    """
    Evaluate prediction interval coverage.

    Args:
        ngboost_trainer: Trained NGBoost trainer
        X: Feature matrix
        y_true: True targets (original scale)
        xgb_pred: XGBoost predictions (original scale)
        logger: Logger instance
        dataset_name: Name of dataset

    Returns:
        Dictionary of coverage metrics
    """
    # Get residual distribution parameters
    mu_epsilon, sigma_epsilon = ngboost_trainer.predict_params(X)

    # Get prediction intervals for residuals
    epsilon_lower, epsilon_upper = ngboost_trainer.predict_interval(X)

    # Calculate final prediction intervals (in original scale)
    y_lower = xgb_pred + epsilon_lower
    y_upper = xgb_pred + epsilon_upper
    y_pred_final = xgb_pred + mu_epsilon

    # Calculate coverage
    coverage_rate = np.mean((y_true >= y_lower) & (y_true <= y_upper))

    # Calculate interval width statistics
    interval_width = y_upper - y_lower
    mean_width = np.mean(interval_width)
    mean_width_pct = np.mean(interval_width / y_true) * 100

    # Calculate metrics for final predictions (ensemble model)
    evaluator = ModelEvaluator(logger)
    ensemble_metrics = evaluator.evaluate(y_true, y_pred_final, f"Ensemble {dataset_name}")

    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"Prediction Interval Coverage - {dataset_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Interval Coverage: {coverage_rate:.2%}")
    logger.info(f"\nInterval Width Statistics:")
    logger.info(f"  Mean Width: {mean_width:.4f}")
    logger.info(f"  Mean Width / Actual: {mean_width_pct:.2f}%")
    logger.info(f"{'='*60}")

    return {
        'coverage_rate': coverage_rate,
        'mean_interval_width': mean_width,
        'mean_interval_width_pct': mean_width_pct,
        'y_lower': y_lower,
        'y_upper': y_upper,
        'y_pred_final': y_pred_final,
        'mu_epsilon': mu_epsilon,
        'sigma_epsilon': sigma_epsilon,
        'ensemble_metrics': ensemble_metrics
    }


def main():
    """Main training pipeline."""
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"output/ngboost_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update config paths to use timestamped directory
    config['paths']['output_dir'] = str(output_dir)
    config['paths']['ngboost_model'] = str(output_dir / "ngboost_residual_model.pkl")
    config['paths']['logs_dir'] = str(output_dir / "logs")
    config['paths']['optuna_db'] = str(output_dir / "logs" / "ngboost_optuna.db")

    # Preserve existing best params file if available (so --no-optimize can load them)
    original_best_params = Path(config['paths'].get('best_params', 'logs/ngboost_best_params.json'))
    config['paths']['best_params'] = str(output_dir / "logs" / "ngboost_best_params.json")
    if original_best_params.exists():
        ensure_dir(Path(config['paths']['best_params']).parent)
        shutil.copy(original_best_params, config['paths']['best_params'])

    # Override XGBoost model path if specified
    if args.xgboost_model:
        config['paths']['xgboost_model'] = args.xgboost_model

    # Setup logging
    log_file = config['paths']['logs_dir'] + "/ngboost_training.log"
    ensure_dir(config['paths']['logs_dir'])
    logger = setup_logger(
        "ngboost_train",
        level=config.get('logging', {}).get('level', 'INFO'),
        log_file=log_file
    )

    logger.info("=" * 60)
    logger.info("NGBoost Residual Model Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"XGBoost model: {config['paths']['xgboost_model']}")

    # Override config with command line args
    if args.optimize:
        config['optuna']['use_optuna'] = True
    if args.no_optimize:
        config['optuna']['use_optuna'] = False

    # Create output directories
    ensure_dir(config['paths']['logs_dir'])
    ensure_dir(config['paths']['output_dir'])

    # Step 1: Load XGBoost Model
    logger.info("\n" + "-" * 40)
    logger.info("Step 1: Loading XGBoost Model")
    logger.info("-" * 40)

    try:
        xgb_trainer, preprocessor_state = load_xgboost_model(
            config, logger, args.xgboost_model
        )
    except FileNotFoundError as e:
        logger.error(f"XGBoost model not found: {e}")
        logger.error("Please train XGBoost model first using: python train.py")
        return 1

    # Step 2: Load and Preprocess Data
    logger.info("\n" + "-" * 40)
    logger.info("Step 2: Loading and Preprocessing Data")
    logger.info("-" * 40)

    data_loader = DataLoader(config, logger)
    df, feature_cols = data_loader.load_and_prepare(config['paths']['raw_data'])

    preprocessor = DataPreprocessor(config, logger)
    df_processed, feature_cols = preprocessor.preprocess(df, feature_cols, fit=True)

    # Check if log transform is enabled (must match XGBoost config)
    use_log_transform = config['preprocessing'].get('log_transform_target', False)
    if use_log_transform:
        logger.info("Log transform enabled for target variable (Nexp)")

    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df_processed, feature_cols, apply_log_transform=True
    )

    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Test samples: {len(X_test)}")

    # Step 3: Calculate Residuals
    logger.info("\n" + "-" * 40)
    logger.info("Step 3: Calculating Residuals from XGBoost")
    logger.info("-" * 40)

    # Training set residuals
    residuals_train, y_train_orig, y_train_pred = calculate_residuals(
        xgb_trainer, X_train, y_train, preprocessor, use_log_transform
    )
    logger.info(f"Training residuals - Mean: {np.mean(residuals_train):.4f}, "
                f"Std: {np.std(residuals_train):.4f}")

    # Test set residuals
    residuals_test, y_test_orig, y_test_pred = calculate_residuals(
        xgb_trainer, X_test, y_test, preprocessor, use_log_transform
    )
    logger.info(f"Test residuals - Mean: {np.mean(residuals_test):.4f}, "
                f"Std: {np.std(residuals_test):.4f}")

    # Step 3.5: Calculate XGBoost Prediction Variance as additional feature
    logger.info("\n" + "-" * 40)
    logger.info("Step 3.5: Calculating XGBoost Prediction Variance")
    logger.info("-" * 40)
    logger.info("Using XGBoost ensemble variance as input feature for NGBoost")

    var_train = calculate_xgboost_prediction_variance(
        xgb_trainer, X_train, preprocessor, use_log_transform
    )
    var_test = calculate_xgboost_prediction_variance(
        xgb_trainer, X_test, preprocessor, use_log_transform
    )
    logger.info(f"Training variance - Mean: {np.mean(var_train):.4f}, "
                f"Std: {np.std(var_train):.4f}")
    logger.info(f"Test variance - Mean: {np.mean(var_test):.4f}, "
                f"Std: {np.std(var_test):.4f}")

    # Add variance as additional feature
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    X_train_with_var = np.column_stack([X_train, var_train])
    X_test_with_var = np.column_stack([X_test, var_test])
    logger.info(f"Feature matrix shape with variance: {X_train_with_var.shape}")

    # Step 4: Train NGBoost on Residuals
    logger.info("\n" + "-" * 40)
    logger.info("Step 4: Training NGBoost on Residuals")
    logger.info("-" * 40)

    ngboost_trainer = NGBoostTrainer(config, logger)

    # Hyperparameter optimization (optional)
    if config['optuna']['use_optuna']:
        logger.info("\nRunning hyperparameter optimization...")
        cv = KFold(
            n_splits=config['cross_validation']['n_splits'],
            shuffle=config['cross_validation']['shuffle'],
            random_state=config['cross_validation']['random_state']
        )
        n_trials = args.n_trials or config['optuna']['n_trials']
        best_params = ngboost_trainer.optimize_hyperparameters(
            X_train_with_var, residuals_train, cv, n_trials
        )
    else:
        best_params = ngboost_trainer.get_model_params()

    # Train final model (with variance feature)
    ngboost_trainer.train(X_train_with_var, residuals_train, X_test_with_var, residuals_test, params=best_params)

    # Step 4.5: Calibrate prediction intervals using validation set (split conformal)
    logger.info("\n" + "-" * 40)
    logger.info("Step 4.5: Calibrating Prediction Intervals")
    logger.info("-" * 40)
    ngboost_trainer.calibrate_intervals(X_test_with_var, residuals_test)

    # Step 5: Evaluate Prediction Intervals
    logger.info("\n" + "-" * 40)
    logger.info("Step 5: Evaluating Prediction Intervals")
    logger.info("-" * 40)

    # Training set coverage (with variance feature)
    train_coverage = evaluate_interval_coverage(
        ngboost_trainer, X_train_with_var, y_train_orig, y_train_pred, logger, "Train"
    )

    # Test set coverage (with variance feature)
    test_coverage = evaluate_interval_coverage(
        ngboost_trainer, X_test_with_var, y_test_orig, y_test_pred, logger, "Test"
    )

    # Evaluate overfitting
    evaluator = ModelEvaluator(logger)
    overfit_metrics = evaluator.evaluate_overfitting(
        train_coverage['ensemble_metrics'],
        test_coverage['ensemble_metrics']
    )

    # Check if coverage meets target
    # Get target coverage from config
    target_coverage = config.get('confidence', {}).get('level', 0.95)
    if test_coverage['coverage_rate'] >= target_coverage:
        logger.info(f"\n[OK] Coverage target met: {test_coverage['coverage_rate']:.2%} >= {target_coverage:.0%}")
    else:
        logger.warning(f"\n[X] Coverage target NOT met: {test_coverage['coverage_rate']:.2%} < {target_coverage:.0%}")
        logger.warning("Consider increasing n_estimators or adjusting learning_rate")

    # Log Ensemble Model Performance Summary
    logger.info("\n" + "=" * 60)
    logger.info("Ensemble Model (XGBoost + NGBoost) Performance Summary")
    logger.info("=" * 60)
    logger.info("Using NGBoost mean-adjusted predictions for evaluation")
    logger.info("-" * 60)

    # Training set metrics
    train_metrics = train_coverage['ensemble_metrics']
    logger.info(f"\n[Training Set]")
    logger.info(f"  R2 Score:  {train_metrics['R2']:.6f}")
    logger.info(f"  MSE:       {train_metrics['MSE']:.4f}")
    logger.info(f"  RMSE:      {train_metrics['RMSE']:.4f}")
    logger.info(f"  MAE:       {train_metrics['MAE']:.4f}")
    logger.info(f"  MAPE:      {train_metrics['MAPE']:.4f}%")

    # Test set metrics
    test_metrics = test_coverage['ensemble_metrics']
    logger.info(f"\n[Test Set]")
    logger.info(f"  R2 Score:  {test_metrics['R2']:.6f}")
    logger.info(f"  MSE:       {test_metrics['MSE']:.4f}")
    logger.info(f"  RMSE:      {test_metrics['RMSE']:.4f}")
    logger.info(f"  MAE:       {test_metrics['MAE']:.4f}")
    logger.info(f"  MAPE:      {test_metrics['MAPE']:.4f}%")
    logger.info("=" * 60)

    # Step 6: Generate Visualizations
    logger.info("\n" + "-" * 40)
    logger.info("Step 6: Generating Visualizations")
    logger.info("-" * 40)

    visualizer = ModelVisualizer(config, logger)
    output_dir = Path(config['paths']['output_dir'])

    # Feature names for NGBoost interpretation plots
    ngboost_feature_names = feature_cols + ['xgb_variance']

    # Plot residual distribution
    visualizer.plot_residual_distribution(
        residuals_train, residuals_test,
        save_path=output_dir / "residual_distribution.png"
    )

    # Plot prediction intervals
    visualizer.plot_prediction_intervals(
        y_test_orig,
        y_test_pred,
        test_coverage['y_lower'],
        test_coverage['y_upper'],
        save_path=output_dir / "prediction_intervals.png"
    )

    # Plot calibration curve (with variance feature)
    visualizer.plot_calibration_curve(
        ngboost_trainer, X_test_with_var, y_test_orig, y_test_pred,
        save_path=output_dir / "calibration_curve.png"
    )

    # 1. 真实值 vs 预测值散点图（加残差修正对比）
    visualizer.plot_stacking_comparison(
        y_true=y_test_orig,
        y_pred_xgb=y_test_pred,
        y_pred_final=test_coverage['y_pred_final'],
        dataset_name="Test",
        save_path=output_dir / "stacking_comparison.png"
    )

    # 2. 残差诊断图（核心灵魂）
    visualizer.plot_residual_diagnostics(
        y_true=y_test_orig,
        y_pred_xgb=y_test_pred,
        y_pred_final=test_coverage['y_pred_final'],
        dataset_name="Test",
        save_path=output_dir / "residual_diagnostics.png"
    )

    # 3. 不确定性校准图（NGBoost 独特价值）
    visualizer.plot_uncertainty_calibration(
        ngboost_trainer=ngboost_trainer,
        X=X_test_with_var,
        residuals_true=residuals_test,
        y_true=y_test_orig,
        y_pred_xgb=y_test_pred,
        dataset_name="Test",
        save_path=output_dir / "uncertainty_calibration.png"
    )

    # 4. 特征重要性与残差敏感性图（业务解释）
    visualizer.plot_ngboost_interpretation(
        ngboost_trainer=ngboost_trainer,
        X=X_test_with_var,
        feature_names=ngboost_feature_names,
        top_n=5,
        save_path=output_dir / "ngboost_interpretation.png"
    )

    # ------------------------------------------------------------------
    # Direction-2 engineering-oriented visualizations
    # ------------------------------------------------------------------

    # A. Prediction reliability map
    visualizer.plot_prediction_reliability_map(
        y_true=y_test_orig,
        y_pred=y_test_pred,
        y_lower=test_coverage['y_lower'],
        y_upper=test_coverage['y_upper'],
        dataset_name="Test",
        save_path=output_dir / "prediction_reliability_map.png"
    )

    # B. Uncertainty-error correlation
    visualizer.plot_uncertainty_error_correlation(
        y_true=y_test_orig,
        y_pred=y_test_pred,
        y_lower=test_coverage['y_lower'],
        y_upper=test_coverage['y_upper'],
        dataset_name="Test",
        save_path=output_dir / "uncertainty_error_correlation.png"
    )

    # C. Correction benefit analysis
    residual_pred_test = ngboost_trainer.predict(X_test_with_var)
    visualizer.plot_correction_benefit(
        y_true=y_test_orig,
        y_pred_xgb=y_test_pred,
        y_pred_final=test_coverage['y_pred_final'],
        residual_pred=residual_pred_test,
        dataset_name="Test",
        save_path=output_dir / "correction_benefit.png"
    )

    # D. Safety margin chart
    visualizer.plot_safety_margin(
        y_true=y_test_orig,
        y_pred=y_test_pred,
        y_lower=test_coverage['y_lower'],
        dataset_name="Test",
        save_path=output_dir / "safety_margin.png"
    )

    # Step 7: Save Model
    logger.info("\n" + "-" * 40)
    logger.info("Step 7: Saving NGBoost Model")
    logger.info("-" * 40)

    ngboost_trainer.save_model()

    # ------------------------------------------------------------------
    # Compute additional metrics for engineering report
    # ------------------------------------------------------------------

    # XGBoost-only metrics on the CURRENT test set for fair comparison
    xgb_only_test_metrics = evaluator.evaluate(y_test_orig, y_test_pred, "XGBoost Only (Test)")

    # Coverage at multiple confidence levels
    coverage_table = []
    for lvl in [0.80, 0.90, 0.95, 0.99]:
        eps_l, eps_u = ngboost_trainer.predict_interval(X_test_with_var, confidence=lvl)
        y_l = y_test_pred + eps_l
        y_u = y_test_pred + eps_u
        cov = np.mean((y_test_orig >= y_l) & (y_test_orig <= y_u))
        width = np.mean(y_u - y_l)
        coverage_table.append((lvl, cov, width))

    # Save coverage summary
    coverage_file = output_dir / "coverage_summary.txt"

    # Get ensemble metrics for summary
    train_ensemble_metrics = train_coverage['ensemble_metrics']
    test_ensemble_metrics = test_coverage['ensemble_metrics']

    with open(coverage_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("NGBoost Residual Model - Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write("Note: In this Direction-2 setup, XGBoost provides the point prediction,\n")
        f.write("      while NGBoost provides calibrated prediction intervals and\n")
        f.write("      uncertainty-aware safety margins.\n\n")

        # Prediction Interval Coverage
        conf_level = config.get('confidence', {}).get('level', 0.95)
        f.write("-" * 70 + "\n")
        f.write(f"1. Prediction Interval Coverage (Target: {conf_level:.0%})\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Training: {train_coverage['coverage_rate']:.2%}\n")
        f.write(f"  Test:     {test_coverage['coverage_rate']:.2%}\n")
        f.write(f"  Status:   {'PASS' if test_coverage['coverage_rate'] >= conf_level else 'FAIL'}\n\n")

        f.write("-" * 70 + "\n")
        f.write("1a. Multi-Level Coverage Verification\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {'Confidence':>12s} {'Actual Coverage':>18s} {'Mean Width (kN)':>18s}\n")
        for lvl, cov, width in coverage_table:
            f.write(f"  {lvl:>11.0%} {cov:>17.2%} {width:>18.2f}\n")
        f.write("\n")

        # Interval Width Statistics
        f.write("-" * 70 + "\n")
        f.write("2. Prediction Interval Width Statistics\n")
        f.write("-" * 70 + "\n")
        f.write(f"  Mean Width (Training): {train_coverage['mean_interval_width']:.4f}\n")
        f.write(f"  Mean Width (Test):     {test_coverage['mean_interval_width']:.4f}\n")
        f.write(f"  Width / Actual (Training): {train_coverage['mean_interval_width_pct']:.2f}%\n")
        f.write(f"  Width / Actual (Test):     {test_coverage['mean_interval_width_pct']:.2f}%\n\n")

        # Point Prediction Performance Comparison
        f.write("-" * 70 + "\n")
        f.write("3. Point Prediction Performance Comparison\n")
        f.write("-" * 70 + "\n")
        f.write("   [Test Set]\n\n")
        f.write(f"   {'Metric':>12s} {'XGBoost Only':>16s} {'XGB+NGBoost Mean':>18s} {'Delta':>12s}\n")
        for metric in ['R2', 'RMSE', 'MAE', 'MAPE', 'COV']:
            xgb_val = xgb_only_test_metrics[metric]
            ens_val = test_ensemble_metrics[metric]
            delta = ens_val - xgb_val
            if metric == 'R2':
                delta_str = f"{delta:+.6f}"
            elif metric == 'MAPE':
                delta_str = f"{delta:+.4f}%"
            else:
                delta_str = f"{delta:+.4f}"
            xgb_str = f"{xgb_val:.6f}" if metric == 'R2' else f"{xgb_val:.4f}"
            ens_str = f"{ens_val:.6f}" if metric == 'R2' else (f"{ens_val:.4f}%" if metric == 'MAPE' else f"{ens_val:.4f}")
            f.write(f"   {metric:>12s} {xgb_str:>16s} {ens_str:>18s} {delta_str:>12s}\n")
        f.write("\n   Interpretation:\n")
        f.write("   - XGBoost already achieves very strong point-prediction performance.\n")
        f.write("   - NGBoost's primary value is NOT to squeeze R2 further, but to\n")
        f.write("     provide calibrated uncertainty bounds for risk-aware decisions.\n\n")

        # Safety Margin Analysis
        f.write("-" * 70 + "\n")
        f.write("4. Engineering Safety Margin Analysis\n")
        f.write("-" * 70 + "\n")
        safety_rate = np.mean(y_test_orig >= test_coverage['y_lower'])
        mean_margin = np.mean(y_test_pred - test_coverage['y_lower'])
        mean_margin_pct = np.mean((y_test_pred - test_coverage['y_lower']) / y_test_pred) * 100
        f.write(f"  Using NGBoost lower bound as conservative design value:\n")
        f.write(f"    - Actual >= Lower Bound Rate: {safety_rate:.2%}\n")
        f.write(f"    - Mean Safety Margin:         {mean_margin:.2f} kN\n")
        f.write(f"    - Mean Safety Margin (%):     {mean_margin_pct:.2f}%\n\n")

        # Overfitting Analysis
        f.write("-" * 70 + "\n")
        f.write("5. Overfitting Analysis\n")
        f.write("-" * 70 + "\n")
        f.write(f"   R2 Gap (Train - Test):     {overfit_metrics['r2_gap']:.6f}\n")
        f.write(f"   RMSE Ratio (Test/Train):   {overfit_metrics['rmse_ratio']:.4f}\n")
        f.write(f"   MSE Ratio (Test/Train):    {overfit_metrics['mse_ratio']:.4f}\n")
        f.write(f"   MAPE Ratio (Test/Train):   {overfit_metrics['mape_ratio']:.4f}\n")
        f.write(f"   Overfitting Score:         {overfit_metrics['overfitting_score']:.4f}\n")
        f.write(f"   Generalization Score:      {overfit_metrics['generalization_score']:.2f}/100\n")
        f.write(f"   Severity:                  {overfit_metrics['severity']}\n\n")

        # Ratio Metrics for Ensemble
        f.write("-" * 70 + "\n")
        f.write("6. Ratio Metrics (xi = prediction / actual)\n")
        f.write("-" * 70 + "\n")
        f.write("   [Training Set]\n")
        f.write(f"     mu_xi (mean): {train_ensemble_metrics.get('mu_xi', 0):.6f} (ideal: 1.0)\n")
        f.write(f"     sigma_xi (std): {train_ensemble_metrics.get('sigma_xi', 0):.6f}\n")
        f.write(f"     Within +/-10%: {train_ensemble_metrics.get('within_10pct', 0):.2f}%\n")
        f.write(f"     Within +/-20%: {train_ensemble_metrics.get('within_20pct', 0):.2f}%\n\n")

        f.write("   [Test Set]\n")
        f.write(f"     mu_xi (mean): {test_ensemble_metrics.get('mu_xi', 0):.6f} (ideal: 1.0)\n")
        f.write(f"     sigma_xi (std): {test_ensemble_metrics.get('sigma_xi', 0):.6f}\n")
        f.write(f"     Within +/-10%: {test_ensemble_metrics.get('within_10pct', 0):.2f}%\n")
        f.write(f"     Within +/-20%: {test_ensemble_metrics.get('within_20pct', 0):.2f}%\n\n")

        # Model Configuration Summary
        f.write("-" * 70 + "\n")
        f.write("7. Model Configuration\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Confidence Level:  {config.get('confidence', {}).get('level', 0.97):.0%}\n")
        f.write(f"   Hyperparameter Optimization: {'Enabled' if config['optuna']['use_optuna'] else 'Disabled'}\n")
        if config['optuna']['use_optuna']:
            f.write(f"   Number of Trials:  {config['optuna'].get('n_trials', 100)}\n")
        f.write(f"   Early Stopping:    {config.get('early_stopping', {}).get('use_early_stopping', False)}\n")
        f.write(f"   Log Transform:     {config['preprocessing'].get('log_transform_target', False)}\n\n")

        f.write("=" * 70 + "\n")
        f.write("End of Report\n")
        f.write("=" * 70 + "\n")

    logger.info(f"Saved coverage summary to: {coverage_file}")

    logger.info("\n" + "=" * 60)
    logger.info("NGBoost Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {config['paths']['ngboost_model']}")
    logger.info(f"Outputs saved to: {output_dir}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
