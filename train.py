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
        default="config/config_3.16_150.yaml",
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

    # Step 9: Generate Training Report
    logger.info("\n" + "-" * 40)
    logger.info("Step 9: Generating Training Report")
    logger.info("-" * 40)

    generate_training_report(
        output_dir=output_dir,
        config=config,
        feature_cols=feature_cols,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        best_params=best_params,
        use_optuna=config['optuna']['use_optuna'],
        n_trials=n_trials if config['optuna']['use_optuna'] else None,
        X_train_shape=X_train.shape,
        X_test_shape=X_test.shape,
        y_train_shape=y_train.shape,
        y_test_shape=y_test.shape,
        use_log_transform=use_log_transform
    )

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Model saved to: {config['paths']['model_file']}")
    logger.info(f"Outputs saved to: {config['paths']['output_dir']}")
    logger.info("=" * 60)

    return 0


def generate_training_report(
    output_dir: Path,
    config: Dict[str, Any],
    feature_cols: list,
    train_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
    best_params: Dict[str, Any],
    use_optuna: bool,
    n_trials: int,
    X_train_shape: tuple,
    X_test_shape: tuple,
    y_train_shape: tuple,
    y_test_shape: tuple,
    use_log_transform: bool
) -> None:
    """
    Generate comprehensive training report.

    Args:
        output_dir: Output directory path
        config: Configuration dictionary
        feature_cols: List of feature column names
        train_metrics: Training metrics dictionary
        test_metrics: Test metrics dictionary
        best_params: Best hyperparameters
        use_optuna: Whether Optuna was used
        n_trials: Number of optimization trials
        X_train_shape: Shape of training features
        X_test_shape: Shape of test features
        y_train_shape: Shape of training targets
        y_test_shape: Shape of test targets
        use_log_transform: Whether log transform was used
    """
    from datetime import datetime

    report_path = output_dir / "训练报告.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("                    XGBoost 模型训练报告\n")
        f.write("=" * 70 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置文件: {config.get('config_path', 'config/config_linear.yaml')}\n")
        f.write("\n")

        # 1. 数据划分依据
        f.write("-" * 70 + "\n")
        f.write("一、数据集划分依据\n")
        f.write("-" * 70 + "\n")
        f.write(f"总样本数: {X_train_shape[0] + X_test_shape[0]}\n")
        f.write(f"训练集样本数: {X_train_shape[0]} ({X_train_shape[0] / (X_train_shape[0] + X_test_shape[0]) * 100:.1f}%)\n")
        f.write(f"测试集样本数: {X_test_shape[0]} ({X_test_shape[0] / (X_train_shape[0] + X_test_shape[0]) * 100:.1f}%)\n")
        f.write(f"特征数量: {X_train_shape[1]}\n")
        f.write(f"\n划分方法:\n")
        f.write(f"  - 划分比例: 训练集 {(1 - config['preprocessing']['test_size']) * 100:.0f}% / 测试集 {config['preprocessing']['test_size'] * 100:.0f}%\n")
        f.write(f"  - 随机种子: {config['preprocessing']['random_state']}\n")
        f.write(f"  - 划分策略: 随机划分 (train_test_split)\n")
        f.write(f"\n特征列表:\n")
        for i, feat in enumerate(feature_cols, 1):
            f.write(f"  {i:2d}. {feat}\n")
        f.write("\n")

        # 2. 超参数调优结果
        f.write("-" * 70 + "\n")
        f.write("二、超参数调优结果\n")
        f.write("-" * 70 + "\n")
        if use_optuna:
            f.write(f"调优方法: Optuna 贝叶斯优化\n")
            f.write(f"调优轮数: {n_trials} 轮\n")
            f.write(f"优化目标: 最小化 RMSE\n")
            f.write(f"交叉验证: {config['cross_validation']['n_splits']} 折\n")
            f.write(f"\n最优超参数:\n")
            f.write("-" * 40 + "\n")
            for key, value in best_params.items():
                if isinstance(value, float):
                    f.write(f"  {key:20s}: {value:.6f}\n")
                else:
                    f.write(f"  {key:20s}: {value}\n")
        else:
            f.write("调优方法: 使用预设参数（未进行自动调优）\n")
            f.write(f"\n使用的参数:\n")
            f.write("-" * 40 + "\n")
            for key, value in best_params.items():
                if isinstance(value, float):
                    f.write(f"  {key:20s}: {value:.6f}\n")
                else:
                    f.write(f"  {key:20s}: {value}\n")
        f.write("\n")

        # 3. 模型评估指标
        f.write("-" * 70 + "\n")
        f.write("三、模型评估指标\n")
        f.write("-" * 70 + "\n")

        f.write("\n【训练集指标】\n")
        f.write("-" * 40 + "\n")
        f.write(f"  R² 决定系数:          {train_metrics['R2']:.6f}\n")
        f.write(f"  RMSE 均方根误差:      {train_metrics['RMSE']:.4f}\n")
        f.write(f"  MAE 平均绝对误差:     {train_metrics['MAE']:.4f}\n")
        f.write(f"  MAPE 平均绝对百分比:  {train_metrics['MAPE']:.4f}%\n")
        f.write(f"  COV 变异系数:         {train_metrics['COV']:.6f}\n")
        f.write(f"\n  比率分析 (预测/实际):\n")
        f.write(f"    - rati_mean (均值): {train_metrics.get('rati_mean', train_metrics.get('mu_xi', 0)):.6f} (理想值: 1.0)\n")
        f.write(f"    - sigma_xi (标准差): {train_metrics['sigma_xi']:.6f}\n")
        f.write(f"    - 预测在 ±10% 内:    {train_metrics['within_10pct']:.2f}%\n")
        f.write(f"    - 预测在 ±20% 内:    {train_metrics['within_20pct']:.2f}%\n")

        f.write("\n【测试集指标】\n")
        f.write("-" * 40 + "\n")
        f.write(f"  R² 决定系数:          {test_metrics['R2']:.6f}\n")
        f.write(f"  RMSE 均方根误差:      {test_metrics['RMSE']:.4f}\n")
        f.write(f"  MAE 平均绝对误差:     {test_metrics['MAE']:.4f}\n")
        f.write(f"  MAPE 平均绝对百分比:  {test_metrics['MAPE']:.4f}%\n")
        f.write(f"  COV 变异系数:         {test_metrics['COV']:.6f}\n")
        f.write(f"\n  比率分析 (预测/实际):\n")
        f.write(f"    - rati_mean (均值): {test_metrics.get('rati_mean', test_metrics.get('mu_xi', 0)):.6f} (理想值: 1.0)\n")
        f.write(f"    - sigma_xi (标准差): {test_metrics['sigma_xi']:.6f}\n")
        f.write(f"    - 预测在 ±10% 内:    {test_metrics['within_10pct']:.2f}%\n")
        f.write(f"    - 预测在 ±20% 内:    {test_metrics['within_20pct']:.2f}%\n")

        f.write("\n【泛化能力评估】\n")
        f.write("-" * 40 + "\n")
        r2_diff = train_metrics['R2'] - test_metrics['R2']
        rmse_ratio = test_metrics['RMSE'] / train_metrics['RMSE'] if train_metrics['RMSE'] > 0 else 0
        f.write(f"  训练集与测试集 R² 差异: {r2_diff:.6f} ({'正常' if r2_diff < 0.05 else '可能存在过拟合'})\n")
        f.write(f"  测试/训练 RMSE 比值:    {rmse_ratio:.4f} ({'正常' if rmse_ratio < 2.0 else '可能存在过拟合'})\n")
        f.write(f"\n  评估结论: {'模型泛化能力良好' if r2_diff < 0.05 and rmse_ratio < 2.0 else '模型可能存在过拟合，建议增加正则化或减少模型复杂度'}\n")
        f.write("\n")

        # 4. 预处理配置
        f.write("-" * 70 + "\n")
        f.write("四、数据预处理配置\n")
        f.write("-" * 70 + "\n")
        f.write(f"  缺失值处理策略: {config['preprocessing']['imputation_strategy']}\n")
        f.write(f"  特征缩放:       {'已启用' if config['preprocessing']['scale_features'] else '未启用'}\n")
        f.write(f"  目标变量对数变换: {'已启用' if use_log_transform else '未启用'}\n")
        if use_log_transform:
            f.write(f"  对数变换 epsilon: {config['preprocessing'].get('log_transform_epsilon', 1.0)}\n")
        f.write("\n")

        # 5. 输出文件列表
        f.write("-" * 70 + "\n")
        f.write("五、输出文件列表\n")
        f.write("-" * 70 + "\n")
        f.write(f"  - 模型文件:              {output_dir / 'xgboost_model.pkl'}\n")
        f.write(f"  - 训练报告:              {output_dir / '训练报告.txt'}\n")
        f.write(f"  - 特征重要性图:          {output_dir / 'feature_importance.png'}\n")
        f.write(f"  - 训练集预测图:          {output_dir / 'predictions_train.png'}\n")
        f.write(f"  - 测试集预测图:          {output_dir / 'predictions_test.png'}\n")
        f.write(f"  - 残差分析图:            {output_dir / 'residuals_test.png'}\n")
        f.write(f"  - 比率分析图(训练集):    {output_dir / 'ratio_analysis_train.png'}\n")
        f.write(f"  - 比率分析图(测试集):    {output_dir / 'ratio_analysis_test.png'}\n")
        f.write(f"  - 特征相关性矩阵:        {output_dir / 'correlation_matrix.png'}\n")
        f.write(f"  - 特征名称列表:          {output_dir / 'feature_names.txt'}\n")
        f.write(f"  - 评估指标:              {output_dir / 'metrics.txt'}\n")
        f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("                    报告结束\n")
        f.write("=" * 70 + "\n")

    print(f"\n训练报告已生成: {report_path}")


if __name__ == "__main__":
    sys.exit(main())
