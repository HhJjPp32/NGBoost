#!/usr/bin/env python3
"""
Main prediction script for making inference with trained XGBoost model.

Usage:
    python predict.py --input data/new_data.csv --output predictions.csv
    python predict.py --input data/new_data.csv --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.predictor import Predictor
from src.utils import load_config, setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions with trained XGBoost model"
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
        default="predictions.csv",
        help="Path to save predictions CSV"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (overrides config)"
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=True,
        help="Apply preprocessing to input data"
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
    logger.info("XGBoost Prediction Pipeline")
    logger.info("=" * 60)

    # Handle preprocess flag
    preprocess = not args.no_preprocess if args.no_preprocess else args.preprocess

    # Initialize predictor
    model_path = args.model or config['paths']['model_file']

    try:
        predictor = Predictor(config, model_path=model_path, logger=logger)
    except FileNotFoundError as e:
        logger.error(f"Error loading model: {e}")
        logger.error("Please train a model first using: python train.py")
        return 1

    # Get model info
    model_info = predictor.get_model_info()
    logger.info("\nModel Information:")
    for key, value in model_info.items():
        logger.info(f"  {key}: {value}")

    # Make predictions
    logger.info(f"\nLoading input data from: {args.input}")

    try:
        results_df = predictor.predict_from_csv(
            csv_path=args.input,
            output_path=args.output,
            preprocess=preprocess
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return 1

    # Display summary
    predictions = results_df['predicted_Nexp'].values

    logger.info("\n" + "-" * 40)
    logger.info("Prediction Summary")
    logger.info("-" * 40)
    logger.info(f"Total samples: {len(predictions)}")
    logger.info(f"Mean prediction: {predictions.mean():.4f}")
    logger.info(f"Std prediction: {predictions.std():.4f}")
    logger.info(f"Min prediction: {predictions.min():.4f}")
    logger.info(f"Max prediction: {predictions.max():.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("Prediction Complete!")
    logger.info(f"Results saved to: {args.output}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
