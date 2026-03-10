"""
Data loading and validation module for CFDST dataset.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .utils import setup_logger


class DataLoader:
    """
    Handles loading and basic validation of CFDST data from CSV files.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize DataLoader with configuration.

        Args:
            config: Configuration dictionary containing paths and settings
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)

        self.data_dir = Path(config['paths']['data_dir'])
        self.target_col = config['target']['name']
        self.drop_cols = config.get('drop_columns', [])

    def load_csv(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            filepath: Path to CSV file. If None, uses config['paths']['raw_data']

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if filepath is None:
            filepath = self.config['paths']['raw_data']

        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        self.logger.info(f"Loading data from: {filepath}")

        # Try different encodings
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']
        df = None

        for encoding in encodings:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                self.logger.debug(f"Successfully loaded with encoding: {encoding}")
                break
            except UnicodeDecodeError:
                continue

        if df is None:
            raise ValueError(f"Could not load file with any encoding: {filepath}")

        self.logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

        return df

    def clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean column names by stripping whitespace and handling special characters.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned column names
        """
        original_cols = list(df.columns)

        # Clean column names
        df.columns = df.columns.str.strip()

        # Log changes
        changed = [(orig, new) for orig, new in zip(original_cols, df.columns) if orig != new]
        if changed:
            self.logger.debug(f"Cleaned column names: {changed}")

        return df

    def validate_data(self, df: pd.DataFrame, required_cols: Optional[List[str]] = None) -> bool:
        """
        Validate the loaded data.

        Args:
            df: Input DataFrame
            required_cols: List of required column names

        Returns:
            True if validation passes

        Raises:
            ValueError: If validation fails
        """
        self.logger.info("Validating data...")

        # Check if DataFrame is empty
        if df.empty:
            raise ValueError("DataFrame is empty")

        # Check for target column
        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data. "
                           f"Available columns: {list(df.columns)}")

        # Check for required columns
        if required_cols:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

        # Check for duplicate columns
        duplicates = df.columns[df.columns.duplicated()].tolist()
        if duplicates:
            self.logger.warning(f"Duplicate columns found: {duplicates}")

        # Check data types
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if self.target_col not in numeric_cols:
            raise ValueError(f"Target column '{self.target_col}' must be numeric")

        self.logger.info("Data validation passed")
        return True

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get feature column names (excluding target and columns to drop).

        Args:
            df: Input DataFrame

        Returns:
            List of feature column names
        """
        all_cols = df.columns.tolist()

        # Remove target and columns to drop
        feature_cols = [
            col for col in all_cols
            if col != self.target_col and col not in self.drop_cols
        ]

        self.logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        self.logger.info(f"Dropped columns: {self.drop_cols}")

        return feature_cols

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic information about the dataset.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with data information
        """
        info = {
            'n_samples': len(df),
            'n_features': len(df.columns) - 1,  # Excluding target
            'target': self.target_col,
            'target_stats': df[self.target_col].describe().to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'columns': df.columns.tolist()
        }

        return info

    def save_processed_data(self, df: pd.DataFrame, filepath: Optional[str] = None) -> None:
        """
        Save processed data to CSV.

        Args:
            df: DataFrame to save
            filepath: Output file path. If None, uses config['paths']['processed_data']
        """
        if filepath is None:
            filepath = self.config['paths']['processed_data']

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        self.logger.info(f"Saved processed data to: {filepath}")

    def load_and_prepare(
        self,
        filepath: Optional[str] = None,
        validate: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete pipeline: load, clean, validate, and return data with feature columns.

        Args:
            filepath: Path to CSV file
            validate: Whether to validate data

        Returns:
            Tuple of (DataFrame, feature_columns)
        """
        # Load data
        df = self.load_csv(filepath)

        # Clean column names
        df = self.clean_column_names(df)

        # Validate
        if validate:
            self.validate_data(df)

        # Get feature columns
        feature_cols = self.get_feature_columns(df)

        # Log data info
        info = self.get_data_info(df)
        self.logger.info(f"Dataset: {info['n_samples']} samples, {len(feature_cols)} features")

        return df, feature_cols
