"""
Data preprocessing module for CFDST dataset.
Handles missing value imputation and data splitting.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import setup_logger


class DataPreprocessor:
    """
    Handles data preprocessing including imputation and train/test splitting.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize DataPreprocessor with configuration.

        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or setup_logger(__name__)

        self.target_col = config['target']['name']
        self.drop_cols = config.get('drop_columns', [])
        self.test_size = config['preprocessing']['test_size']
        self.random_state = config['preprocessing']['random_state']
        self.imputation_strategy = config['preprocessing']['imputation_strategy']
        self.scale_features = config['preprocessing']['scale_features']

        # Log transform settings (for handling wide dynamic range in target variable)
        self.log_transform_target = config['preprocessing'].get('log_transform_target', False)
        self.log_transform_epsilon = config['preprocessing'].get('log_transform_epsilon', 1.0)

        # Initialize imputer and scaler
        self.imputer: Optional[SimpleImputer] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_cols: Optional[List[str]] = None

        # Store original target statistics for inverse transform
        self.target_min: Optional[float] = None

    def drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with columns dropped
        """
        cols_to_drop = [col for col in self.drop_cols if col in df.columns]

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.logger.info(f"Dropped columns: {cols_to_drop}")

        return df

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Handle missing values using imputation.

        Args:
            df: Input DataFrame
            fit: Whether to fit the imputer (True for train, False for test)

        Returns:
            DataFrame with missing values imputed
        """
        # Check for missing values
        missing_count = df.isnull().sum().sum()

        if missing_count == 0:
            self.logger.info("No missing values found")
            return df

        self.logger.info(f"Found {missing_count} missing values")

        # Separate target and features
        if self.target_col in df.columns:
            target_data = df[self.target_col]
            feature_data = df.drop(columns=[self.target_col])
        else:
            target_data = None
            feature_data = df

        # Initialize or use existing imputer
        if fit or self.imputer is None:
            self.imputer = SimpleImputer(strategy=self.imputation_strategy)
            imputed_features = self.imputer.fit_transform(feature_data)
            self.logger.info(f"Fitted imputer with strategy: {self.imputation_strategy}")
        else:
            imputed_features = self.imputer.transform(feature_data)

        # Convert back to DataFrame
        feature_df = pd.DataFrame(
            imputed_features,
            columns=feature_data.columns,
            index=feature_data.index
        )

        # Recombine with target
        if target_data is not None:
            result = pd.concat([feature_df, target_data], axis=1)
        else:
            result = feature_df

        self.logger.info("Missing values imputed successfully")
        return result

    def scale_features_func(
        self,
        df: pd.DataFrame,
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Scale features using StandardScaler.

        Args:
            df: Input DataFrame
            fit: Whether to fit the scaler

        Returns:
            DataFrame with scaled features
        """
        if not self.scale_features:
            return df

        # Separate target and features
        if self.target_col in df.columns:
            target_data = df[self.target_col]
            feature_data = df.drop(columns=[self.target_col])
        else:
            target_data = None
            feature_data = df

        # Initialize or use existing scaler
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(feature_data)
            self.logger.info("Fitted StandardScaler")
        else:
            scaled_features = self.scaler.transform(feature_data)

        # Convert back to DataFrame
        feature_df = pd.DataFrame(
            scaled_features,
            columns=feature_data.columns,
            index=feature_data.index
        )

        # Recombine with target
        if target_data is not None:
            result = pd.concat([feature_df, target_data], axis=1)
        else:
            result = feature_df

        return result

    def split_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        apply_log_transform: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.

        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            apply_log_transform: Whether to apply log transform to target

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X = df[feature_cols].values
        y = df[self.target_col].values

        # Apply log transform if enabled (for training)
        if apply_log_transform and self.log_transform_target:
            y = self.transform_target(y, fit=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )

        self.logger.info(
            f"Data split: Train={len(X_train)} samples, Test={len(X_test)} samples"
        )

        # Store feature columns for later use
        self.feature_cols = feature_cols

        return X_train, X_test, y_train, y_test

    def preprocess(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        fit: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Complete preprocessing pipeline.

        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            fit: Whether to fit transformers

        Returns:
            Tuple of (processed DataFrame, updated feature_cols)
        """
        # Drop specified columns
        df = self.drop_columns(df.copy())

        # Update feature columns after dropping
        updated_features = [col for col in feature_cols if col in df.columns]

        # Handle missing values
        df = self.handle_missing_values(df, fit=fit)

        # Scale features if enabled
        if self.scale_features:
            df = self.scale_features_func(df, fit=fit)

        self.logger.info(f"Preprocessing complete. Features: {len(updated_features)}")

        return df, updated_features

    def transform_target(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Apply log transform to target variable if enabled.

        This helps handle wide dynamic ranges (e.g., 228 ~ 15850 kN) by:
        - Compressing the scale of large values
        - Reducing heteroscedasticity (variance dependence on magnitude)
        - Making the model treat small and large values more equally

        Args:
            y: Target values
            fit: Whether to compute and store statistics (for inverse transform)

        Returns:
            Transformed target values
        """
        if not self.log_transform_target:
            return y

        y = np.array(y).astype(float)

        if fit:
            self.target_min = np.min(y)
            self.logger.info(f"Target log transform enabled (epsilon={self.log_transform_epsilon})")
            self.logger.info(f"Target range before transform: {self.target_min:.2f} ~ {np.max(y):.2f}")

        # Apply log transform with epsilon offset to avoid log(0)
        # Using log1p for numerical stability: log(1 + x) ≈ log(x) for large x
        y_transformed = np.log(y + self.log_transform_epsilon)

        if fit:
            self.logger.info(f"Target range after transform: {np.min(y_transformed):.4f} ~ {np.max(y_transformed):.4f}")

        return y_transformed

    def inverse_transform_target(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform target from log space back to original space.

        Args:
            y_transformed: Transformed target values (log space)

        Returns:
            Original scale target values
        """
        if not self.log_transform_target:
            return y_transformed

        y_transformed = np.array(y_transformed).astype(float)

        # Inverse of log: exp(y) - epsilon
        y_original = np.exp(y_transformed) - self.log_transform_epsilon

        # Ensure non-negative values (physical constraint for bearing capacity)
        y_original = np.maximum(y_original, 0)

        return y_original

    def get_preprocessor_state(self) -> Dict[str, Any]:
        """
        Get the state of preprocessors for saving.

        Returns:
            Dictionary containing preprocessor states
        """
        state = {
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_cols': self.feature_cols,
            'config': self.config,
            'target_min': self.target_min,
            'log_transform_target': self.log_transform_target,
            'log_transform_epsilon': self.log_transform_epsilon
        }
        return state

    def set_preprocessor_state(self, state: Dict[str, Any]) -> None:
        """
        Set the state of preprocessors from saved state.

        Args:
            state: Dictionary containing preprocessor states
        """
        self.imputer = state.get('imputer')
        self.scaler = state.get('scaler')
        self.feature_cols = state.get('feature_cols')
        self.target_min = state.get('target_min')
        self.log_transform_target = state.get('log_transform_target', False)
        self.log_transform_epsilon = state.get('log_transform_epsilon', 1.0)

    def get_feature_importance_mask(
        self,
        selected_features: List[str],
        all_features: List[str]
    ) -> List[int]:
        """
        Get mask for selected features.

        Args:
            selected_features: List of selected feature names
            all_features: List of all feature names

        Returns:
            List of indices of selected features
        """
        return [all_features.index(f) for f in selected_features if f in all_features]
