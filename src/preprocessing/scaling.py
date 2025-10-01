"""
Advanced Feature Scaling Module
Handles different scaling strategies for various feature types
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
    Normalizer
)
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from category_encoders import TargetEncoder, BinaryEncoder, HashingEncoder
import joblib
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class AdvancedFeatureScaler:
    """
    Industrial-grade feature scaling with multiple strategies
    """

    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.scaling_report = {}
        self.feature_types = {}
        self.fitted = False

    def detect_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically detect feature types for appropriate scaling

        Args:
            df: Input DataFrame

        Returns:
            Dictionary of feature types
        """
        logger.info("Detecting feature types")

        feature_types = {
            'numeric_normal': [],
            'numeric_skewed': [],
            'numeric_heavy_tail': [],
            'binary': [],
            'categorical_low': [],
            'categorical_high': [],
            'ordinal': [],
            'text': [],
            'id': []
        }

        for col in df.columns:
            # Skip ID columns
            if 'id' in col.lower() or col.lower().endswith('_id'):
                feature_types['id'].append(col)
                continue

            # Check data type
            if df[col].dtype in ['object', 'category']:
                unique_count = df[col].nunique()

                if unique_count == 2:
                    feature_types['binary'].append(col)
                elif unique_count <= 10:
                    feature_types['categorical_low'].append(col)
                elif unique_count <= 50:
                    feature_types['categorical_high'].append(col)
                else:
                    feature_types['text'].append(col)

            elif np.issubdtype(df[col].dtype, np.number):
                # Check if binary
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1}):
                    feature_types['binary'].append(col)
                else:
                    # Check distribution
                    skewness = df[col].skew()
                    kurtosis = df[col].kurtosis()

                    if abs(skewness) < 0.5:
                        feature_types['numeric_normal'].append(col)
                    elif abs(skewness) < 2:
                        feature_types['numeric_skewed'].append(col)
                    else:
                        feature_types['numeric_heavy_tail'].append(col)

        self.feature_types = feature_types

        # Log feature type distribution
        for feat_type, features in feature_types.items():
            if features:
                logger.info(f"{feat_type}: {len(features)} features")

        return feature_types

    def fit_transform(self,
                      df: pd.DataFrame,
                      y: Optional[pd.Series] = None,
                      scaling_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Fit and transform features with appropriate scaling

        Args:
            df: Input DataFrame
            y: Target variable (for target encoding)
            scaling_config: Custom scaling configuration

        Returns:
            Scaled DataFrame
        """
        logger.info("Fitting and transforming features")

        # Detect feature types
        if not self.feature_types:
            self.detect_feature_types(df)

        # Default scaling configuration
        if scaling_config is None:
            scaling_config = {
                'numeric_normal': 'standard',
                'numeric_skewed': 'quantile',
                'numeric_heavy_tail': 'robust',
                'binary': 'passthrough',
                'categorical_low': 'onehot',
                'categorical_high': 'target' if y is not None else 'binary',
                'ordinal': 'ordinal',
                'text': 'hashing',
                'id': 'passthrough'
            }

        # Initialize result DataFrame
        result_dfs = []

        # Process each feature type
        for feat_type, features in self.feature_types.items():
            if not features:
                continue

            method = scaling_config.get(feat_type, 'passthrough')

            if method == 'passthrough':
                result_dfs.append(df[features])

            elif feat_type.startswith('numeric'):
                scaled_df = self._scale_numeric_features(
                    df[features],
                    method=method,
                    feat_type=feat_type
                )
                result_dfs.append(scaled_df)

            elif feat_type in ['binary', 'categorical_low', 'categorical_high']:
                encoded_df = self._encode_categorical_features(
                    df[features],
                    y=y,
                    method=method,
                    feat_type=feat_type
                )
                result_dfs.append(encoded_df)

        # Combine all processed features
        result = pd.concat(result_dfs, axis=1)

        self.fitted = True

        logger.info(f"Scaled features from {len(df.columns)} to {len(result.columns)} columns")

        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scalers

        Args:
            df: Input DataFrame

        Returns:
            Scaled DataFrame
        """
        if not self.fitted:
            raise ValueError("Scaler must be fitted before transform")

        logger.info("Transforming features")

        result_dfs = []

        # Process each feature type
        for feat_type, features in self.feature_types.items():
            if not features:
                continue

            # Get features that exist in input df
            existing_features = [f for f in features if f in df.columns]
            if not existing_features:
                continue

            if feat_type == 'id' or feat_type not in self.scalers:
                result_dfs.append(df[existing_features])

            elif feat_type.startswith('numeric'):
                scaler = self.scalers[feat_type]
                scaled_data = scaler.transform(df[existing_features])
                scaled_df = pd.DataFrame(
                    scaled_data,
                    columns=existing_features,
                    index=df.index
                )
                result_dfs.append(scaled_df)

            elif feat_type in self.encoders:
                encoder = self.encoders[feat_type]

                if hasattr(encoder, 'transform'):
                    encoded_data = encoder.transform(df[existing_features])

                    if hasattr(encoder, 'get_feature_names_out'):
                        columns = encoder.get_feature_names_out()
                    else:
                        columns = self._get_encoded_columns(feat_type, existing_features)

                    encoded_df = pd.DataFrame(
                        encoded_data,
                        columns=columns,
                        index=df.index
                    )
                    result_dfs.append(encoded_df)
                else:
                    result_dfs.append(df[existing_features])

        result = pd.concat(result_dfs, axis=1)

        return result

    def _scale_numeric_features(self,
                                df: pd.DataFrame,
                                method: str,
                                feat_type: str) -> pd.DataFrame:
        """
        Scale numeric features

        Args:
            df: DataFrame with numeric features
            method: Scaling method
            feat_type: Feature type identifier

        Returns:
            Scaled DataFrame
        """
        logger.info(f"Scaling {len(df.columns)} {feat_type} features with {method}")

        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'maxabs':
            scaler = MaxAbsScaler()
        elif method == 'quantile':
            scaler = QuantileTransformer(
                n_quantiles=min(1000, len(df)),
                output_distribution='normal'
            )
        elif method == 'power':
            scaler = PowerTransformer(method='yeo-johnson')
        elif method == 'normalizer':
            scaler = Normalizer()
        else:
            logger.warning(f"Unknown scaling method: {method}, using StandardScaler")
            scaler = StandardScaler()

        # Fit and transform
        scaled_data = scaler.fit_transform(df)

        # Store scaler
        self.scalers[feat_type] = scaler

        # Create DataFrame
        scaled_df = pd.DataFrame(
            scaled_data,
            columns=df.columns,
            index=df.index
        )

        # Log scaling statistics
        self.scaling_report[feat_type] = {
            'method': method,
            'features': df.columns.tolist(),
            'statistics': {}
        }

        if hasattr(scaler, 'mean_'):
            self.scaling_report[feat_type]['statistics']['mean'] = scaler.mean_.tolist()
        if hasattr(scaler, 'scale_'):
            self.scaling_report[feat_type]['statistics']['scale'] = scaler.scale_.tolist()

        return scaled_df

    def _encode_categorical_features(self,
                                     df: pd.DataFrame,
                                     y: Optional[pd.Series],
                                     method: str,
                                     feat_type: str) -> pd.DataFrame:
        """
        Encode categorical features

        Args:
            df: DataFrame with categorical features
            y: Target variable
            method: Encoding method
            feat_type: Feature type identifier

        Returns:
            Encoded DataFrame
        """
        logger.info(f"Encoding {len(df.columns)} {feat_type} features with {method}")

        if method == 'onehot':
            # One-hot encoding
            encoded_df = pd.get_dummies(df, drop_first=True)
            self.encoders[feat_type] = 'onehot'

        elif method == 'target' and y is not None:
            # Target encoding
            encoder = TargetEncoder()
            encoded_data = encoder.fit_transform(df, y)
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=df.columns,
                index=df.index
            )
            self.encoders[feat_type] = encoder

        elif method == 'binary':
            # Binary encoding
            encoder = BinaryEncoder()
            encoded_df = encoder.fit_transform(df)
            self.encoders[feat_type] = encoder

        elif method == 'hashing':
            # Hashing encoding
            encoder = HashingEncoder(n_components=32)
            encoded_df = encoder.fit_transform(df)
            self.encoders[feat_type] = encoder

        elif method == 'ordinal':
            # Ordinal encoding
            encoder = OrdinalEncoder()
            encoded_data = encoder.fit_transform(df)
            encoded_df = pd.DataFrame(
                encoded_data,
                columns=df.columns,
                index=df.index
            )
            self.encoders[feat_type] = encoder

        else:
            # Default to label encoding
            encoded_df = df.copy()
            encoders = {}
            for col in df.columns:
                le = LabelEncoder()
                encoded_df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
            self.encoders[feat_type] = encoders

        # Store encoded columns for transform
        self._store_encoded_columns(feat_type, encoded_df.columns.tolist())

        return encoded_df

    def _store_encoded_columns(self, feat_type: str, columns: List[str]):
        """Store encoded column names for transform"""
        if 'encoded_columns' not in self.scaling_report:
            self.scaling_report['encoded_columns'] = {}
        self.scaling_report['encoded_columns'][feat_type] = columns

    def _get_encoded_columns(self, feat_type: str, original_features: List[str]) -> List[str]:
        """Get encoded column names"""
        if 'encoded_columns' in self.scaling_report and feat_type in self.scaling_report['encoded_columns']:
            return self.scaling_report['encoded_columns'][feat_type]
        return original_features

    def save_scalers(self, filepath: str):
        """
        Save fitted scalers and encoders

        Args:
            filepath: Path to save file
        """
        save_dict = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_types': self.feature_types,
            'scaling_report': self.scaling_report,
            'fitted': self.fitted
        }

        joblib.dump(save_dict, filepath)
        logger.info(f"Scalers saved to {filepath}")

    def load_scalers(self, filepath: str):
        """
        Load fitted scalers and encoders

        Args:
            filepath: Path to saved file
        """
        save_dict = joblib.load(filepath)

        self.scalers = save_dict['scalers']
        self.encoders = save_dict['encoders']
        self.feature_types = save_dict['feature_types']
        self.scaling_report = save_dict['scaling_report']
        self.fitted = save_dict['fitted']

        logger.info(f"Scalers loaded from {filepath}")

    def get_feature_names_out(self) -> List[str]:
        """
        Get output feature names after transformation

        Returns:
            List of feature names
        """
        feature_names = []

        for feat_type, features in self.feature_types.items():
            if feat_type in ['id', 'passthrough']:
                feature_names.extend(features)
            elif 'encoded_columns' in self.scaling_report and feat_type in self.scaling_report['encoded_columns']:
                feature_names.extend(self.scaling_report['encoded_columns'][feat_type])
            else:
                feature_names.extend(features)

        return feature_names

    def get_scaling_report(self) -> Dict:
        """
        Get comprehensive scaling report

        Returns:
            Dictionary with scaling details
        """
        report = {
            'feature_types': {k: len(v) for k, v in self.feature_types.items()},
            'scaling_methods': {},
            'total_features_in': sum(len(v) for v in self.feature_types.values()),
            'total_features_out': len(self.get_feature_names_out())
        }

        for feat_type in self.scalers:
            scaler = self.scalers[feat_type]
            report['scaling_methods'][feat_type] = type(scaler).__name__

        for feat_type in self.encoders:
            encoder = self.encoders[feat_type]
            if isinstance(encoder, dict):
                report['scaling_methods'][feat_type] = 'LabelEncoder'
            else:
                report['scaling_methods'][feat_type] = type(encoder).__name__

        return report