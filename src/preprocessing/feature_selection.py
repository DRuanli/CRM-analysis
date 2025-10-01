"""
Advanced Feature Selection Module for Phase 2
Handles correlation removal, RFE, and SHAP-based selection
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import shap
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class AdvancedFeatureSelector:
    """
    Industrial-grade feature selection with multiple strategies
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features = []
        self.feature_scores = {}
        self.removed_features = {}
        self.selection_report = {}

    def remove_correlated_features(self,
                                  df: pd.DataFrame,
                                  threshold: float = 0.95,
                                  target_col: Optional[str] = None) -> pd.DataFrame:
        """
        Remove highly correlated features while keeping the most predictive ones

        Args:
            df: Input DataFrame
            threshold: Correlation threshold (default 0.95)
            target_col: Target column to preserve correlation with

        Returns:
            DataFrame with reduced features
        """
        logger.info(f"Removing correlated features with threshold {threshold}")

        # Separate numeric features
        numeric_df = df.select_dtypes(include=[np.number])
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr().abs()

        # If target column exists, calculate correlation with target
        target_corr = None
        if target_col and target_col in numeric_df.columns:
            target_corr = numeric_df.corr()[target_col].abs()
            # Remove target from correlation analysis
            corr_matrix = corr_matrix.drop(target_col, axis=0).drop(target_col, axis=1)
            numeric_df = numeric_df.drop(columns=[target_col])

        # Create upper triangle mask
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features to drop
        to_drop = set()
        correlation_pairs = []

        for column in upper_tri.columns:
            high_corr = upper_tri[column][upper_tri[column] > threshold]

            for corr_feature in high_corr.index:
                # Keep feature with higher correlation to target
                if target_corr is not None:
                    if target_corr[column] > target_corr[corr_feature]:
                        to_drop.add(corr_feature)
                        keep_feature = column
                    else:
                        to_drop.add(column)
                        keep_feature = corr_feature
                else:
                    # Keep feature with higher variance
                    if numeric_df[column].var() > numeric_df[corr_feature].var():
                        to_drop.add(corr_feature)
                        keep_feature = column
                    else:
                        to_drop.add(column)
                        keep_feature = corr_feature

                correlation_pairs.append({
                    'feature1': column,
                    'feature2': corr_feature,
                    'correlation': high_corr[corr_feature],
                    'dropped': list(to_drop & {column, corr_feature})[0] if to_drop & {column, corr_feature} else None,
                    'kept': keep_feature
                })

        # Store removed features info
        self.removed_features['correlation'] = list(to_drop)
        self.selection_report['correlation_analysis'] = {
            'pairs_found': len(correlation_pairs),
            'features_removed': len(to_drop),
            'features_kept': len(numeric_df.columns) - len(to_drop)
        }

        logger.info(f"Removed {len(to_drop)} correlated features")
        logger.info(f"Remaining features: {len(numeric_df.columns) - len(to_drop)}")

        # Combine back with non-numeric columns and target
        result_cols = [col for col in numeric_df.columns if col not in to_drop] + non_numeric_cols
        if target_col and target_col in df.columns:
            result_cols.append(target_col)

        return df[result_cols]

    def select_k_best(self,
                     X: pd.DataFrame,
                     y: pd.Series,
                     k: int = 50,
                     score_func=f_classif) -> pd.DataFrame:
        """
        Select K best features using statistical tests

        Args:
            X: Feature DataFrame
            y: Target Series
            k: Number of features to select
            score_func: Scoring function (f_classif or mutual_info_classif)

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Selecting {k} best features using {score_func.__name__}")

        # Handle non-numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # Apply SelectKBest
        selector = SelectKBest(score_func=score_func, k=min(k, len(numeric_cols)))
        X_selected = selector.fit_transform(X_numeric, y)

        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = X_numeric.columns[selected_mask].tolist()

        # Store scores
        self.feature_scores['kbest'] = dict(zip(
            X_numeric.columns,
            selector.scores_
        ))

        logger.info(f"Selected {len(selected_features)} features")

        return X[selected_features]

    def recursive_feature_elimination(self,
                                     X: pd.DataFrame,
                                     y: pd.Series,
                                     n_features: int = 50,
                                     cv: int = 5,
                                     estimator=None) -> pd.DataFrame:
        """
        Apply RFE with cross-validation

        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            cv: Number of cross-validation folds
            estimator: Base estimator (default: RandomForest)

        Returns:
            DataFrame with selected features
        """
        logger.info(f"Applying RFE with {cv}-fold CV")

        # Use only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # Default estimator
        if estimator is None:
            estimator = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )

        # Apply RFECV
        rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=cv,
            scoring='roc_auc',
            min_features_to_select=n_features,
            n_jobs=-1
        )

        rfecv.fit(X_numeric, y)

        # Get selected features
        selected_features = X_numeric.columns[rfecv.support_].tolist()

        # Store rankings
        self.feature_scores['rfe_ranking'] = dict(zip(
            X_numeric.columns,
            rfecv.ranking_
        ))

        self.selection_report['rfe'] = {
            'optimal_features': rfecv.n_features_,
            'cv_scores': rfecv.cv_results_['mean_test_score'].tolist()
        }

        logger.info(f"RFE selected {len(selected_features)} features")
        logger.info(f"Optimal number of features: {rfecv.n_features_}")

        return X[selected_features]

    def shap_based_selection(self,
                           X: pd.DataFrame,
                           y: pd.Series,
                           n_features: int = 50,
                           model=None) -> pd.DataFrame:
        """
        Select features based on SHAP values

        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            model: Model to use for SHAP (default: RandomForest)

        Returns:
            DataFrame with selected features
        """
        logger.info("Calculating SHAP values for feature selection")

        # Use only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # Default model
        if model is None:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_numeric, y)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_numeric)

        # For binary classification, use class 1 SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calculate mean absolute SHAP values
        shap_importance = pd.DataFrame({
            'feature': X_numeric.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        # Select top n features
        selected_features = shap_importance.head(n_features)['feature'].tolist()

        # Store SHAP scores
        self.feature_scores['shap'] = dict(zip(
            shap_importance['feature'],
            shap_importance['importance']
        ))

        logger.info(f"SHAP selected {len(selected_features)} features")

        return X[selected_features]

    def variance_threshold_selection(self,
                                    X: pd.DataFrame,
                                    threshold: float = 0.01) -> pd.DataFrame:
        """
        Remove low variance features

        Args:
            X: Feature DataFrame
            threshold: Variance threshold

        Returns:
            DataFrame with high variance features
        """
        from sklearn.feature_selection import VarianceThreshold

        logger.info(f"Removing features with variance < {threshold}")

        # Use only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_numeric)

        # Get selected features
        selected_features = X_numeric.columns[selector.get_support()].tolist()

        # Store removed features
        removed = X_numeric.columns[~selector.get_support()].tolist()
        self.removed_features['low_variance'] = removed

        logger.info(f"Removed {len(removed)} low variance features")

        return X[selected_features]

    def ensemble_selection(self,
                          X: pd.DataFrame,
                          y: pd.Series,
                          n_features: int = 50,
                          methods: List[str] = None) -> pd.DataFrame:
        """
        Ensemble feature selection combining multiple methods

        Args:
            X: Feature DataFrame
            y: Target Series
            n_features: Number of features to select
            methods: List of methods to use

        Returns:
            DataFrame with selected features
        """
        if methods is None:
            methods = ['kbest', 'rfe', 'shap', 'forest']

        logger.info(f"Ensemble selection using methods: {methods}")

        feature_votes = {}

        # Apply each method
        if 'kbest' in methods:
            X_kbest = self.select_k_best(X, y, k=n_features*2)
            for feat in X_kbest.columns:
                feature_votes[feat] = feature_votes.get(feat, 0) + 1

        if 'rfe' in methods:
            X_rfe = self.recursive_feature_elimination(X, y, n_features=n_features*2)
            for feat in X_rfe.columns:
                feature_votes[feat] = feature_votes.get(feat, 0) + 1

        if 'shap' in methods:
            X_shap = self.shap_based_selection(X, y, n_features=n_features*2)
            for feat in X_shap.columns:
                feature_votes[feat] = feature_votes.get(feat, 0) + 1

        if 'forest' in methods:
            # Feature importance from Random Forest
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_cols]

            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X_numeric, y)

            importance_df = pd.DataFrame({
                'feature': X_numeric.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)

            top_features = importance_df.head(n_features*2)['feature'].tolist()
            for feat in top_features:
                feature_votes[feat] = feature_votes.get(feat, 0) + 1

        # Select features with most votes
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, votes in sorted_features[:n_features]]

        self.selected_features = selected_features
        self.selection_report['ensemble'] = {
            'total_features_considered': len(feature_votes),
            'features_selected': len(selected_features),
            'voting_distribution': dict(sorted_features[:n_features])
        }

        logger.info(f"Ensemble selected {len(selected_features)} features")

        return X[selected_features]

    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Get comprehensive feature importance report

        Returns:
            DataFrame with feature scores from all methods
        """
        # Combine all scores
        all_features = set()
        for scores in self.feature_scores.values():
            all_features.update(scores.keys())

        # Create report DataFrame
        report_data = []
        for feature in all_features:
            row = {'feature': feature}
            for method, scores in self.feature_scores.items():
                row[f'{method}_score'] = scores.get(feature, np.nan)
            report_data.append(row)

        report_df = pd.DataFrame(report_data)

        # Add ensemble score
        if self.selected_features:
            report_df['selected'] = report_df['feature'].isin(self.selected_features)

        return report_df.sort_values('feature')

    def save_selection_report(self, filepath: str):
        """Save feature selection report to file"""
        import json

        report = {
            'selected_features': self.selected_features,
            'removed_features': self.removed_features,
            'selection_report': self.selection_report,
            'feature_scores': {
                method: {str(k): float(v) for k, v in scores.items()}
                for method, scores in self.feature_scores.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Selection report saved to {filepath}")