"""
Advanced Class Balancing Module
Handles imbalanced data with multiple strategies
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional, Any
from imblearn.over_sampling import (
    SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE,
    RandomOverSampler, SMOTENC
)
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, EditedNearestNeighbours,
    NearMissing, CondensedNearestNeighbour, NearMiss
)
from imblearn.combine import SMOTETomek, SMOTEENN
from sklearn.model_selection import StratifiedKFold
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class AdvancedDataBalancer:
    """
    Industrial-grade class balancing with multiple strategies
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.balancing_report = {}
        self.original_distribution = {}
        self.balanced_distribution = {}

    def analyze_imbalance(self, y: pd.Series) -> Dict:
        """
        Analyze class imbalance in the dataset

        Args:
            y: Target variable

        Returns:
            Dictionary with imbalance metrics
        """
        value_counts = y.value_counts()
        total = len(y)

        analysis = {
            'class_distribution': value_counts.to_dict(),
            'class_percentages': (value_counts / total * 100).to_dict(),
            'imbalance_ratio': value_counts.max() / value_counts.min(),
            'minority_class': value_counts.idxmin(),
            'majority_class': value_counts.idxmax(),
            'total_samples': total
        }

        self.original_distribution = analysis

        logger.info(f"Class distribution: {analysis['class_distribution']}")
        logger.info(f"Imbalance ratio: {analysis['imbalance_ratio']:.2f}:1")

        return analysis

    def balance_with_smote(self,
                           X: pd.DataFrame,
                           y: pd.Series,
                           sampling_strategy: str = 'auto',
                           k_neighbors: int = 5,
                           variant: str = 'regular') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance data using SMOTE variants

        Args:
            X: Feature DataFrame
            y: Target Series
            sampling_strategy: Resampling strategy
            k_neighbors: Number of nearest neighbors
            variant: SMOTE variant to use

        Returns:
            Balanced X and y
        """
        logger.info(f"Applying {variant} SMOTE with k={k_neighbors}")

        # Get categorical columns for SMOTENC
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_indices = [X.columns.get_loc(col) for col in cat_cols] if cat_cols else None

        # Select SMOTE variant
        if variant == 'regular' and cat_indices:
            sampler = SMOTENC(
                categorical_features=cat_indices,
                random_state=self.random_state,
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors
            )
        elif variant == 'regular':
            sampler = SMOTE(
                random_state=self.random_state,
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors
            )
        elif variant == 'borderline':
            sampler = BorderlineSMOTE(
                random_state=self.random_state,
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                kind='borderline-1'
            )
        elif variant == 'svm':
            sampler = SVMSMOTE(
                random_state=self.random_state,
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors
            )
        elif variant == 'adasyn':
            sampler = ADASYN(
                random_state=self.random_state,
                sampling_strategy=sampling_strategy,
                n_neighbors=k_neighbors
            )
        else:
            raise ValueError(f"Unknown SMOTE variant: {variant}")

        # Apply sampling
        X_balanced, y_balanced = sampler.fit_resample(X, y)

        # Convert back to DataFrame/Series
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        y_balanced = pd.Series(y_balanced, name=y.name)

        # Update distribution
        self._update_balanced_distribution(y_balanced)

        logger.info(f"Balanced samples: {len(X_balanced)} (original: {len(X)})")

        return X_balanced, y_balanced

    def balance_with_undersampling(self,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   method: str = 'random',
                                   sampling_strategy: str = 'auto') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance data using undersampling techniques

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Undersampling method
            sampling_strategy: Resampling strategy

        Returns:
            Balanced X and y
        """
        logger.info(f"Applying {method} undersampling")

        # Select undersampling method
        if method == 'random':
            sampler = RandomUnderSampler(
                random_state=self.random_state,
                sampling_strategy=sampling_strategy
            )
        elif method == 'tomek':
            sampler = TomekLinks()
        elif method == 'enn':
            sampler = EditedNearestNeighbours()
        elif method == 'nearmiss':
            sampler = NearMiss(version=1)
        elif method == 'cnn':
            sampler = CondensedNearestNeighbour(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown undersampling method: {method}")

        # Apply sampling
        X_balanced, y_balanced = sampler.fit_resample(X, y)

        # Convert back to DataFrame/Series
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        y_balanced = pd.Series(y_balanced, name=y.name)

        # Update distribution
        self._update_balanced_distribution(y_balanced)

        logger.info(f"Balanced samples: {len(X_balanced)} (original: {len(X)})")

        return X_balanced, y_balanced

    def balance_with_combination(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 method: str = 'smote_tomek') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance data using combination of over and undersampling

        Args:
            X: Feature DataFrame
            y: Target Series
            method: Combination method

        Returns:
            Balanced X and y
        """
        logger.info(f"Applying {method} combination sampling")

        # Select combination method
        if method == 'smote_tomek':
            sampler = SMOTETomek(
                random_state=self.random_state,
                smote=SMOTE(random_state=self.random_state)
            )
        elif method == 'smote_enn':
            sampler = SMOTEENN(
                random_state=self.random_state,
                smote=SMOTE(random_state=self.random_state)
            )
        else:
            raise ValueError(f"Unknown combination method: {method}")

        # Apply sampling
        X_balanced, y_balanced = sampler.fit_resample(X, y)

        # Convert back to DataFrame/Series
        X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
        y_balanced = pd.Series(y_balanced, name=y.name)

        # Update distribution
        self._update_balanced_distribution(y_balanced)

        logger.info(f"Balanced samples: {len(X_balanced)} (original: {len(X)})")

        return X_balanced, y_balanced

    def adaptive_synthetic_sampling(self,
                                    X: pd.DataFrame,
                                    y: pd.Series,
                                    target_ratio: float = 1.0) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Adaptive synthetic sampling based on data characteristics

        Args:
            X: Feature DataFrame
            y: Target Series
            target_ratio: Target imbalance ratio

        Returns:
            Balanced X and y
        """
        logger.info("Applying adaptive synthetic sampling")

        # Analyze current imbalance
        imbalance = self.analyze_imbalance(y)
        current_ratio = imbalance['imbalance_ratio']

        # Determine sampling strategy
        if current_ratio > 10:
            # Severe imbalance - use combination
            logger.info("Severe imbalance detected - using SMOTETomek")
            return self.balance_with_combination(X, y, 'smote_tomek')

        elif current_ratio > 5:
            # High imbalance - use ADASYN
            logger.info("High imbalance detected - using ADASYN")
            return self.balance_with_smote(X, y, variant='adasyn')

        elif current_ratio > 3:
            # Moderate imbalance - use Borderline SMOTE
            logger.info("Moderate imbalance detected - using Borderline SMOTE")
            return self.balance_with_smote(X, y, variant='borderline')

        else:
            # Low imbalance - use regular SMOTE
            logger.info("Low imbalance detected - using regular SMOTE")
            return self.balance_with_smote(X, y, variant='regular')

    def get_sample_weights(self, y: pd.Series, strategy: str = 'balanced') -> np.ndarray:
        """
        Calculate sample weights for weighted training

        Args:
            y: Target variable
            strategy: Weighting strategy

        Returns:
            Array of sample weights
        """
        from sklearn.utils.class_weight import compute_sample_weight

        logger.info(f"Computing sample weights with strategy: {strategy}")

        sample_weights = compute_sample_weight(
            class_weight=strategy,
            y=y
        )

        # Log weight statistics
        unique_weights = np.unique(sample_weights)
        for class_val, weight in zip(np.unique(y), unique_weights):
            logger.info(f"Class {class_val}: weight = {weight:.3f}")

        return sample_weights

    def get_class_weights(self, y: pd.Series, strategy: str = 'balanced') -> Dict:
        """
        Calculate class weights for model training

        Args:
            y: Target variable
            strategy: Weighting strategy

        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        logger.info(f"Computing class weights with strategy: {strategy}")

        classes = np.unique(y)

        if strategy == 'balanced':
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y
            )
        elif strategy == 'balanced_subsample':
            # For tree-based models
            weights = compute_class_weight(
                class_weight='balanced_subsample',
                classes=classes,
                y=y
            )
        elif strategy == 'custom':
            # Custom weights based on sqrt of imbalance
            counts = np.bincount(y)
            weights = np.sqrt(counts.max() / counts)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        class_weights = dict(zip(classes, weights))

        logger.info(f"Class weights: {class_weights}")

        return class_weights

    def cross_validate_balancing(self,
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 methods: list,
                                 cv: int = 5) -> Dict:
        """
        Cross-validate different balancing methods

        Args:
            X: Feature DataFrame
            y: Target Series
            methods: List of balancing methods to test
            cv: Number of CV folds

        Returns:
            Dictionary with CV results
        """
        from sklearn.metrics import roc_auc_score, f1_score
        from sklearn.ensemble import RandomForestClassifier

        logger.info(f"Cross-validating {len(methods)} balancing methods")

        results = {}
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for method in methods:
            logger.info(f"Testing method: {method}")

            scores_auc = []
            scores_f1 = []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Apply balancing method
                if method == 'none':
                    X_balanced, y_balanced = X_train, y_train
                elif method == 'smote':
                    X_balanced, y_balanced = self.balance_with_smote(X_train, y_train)
                elif method == 'adasyn':
                    X_balanced, y_balanced = self.balance_with_smote(X_train, y_train, variant='adasyn')
                elif method == 'smote_tomek':
                    X_balanced, y_balanced = self.balance_with_combination(X_train, y_train)
                elif method == 'undersample':
                    X_balanced, y_balanced = self.balance_with_undersampling(X_train, y_train)
                else:
                    continue

                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                model.fit(X_balanced, y_balanced)

                # Evaluate
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = model.predict(X_val)

                scores_auc.append(roc_auc_score(y_val, y_pred_proba))
                scores_f1.append(f1_score(y_val, y_pred))

            results[method] = {
                'auc_mean': np.mean(scores_auc),
                'auc_std': np.std(scores_auc),
                'f1_mean': np.mean(scores_f1),
                'f1_std': np.std(scores_f1)
            }

            logger.info(f"{method} - AUC: {results[method]['auc_mean']:.3f} ± {results[method]['auc_std']:.3f}")

        # Find best method
        best_method = max(results.keys(), key=lambda k: results[k]['auc_mean'])
        results['best_method'] = best_method

        self.balancing_report['cv_results'] = results

        return results

    def _update_balanced_distribution(self, y_balanced: pd.Series):
        """Update balanced distribution statistics"""
        value_counts = y_balanced.value_counts()
        total = len(y_balanced)

        self.balanced_distribution = {
            'class_distribution': value_counts.to_dict(),
            'class_percentages': (value_counts / total * 100).to_dict(),
            'imbalance_ratio': value_counts.max() / value_counts.min(),
            'total_samples': total
        }

        self.balancing_report['distribution_change'] = {
            'original': self.original_distribution,
            'balanced': self.balanced_distribution
        }

    def save_balancing_report(self, filepath: str):
        """Save balancing report to file"""
        import json

        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        report = convert_types(self.balancing_report)

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Balancing report saved to {filepath}")