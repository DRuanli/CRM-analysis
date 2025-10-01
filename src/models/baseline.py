"""
Baseline Machine Learning Models
Logistic Regression, Decision Tree, Naive Bayes with comprehensive tuning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
    cross_val_score, StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import mlflow
import mlflow.sklearn
from loguru import logger
import warnings

warnings.filterwarnings('ignore')


class BaselineModels:
    """
    Collection of baseline models with hyperparameter tuning
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.cv_scores = {}
        self.trained_models = {}

    def train_logistic_regression(self,
                                  X_train: pd.DataFrame,
                                  y_train: pd.Series,
                                  X_val: Optional[pd.DataFrame] = None,
                                  y_val: Optional[pd.Series] = None,
                                  tune_hyperparameters: bool = True,
                                  cv_folds: int = 5) -> Dict:
        """
        Train Logistic Regression with regularization

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            tune_hyperparameters: Whether to tune hyperparameters
            cv_folds: Number of CV folds

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Logistic Regression")

        if tune_hyperparameters:
            # Hyperparameter grid
            param_grid = {
                'penalty': ['l1', 'l2', 'elasticnet'],
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'solver': ['saga'],  # Supports all penalties
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],  # For elasticnet
                'max_iter': [500],
                'class_weight': [None, 'balanced']
            }

            # Base model
            base_model = LogisticRegression(random_state=self.random_state)

            # Grid search with stratified CV
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

            # Custom scorer
            def custom_scorer(estimator, X, y):
                # Prioritize recall for minority class
                y_pred = estimator.predict(X)
                recall = recall_score(y, y_pred, pos_label=1)
                precision = precision_score(y, y_pred, pos_label=1)
                # F2 score gives more weight to recall
                f2 = (5 * precision * recall) / (4 * precision + recall) if (precision + recall) > 0 else 0
                return f2

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )

            # Fit grid search
            grid_search.fit(X_train, y_train)

            # Best model
            model = grid_search.best_estimator_
            self.best_params['logistic_regression'] = grid_search.best_params_

            logger.info(f"Best params: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        else:
            # Use LogisticRegressionCV for automatic regularization tuning
            model = LogisticRegressionCV(
                cv=cv_folds,
                penalty='l2',
                solver='lbfgs',
                class_weight='balanced',
                random_state=self.random_state,
                max_iter=500,
                scoring='roc_auc'
            )
            model.fit(X_train, y_train)

        # Store model
        self.models['logistic_regression'] = model

        # Evaluate on training set
        train_metrics = self._evaluate_model(model, X_train, y_train, 'train')

        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate_model(model, X_val, y_val, 'validation')

        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring='roc_auc',
            n_jobs=-1
        )

        self.cv_scores['logistic_regression'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }

        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': model.coef_[0],
            'abs_coefficient': np.abs(model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)

        # Log to MLflow
        with mlflow.start_run(run_name="LogisticRegression"):
            mlflow.log_params(self.best_params.get('logistic_regression', {}))
            mlflow.log_metrics(train_metrics)
            if val_metrics:
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.sklearn.log_model(model, "model")

        return {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'cv_scores': self.cv_scores['logistic_regression'],
            'feature_importance': feature_importance
        }

    def train_decision_tree(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None,
                            tune_hyperparameters: bool = True,
                            cv_folds: int = 5) -> Dict:
        """
        Train Decision Tree with pruning

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            tune_hyperparameters: Whether to tune hyperparameters
            cv_folds: Number of CV folds

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Decision Tree")

        if tune_hyperparameters:
            # Hyperparameter grid
            param_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': [3, 5, 7, 10, 15, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': [None, 'balanced'],
                'ccp_alpha': [0.0, 0.01, 0.02, 0.05]  # Pruning parameter
            }

            # Base model
            base_model = DecisionTreeClassifier(random_state=self.random_state)

            # Randomized search for efficiency
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

            random_search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=50,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0,
                random_state=self.random_state
            )

            # Fit search
            random_search.fit(X_train, y_train)

            # Best model
            model = random_search.best_estimator_
            self.best_params['decision_tree'] = random_search.best_params_

            logger.info(f"Best params: {random_search.best_params_}")
            logger.info(f"Best CV score: {random_search.best_score_:.4f}")

        else:
            # Default model with pruning
            model = DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=5,
                class_weight='balanced',
                random_state=self.random_state
            )
            model.fit(X_train, y_train)

        # Perform cost complexity pruning
        path = model.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas[:-1]  # Remove the last value (full tree)

        if len(ccp_alphas) > 0:
            # Train trees with different alpha values
            pruned_trees = []
            for ccp_alpha in ccp_alphas[:10]:  # Limit to 10 values
                pruned_tree = DecisionTreeClassifier(
                    random_state=self.random_state,
                    ccp_alpha=ccp_alpha,
                    **model.get_params()
                )
                pruned_tree.fit(X_train, y_train)
                score = roc_auc_score(y_train, pruned_tree.predict_proba(X_train)[:, 1])
                pruned_trees.append((pruned_tree, score, ccp_alpha))

            # Select best pruned tree
            best_tree = max(pruned_trees, key=lambda x: x[1])
            model = best_tree[0]
            logger.info(f"Best pruning alpha: {best_tree[2]:.4f}")

        # Store model
        self.models['decision_tree'] = model

        # Evaluate
        train_metrics = self._evaluate_model(model, X_train, y_train, 'train')

        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate_model(model, X_val, y_val, 'validation')

        # Cross-validation scores
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds, scoring='roc_auc',
            n_jobs=-1
        )

        self.cv_scores['decision_tree'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Log to MLflow
        with mlflow.start_run(run_name="DecisionTree"):
            mlflow.log_params(self.best_params.get('decision_tree', {}))
            mlflow.log_metrics(train_metrics)
            if val_metrics:
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.sklearn.log_model(model, "model")

        return {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'cv_scores': self.cv_scores['decision_tree'],
            'feature_importance': feature_importance
        }

    def train_naive_bayes(self,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_val: Optional[pd.DataFrame] = None,
                          y_val: Optional[pd.Series] = None,
                          nb_type: str = 'gaussian') -> Dict:
        """
        Train Naive Bayes classifier

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            nb_type: Type of Naive Bayes ('gaussian', 'bernoulli', 'multinomial')

        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training {nb_type} Naive Bayes")

        # Select Naive Bayes variant
        if nb_type == 'gaussian':
            model = GaussianNB()
        elif nb_type == 'bernoulli':
            # For binary features
            model = BernoulliNB(alpha=1.0)
        elif nb_type == 'multinomial':
            # For count features (ensure non-negative)
            X_train = X_train.clip(lower=0)
            if X_val is not None:
                X_val = X_val.clip(lower=0)
            model = MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Unknown Naive Bayes type: {nb_type}")

        # Train model
        model.fit(X_train, y_train)

        # Calibrate probabilities for better performance
        calibrated_model = CalibratedClassifierCV(
            model,
            method='sigmoid',
            cv=3
        )
        calibrated_model.fit(X_train, y_train)

        # Store model
        self.models[f'naive_bayes_{nb_type}'] = calibrated_model

        # Evaluate
        train_metrics = self._evaluate_model(calibrated_model, X_train, y_train, 'train')

        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate_model(calibrated_model, X_val, y_val, 'validation')

        # Cross-validation scores
        cv_scores = cross_val_score(
            calibrated_model, X_train, y_train,
            cv=5, scoring='roc_auc',
            n_jobs=-1
        )

        self.cv_scores[f'naive_bayes_{nb_type}'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }

        # Log to MLflow
        with mlflow.start_run(run_name=f"NaiveBayes_{nb_type}"):
            mlflow.log_param("nb_type", nb_type)
            mlflow.log_metrics(train_metrics)
            if val_metrics:
                mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.sklearn.log_model(calibrated_model, "model")

        return {
            'model': calibrated_model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'cv_scores': self.cv_scores[f'naive_bayes_{nb_type}']
        }

    def train_svm(self,
                  X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_val: Optional[pd.DataFrame] = None,
                  y_val: Optional[pd.Series] = None,
                  kernel: str = 'rbf',
                  tune_hyperparameters: bool = True) -> Dict:
        """
        Train Support Vector Machine

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            kernel: SVM kernel type
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary with model and metrics
        """
        logger.info(f"Training SVM with {kernel} kernel")

        if tune_hyperparameters:
            # Hyperparameter grid (reduced for speed)
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': [kernel],
                'class_weight': [None, 'balanced']
            }

            base_model = SVC(probability=True, random_state=self.random_state)

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=3,  # Reduced for speed
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            self.best_params[f'svm_{kernel}'] = grid_search.best_params_

        else:
            model = SVC(
                kernel=kernel,
                probability=True,
                class_weight='balanced',
                random_state=self.random_state
            )
            model.fit(X_train, y_train)

        # Store model
        self.models[f'svm_{kernel}'] = model

        # Evaluate
        train_metrics = self._evaluate_model(model, X_train, y_train, 'train')

        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate_model(model, X_val, y_val, 'validation')

        return {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

    def train_knn(self,
                  X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_val: Optional[pd.DataFrame] = None,
                  y_val: Optional[pd.Series] = None,
                  tune_hyperparameters: bool = True) -> Dict:
        """
        Train K-Nearest Neighbors

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training K-Nearest Neighbors")

        if tune_hyperparameters:
            # Hyperparameter grid
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski'],
                'p': [1, 2]
            }

            base_model = KNeighborsClassifier()

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )

            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            self.best_params['knn'] = grid_search.best_params_

        else:
            model = KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            )
            model.fit(X_train, y_train)

        # Store model
        self.models['knn'] = model

        # Evaluate
        train_metrics = self._evaluate_model(model, X_train, y_train, 'train')

        val_metrics = {}
        if X_val is not None and y_val is not None:
            val_metrics = self._evaluate_model(model, X_val, y_val, 'validation')

        return {
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

    def _evaluate_model(self,
                        model,
                        X: pd.DataFrame,
                        y: pd.Series,
                        dataset_name: str = '') -> Dict:
        """
        Evaluate model performance

        Args:
            model: Trained model
            X: Features
            y: True labels
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary with metrics
        """
        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_pred_proba)
        }

        # Log metrics
        for metric_name, value in metrics.items():
            logger.info(f"{dataset_name} {metric_name}: {value:.4f}")

        return metrics

    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare all trained models

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model_name, model in self.models.items():
            metrics = self._evaluate_model(model, X_test, y_test, model_name)

            result = {'model': model_name}
            result.update(metrics)

            # Add CV scores if available
            if model_name in self.cv_scores:
                result['cv_auc_mean'] = self.cv_scores[model_name]['mean']
                result['cv_auc_std'] = self.cv_scores[model_name]['std']

            results.append(result)

        comparison_df = pd.DataFrame(results).sort_values('auc_roc', ascending=False)

        logger.info("\nModel Comparison:")
        logger.info(comparison_df.to_string())

        return comparison_df

    def save_models(self, path: str):
        """Save all trained models"""
        for model_name, model in self.models.items():
            filepath = f"{path}/{model_name}.joblib"
            joblib.dump(model, filepath)
            logger.info(f"Saved {model_name} to {filepath}")

    def load_model(self, filepath: str):
        """Load a saved model"""
        model = joblib.load(filepath)
        model_name = filepath.split('/')[-1].replace('.joblib', '')
        self.models[model_name] = model
        return model