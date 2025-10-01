"""
Advanced Ensemble Models
Random Forest, XGBoost, LightGBM, CatBoost with Bayesian Optimization
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier
)
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score,
    cross_validate
)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, make_scorer
)
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import joblib
from loguru import logger
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class EnsembleModels:
    """
    Advanced ensemble models with Bayesian optimization
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.optimization_history = {}
        self.feature_importance = {}

    def train_random_forest(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            X_val: Optional[pd.DataFrame] = None,
                            y_val: Optional[pd.Series] = None,
                            optimize: bool = True,
                            n_trials: int = 50) -> Dict:
        """
        Train Random Forest with Optuna optimization

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            optimize: Whether to optimize hyperparameters
            n_trials: Number of Optuna trials

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training Random Forest")

        if optimize:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced', 'balanced_subsample']),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }

                # Create and evaluate model
                model = RandomForestClassifier(**params)

                # Use cross-validation for evaluation
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=5, scoring='roc_auc', n_jobs=-1
                )

                return cv_scores.mean()

            # Run optimization
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state)
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

            # Best parameters
            self.best_params['random_forest'] = study.best_params
            self.optimization_history['random_forest'] = study

            logger.info(f"Best params: {study.best_params}")
            logger.info(f"Best CV score: {study.best_value:.4f}")

            # Train final model with best params
            model = RandomForestClassifier(
                **study.best_params,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            # Default parameters
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )

        # Train model
        model.fit(X_train, y_train)

        # Store model
        self.models['random_forest'] = model

        # Calculate feature importance
        self.feature_importance['random_forest'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Evaluate
        results = self._evaluate_model(model, X_train, y_train, X_val, y_val)

        # Log to MLflow
        with mlflow.start_run(run_name="RandomForest"):
            if optimize:
                mlflow.log_params(study.best_params)
            mlflow.log_metrics(results['train_metrics'])
            if results['val_metrics']:
                mlflow.log_metrics({f"val_{k}": v for k, v in results['val_metrics'].items()})
            mlflow.sklearn.log_model(model, "model")

        return {
            'model': model,
            **results,
            'feature_importance': self.feature_importance['random_forest']
        }

    def train_xgboost(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_val: Optional[pd.DataFrame] = None,
                      y_val: Optional[pd.Series] = None,
                      optimize: bool = True,
                      n_trials: int = 50,
                      early_stopping: bool = True) -> Dict:
        """
        Train XGBoost with Bayesian optimization

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            optimize: Whether to optimize hyperparameters
            n_trials: Number of Optuna trials
            early_stopping: Use early stopping

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training XGBoost")

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        if optimize:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('gamma', 0, 1),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'scale_pos_weight': scale_pos_weight,
                    'objective': 'binary:logistic',
                    'eval_metric': 'auc',
                    'use_label_encoder': False,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }

                # Create model
                model = xgb.XGBClassifier(**params)

                # Use validation set for early stopping if available
                if X_val is not None and y_val is not None and early_stopping:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                    score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                else:
                    # Use cross-validation
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=5, scoring='roc_auc', n_jobs=-1
                    )
                    score = cv_scores.mean()

                # Prune unpromising trials
                trial.report(score, 0)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                return score

            # Run optimization with pruning
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner()
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

            # Best parameters
            self.best_params['xgboost'] = study.best_params
            self.optimization_history['xgboost'] = study

            logger.info(f"Best params: {study.best_params}")
            logger.info(f"Best score: {study.best_value:.4f}")

            # Train final model with best params
            model = xgb.XGBClassifier(
                **study.best_params,
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            # Default parameters
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                objective='binary:logistic',
                eval_metric='auc',
                use_label_encoder=False,
                random_state=self.random_state,
                n_jobs=-1
            )

        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None and early_stopping:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)

        # Store model
        self.models['xgboost'] = model

        # Feature importance
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Evaluate
        results = self._evaluate_model(model, X_train, y_train, X_val, y_val)

        # Log to MLflow
        with mlflow.start_run(run_name="XGBoost"):
            if optimize:
                mlflow.log_params(study.best_params)
            mlflow.log_metrics(results['train_metrics'])
            if results['val_metrics']:
                mlflow.log_metrics({f"val_{k}": v for k, v in results['val_metrics'].items()})
            mlflow.xgboost.log_model(model, "model")

        return {
            'model': model,
            **results,
            'feature_importance': self.feature_importance['xgboost']
        }

    def train_lightgbm(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.Series] = None,
                       optimize: bool = True,
                       n_trials: int = 50) -> Dict:
        """
        Train LightGBM with Optuna optimization

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            optimize: Whether to optimize hyperparameters
            n_trials: Number of Optuna trials

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training LightGBM")

        # Calculate scale_pos_weight
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        if optimize:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'scale_pos_weight': scale_pos_weight,
                    'objective': 'binary',
                    'metric': 'auc',
                    'boosting_type': 'gbdt',
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'verbosity': -1
                }

                # Create model
                model = lgb.LGBMClassifier(**params)

                # Train with validation
                if X_val is not None and y_val is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        eval_metric='auc',
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                    )
                    score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                else:
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=5, scoring='roc_auc', n_jobs=-1
                    )
                    score = cv_scores.mean()

                return score

            # Run optimization
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state)
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

            # Best parameters
            self.best_params['lightgbm'] = study.best_params
            self.optimization_history['lightgbm'] = study

            logger.info(f"Best params: {study.best_params}")
            logger.info(f"Best score: {study.best_value:.4f}")

            # Train final model
            model = lgb.LGBMClassifier(
                **study.best_params,
                scale_pos_weight=scale_pos_weight,
                objective='binary',
                metric='auc',
                boosting_type='gbdt',
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )
        else:
            # Default parameters
            model = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                max_depth=7,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                scale_pos_weight=scale_pos_weight,
                objective='binary',
                metric='auc',
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1
            )

        # Train model
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
        else:
            model.fit(X_train, y_train)

        # Store model
        self.models['lightgbm'] = model

        # Feature importance
        self.feature_importance['lightgbm'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Evaluate
        results = self._evaluate_model(model, X_train, y_train, X_val, y_val)

        # Log to MLflow
        with mlflow.start_run(run_name="LightGBM"):
            if optimize:
                mlflow.log_params(study.best_params)
            mlflow.log_metrics(results['train_metrics'])
            if results['val_metrics']:
                mlflow.log_metrics({f"val_{k}": v for k, v in results['val_metrics'].items()})
            mlflow.lightgbm.log_model(model, "model")

        return {
            'model': model,
            **results,
            'feature_importance': self.feature_importance['lightgbm']
        }

    def train_catboost(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: Optional[pd.DataFrame] = None,
                       y_val: Optional[pd.Series] = None,
                       cat_features: Optional[List[str]] = None,
                       optimize: bool = True,
                       n_trials: int = 30) -> Dict:
        """
        Train CatBoost with optimization

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            cat_features: List of categorical feature names
            optimize: Whether to optimize hyperparameters
            n_trials: Number of Optuna trials

        Returns:
            Dictionary with model and metrics
        """
        logger.info("Training CatBoost")

        # Identify categorical features if not provided
        if cat_features is None:
            cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        # Get categorical feature indices
        cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_features] if cat_features else None

        if optimize:
            def objective(trial):
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 4, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'border_count': trial.suggest_int('border_count', 32, 255),
                    'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                    'random_strength': trial.suggest_float('random_strength', 0, 1),
                    'auto_class_weights': 'Balanced',
                    'eval_metric': 'AUC',
                    'random_state': self.random_state,
                    'verbose': False
                }

                # Create model
                model = cb.CatBoostClassifier(**params)

                # Train with validation
                if X_val is not None and y_val is not None:
                    model.fit(
                        X_train, y_train,
                        eval_set=(X_val, y_val),
                        cat_features=cat_feature_indices,
                        early_stopping_rounds=50,
                        verbose=False
                    )
                    score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
                else:
                    # Use cross-validation
                    cv_scores = cross_val_score(
                        model, X_train, y_train,
                        cv=3, scoring='roc_auc'
                    )
                    score = cv_scores.mean()

                return score

            # Run optimization
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.random_state)
            )
            study.optimize(objective, n_trials=n_trials, n_jobs=1)

            # Best parameters
            self.best_params['catboost'] = study.best_params
            self.optimization_history['catboost'] = study

            logger.info(f"Best params: {study.best_params}")
            logger.info(f"Best score: {study.best_value:.4f}")

            # Train final model
            model = cb.CatBoostClassifier(
                **study.best_params,
                auto_class_weights='Balanced',
                eval_metric='AUC',
                random_state=self.random_state,
                verbose=False
            )
        else:
            # Default parameters
            model = cb.CatBoostClassifier(
                iterations=200,
                depth=6,
                learning_rate=0.1,
                auto_class_weights='Balanced',
                eval_metric='AUC',
                random_state=self.random_state,
                verbose=False
            )

        # Train model
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_feature_indices,
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model.fit(X_train, y_train, cat_features=cat_feature_indices)

        # Store model
        self.models['catboost'] = model

        # Feature importance
        self.feature_importance['catboost'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Evaluate
        results = self._evaluate_model(model, X_train, y_train, X_val, y_val)

        # Log to MLflow
        with mlflow.start_run(run_name="CatBoost"):
            if optimize:
                mlflow.log_params(study.best_params)
            mlflow.log_metrics(results['train_metrics'])
            if results['val_metrics']:
                mlflow.log_metrics({f"val_{k}": v for k, v in results['val_metrics'].items()})
            mlflow.catboost.log_model(model, "model")

        return {
            'model': model,
            **results,
            'feature_importance': self.feature_importance['catboost']
        }

    def _evaluate_model(self,
                        model,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame] = None,
                        y_val: Optional[pd.Series] = None) -> Dict:
        """
        Evaluate model performance

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Dictionary with metrics
        """
        # Training metrics
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]

        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'auc_roc': roc_auc_score(y_train, y_train_proba)
        }

        # Validation metrics
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)[:, 1]

            val_metrics = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred),
                'recall': recall_score(y_val, y_val_pred),
                'f1': f1_score(y_val, y_val_pred),
                'auc_roc': roc_auc_score(y_val, y_val_proba)
            }

        # Log metrics
        logger.info(f"Train metrics: {train_metrics}")
        if val_metrics:
            logger.info(f"Validation metrics: {val_metrics}")

        return {
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }

    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compare all trained ensemble models

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with comparison results
        """
        results = []

        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            result = {
                'model': model_name,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'auc_roc': roc_auc_score(y_test, y_proba)
            }

            results.append(result)

        comparison_df = pd.DataFrame(results).sort_values('auc_roc', ascending=False)

        logger.info("\nEnsemble Model Comparison:")
        logger.info(comparison_df.to_string())

        return comparison_df

    def get_feature_importance_summary(self) -> pd.DataFrame:
        """
        Get aggregated feature importance across all models

        Returns:
            DataFrame with feature importance summary
        """
        if not self.feature_importance:
            return pd.DataFrame()

        # Combine all feature importances
        importance_dfs = []

        for model_name, importance_df in self.feature_importance.items():
            df = importance_df.copy()
            df['model'] = model_name
            importance_dfs.append(df)

        # Combine and pivot
        combined_df = pd.concat(importance_dfs)
        pivot_df = combined_df.pivot(index='feature', columns='model', values='importance').fillna(0)

        # Add average importance
        pivot_df['avg_importance'] = pivot_df.mean(axis=1)

        # Sort by average importance
        pivot_df = pivot_df.sort_values('avg_importance', ascending=False)

        return pivot_df

    def save_models(self, path: str):
        """Save all trained models"""
        import os
        os.makedirs(path, exist_ok=True)

        for model_name, model in self.models.items():
            filepath = f"{path}/{model_name}_ensemble.pkl"
            joblib.dump(model, filepath)
            logger.info(f"Saved {model_name} to {filepath}")