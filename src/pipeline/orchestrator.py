import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pickle
import joblib
from datetime import datetime
import traceback
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')

from loguru import logger
from tqdm import tqdm

from src.data.collector import DataCollector
from src.data.cleaner import DataCleaner
from src.data.validator import DataValidator
from src.features.engineer import FeatureEngineer
from src.analysis.eda import ExploratoryDataAnalysis
from src.utils.decorators import timer, memory_monitor
from src.utils.database import DatabaseManager
from config.settings import get_settings


class PipelineStage(Enum):
    """Pipeline stages enumeration"""
    DATA_COLLECTION = "data_collection"
    DATA_CLEANING = "data_cleaning"
    DATA_VALIDATION = "data_validation"
    FEATURE_ENGINEERING = "feature_engineering"
    EXPLORATORY_ANALYSIS = "exploratory_analysis"
    MODEL_PREPARATION = "model_preparation"


@dataclass
class PipelineResult:
    """Pipeline execution result"""
    stage: PipelineStage
    success: bool
    data: Any = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict = field(default_factory=dict)


class PipelineOrchestrator:
    """
    Industrial-grade pipeline orchestrator for Phase 1
    Manages end-to-end data processing workflow
    """

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.results = {}
        self.data_cache = {}
        self.pipeline_state = {}
        self.start_time = None
        self.db_manager = DatabaseManager(self.settings.database)

        # Initialize components
        self.collector = None
        self.cleaner = None
        self.validator = None
        self.engineer = None
        self.analyzer = None

        # Pipeline configuration
        self.pipeline_config = self._load_pipeline_config()

        logger.info("Pipeline Orchestrator initialized")

    def _load_pipeline_config(self) -> Dict:
        """Load pipeline configuration"""
        config_path = self.settings.paths.CONFIG_DIR / 'pipeline_config.yaml'

        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'stages': {
                    'data_collection': {'enabled': True, 'retry_attempts': 3},
                    'data_cleaning': {'enabled': True, 'deep_clean': True},
                    'data_validation': {'enabled': True, 'strict_mode': False},
                    'feature_engineering': {'enabled': True, 'feature_selection': True},
                    'exploratory_analysis': {'enabled': True, 'generate_plots': True}
                },
                'parallel_processing': True,
                'checkpoint_enabled': True,
                'monitoring_enabled': True
            }

    @timer
    @memory_monitor
    def run_pipeline(self,
                     stages: Optional[List[PipelineStage]] = None,
                     resume_from: Optional[PipelineStage] = None,
                     **kwargs) -> Dict[str, PipelineResult]:
        """
        Run the complete data processing pipeline

        Args:
            stages: List of stages to run (None for all stages)
            resume_from: Resume from specific stage
            **kwargs: Additional parameters for stages

        Returns:
            Dictionary of pipeline results
        """
        self.start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("STARTING INDUSTRIAL CRM DATA ANALYSIS PIPELINE - PHASE 1")
        logger.info("=" * 80)
        logger.info(f"Start time: {self.start_time}")
        logger.info(f"Environment: {self.settings.env}")
        logger.info(f"Configuration loaded from: {self.settings.paths.CONFIG_DIR}")

        # Define pipeline stages
        if stages is None:
            stages = [
                PipelineStage.DATA_COLLECTION,
                PipelineStage.DATA_CLEANING,
                PipelineStage.DATA_VALIDATION,
                PipelineStage.FEATURE_ENGINEERING,
                PipelineStage.EXPLORATORY_ANALYSIS,
                PipelineStage.MODEL_PREPARATION
            ]

        # Resume from checkpoint if specified
        if resume_from:
            stage_index = stages.index(resume_from)
            stages = stages[stage_index:]
            logger.info(f"Resuming pipeline from stage: {resume_from.value}")
            self._load_checkpoint(resume_from)

        # Execute pipeline stages
        for stage in stages:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"EXECUTING STAGE: {stage.value.upper()}")
            logger.info("=" * 50)

            try:
                if stage == PipelineStage.DATA_COLLECTION:
                    result = self._run_data_collection(**kwargs)

                elif stage == PipelineStage.DATA_CLEANING:
                    result = self._run_data_cleaning(**kwargs)

                elif stage == PipelineStage.DATA_VALIDATION:
                    result = self._run_data_validation(**kwargs)

                elif stage == PipelineStage.FEATURE_ENGINEERING:
                    result = self._run_feature_engineering(**kwargs)

                elif stage == PipelineStage.EXPLORATORY_ANALYSIS:
                    result = self._run_exploratory_analysis(**kwargs)

                elif stage == PipelineStage.MODEL_PREPARATION:
                    result = self._run_model_preparation(**kwargs)

                self.results[stage] = result

                if result.success:
                    logger.success(f"✓ Stage {stage.value} completed successfully")

                    # Save checkpoint
                    if self.pipeline_config.get('checkpoint_enabled', True):
                        self._save_checkpoint(stage, result)
                else:
                    logger.error(f"✗ Stage {stage.value} failed: {result.error}")

                    # Decide whether to continue
                    if not kwargs.get('continue_on_error', False):
                        break

            except Exception as e:
                logger.error(f"Pipeline stage {stage.value} failed with exception: {str(e)}")
                logger.error(traceback.format_exc())

                self.results[stage] = PipelineResult(
                    stage=stage,
                    success=False,
                    error=str(e)
                )

                if not kwargs.get('continue_on_error', False):
                    break

        # Generate final report
        self._generate_pipeline_report()

        # Clean up resources
        self._cleanup()

        end_time = datetime.now()
        execution_time = (end_time - self.start_time).total_seconds()

        logger.info("=" * 80)
        logger.info("PIPELINE EXECUTION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"End time: {end_time}")
        logger.info(f"Total execution time: {execution_time:.2f} seconds")

        # Summary
        successful_stages = sum(1 for r in self.results.values() if r.success)
        failed_stages = len(self.results) - successful_stages

        logger.info(f"Successful stages: {successful_stages}")
        logger.info(f"Failed stages: {failed_stages}")

        return self.results

    def _run_data_collection(self, **kwargs) -> PipelineResult:
        """Run data collection stage"""
        start_time = datetime.now()

        try:
            self.collector = DataCollector(self.settings)

            # Collect data from all sources
            sources = kwargs.get('data_sources', ['customers', 'transactions', 'interactions', 'marketing'])
            use_cache = kwargs.get('use_cache', True)

            collected_data = self.collector.collect_all_data(
                sources=sources,
                use_cache=use_cache
            )

            # Store in cache
            self.data_cache['raw_data'] = collected_data

            # Generate collection summary
            summary = {
                'total_datasets': len(collected_data),
                'datasets': {
                    name: {
                        'records': len(df),
                        'columns': len(df.columns),
                        'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                    }
                    for name, df in collected_data.items()
                }
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return PipelineResult(
                stage=PipelineStage.DATA_COLLECTION,
                success=True,
                data=collected_data,
                execution_time=execution_time,
                metadata=summary
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.DATA_COLLECTION,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _run_data_cleaning(self, **kwargs) -> PipelineResult:
        """Run data cleaning stage"""
        start_time = datetime.now()

        try:
            self.cleaner = DataCleaner(self.settings)

            # Get raw data
            raw_data = self.data_cache.get('raw_data')
            if raw_data is None:
                raise ValueError("No raw data found. Run data collection first.")

            # Clean data
            deep_clean = kwargs.get('deep_clean', True)
            cleaned_data = self.cleaner.clean_all_data(
                data_dict=raw_data,
                deep_clean=deep_clean
            )

            # Store in cache
            self.data_cache['cleaned_data'] = cleaned_data

            # Get cleaning reports
            metadata = {
                'cleaning_reports': self.cleaner.cleaning_reports
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return PipelineResult(
                stage=PipelineStage.DATA_CLEANING,
                success=True,
                data=cleaned_data,
                execution_time=execution_time,
                metadata=metadata
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.DATA_CLEANING,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _run_data_validation(self, **kwargs) -> PipelineResult:
        """Run data validation stage"""
        start_time = datetime.now()

        try:
            self.validator = DataValidator(self.settings)

            # Get cleaned data
            cleaned_data = self.data_cache.get('cleaned_data')
            if cleaned_data is None:
                raise ValueError("No cleaned data found. Run data cleaning first.")

            # Validate each dataset
            validation_results = {}
            all_valid = True

            for name, df in cleaned_data.items():
                is_valid = self.validator.validate_data(df)
                validation_results[name] = {
                    'valid': is_valid,
                    'checks': self.validator.validation_results
                }

                if not is_valid:
                    all_valid = False
                    logger.warning(f"Dataset {name} failed validation")

            execution_time = (datetime.now() - start_time).total_seconds()

            return PipelineResult(
                stage=PipelineStage.DATA_VALIDATION,
                success=all_valid,
                data=validation_results,
                execution_time=execution_time,
                metadata={'all_datasets_valid': all_valid}
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.DATA_VALIDATION,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _run_feature_engineering(self, **kwargs) -> PipelineResult:
        """Run feature engineering stage"""
        start_time = datetime.now()

        try:
            self.engineer = FeatureEngineer(self.settings)

            # Get cleaned data
            cleaned_data = self.data_cache.get('cleaned_data')
            if cleaned_data is None:
                raise ValueError("No cleaned data found. Run data cleaning first.")

            # Create features
            master_features = self.engineer.create_features(cleaned_data)

            # Store in cache
            self.data_cache['master_features'] = master_features

            # Get feature statistics
            metadata = {
                'total_features': len(master_features.columns),
                'numeric_features': len(master_features.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(master_features.select_dtypes(include=['object']).columns),
                'total_records': len(master_features)
            }

            # Add feature importance if available
            if hasattr(self.engineer, 'feature_importance'):
                metadata['top_features'] = self.engineer.feature_importance.head(10).to_dict('records')

            execution_time = (datetime.now() - start_time).total_seconds()

            return PipelineResult(
                stage=PipelineStage.FEATURE_ENGINEERING,
                success=True,
                data=master_features,
                execution_time=execution_time,
                metadata=metadata
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.FEATURE_ENGINEERING,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _run_exploratory_analysis(self, **kwargs) -> PipelineResult:
        """Run exploratory data analysis stage"""
        start_time = datetime.now()

        try:
            self.analyzer = ExploratoryDataAnalysis(self.settings)

            # Get master features
            master_features = self.data_cache.get('master_features')
            if master_features is None:
                raise ValueError("No feature data found. Run feature engineering first.")

            # Perform EDA
            target_col = kwargs.get('target_column', 'churned')
            generate_plots = kwargs.get('generate_plots', True)

            eda_report = self.analyzer.perform_eda(
                df=master_features,
                target_col=target_col,
                generate_plots=generate_plots
            )

            # Store in cache
            self.data_cache['eda_report'] = eda_report

            execution_time = (datetime.now() - start_time).total_seconds()

            return PipelineResult(
                stage=PipelineStage.EXPLORATORY_ANALYSIS,
                success=True,
                data=eda_report,
                execution_time=execution_time,
                metadata={
                    'visualizations_generated': len(self.analyzer.visualizations),
                    'report_sections': list(eda_report.keys())
                }
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.EXPLORATORY_ANALYSIS,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _run_model_preparation(self, **kwargs) -> PipelineResult:
        """Prepare data for Phase 2 modeling"""
        start_time = datetime.now()

        try:
            # Get master features
            master_features = self.data_cache.get('master_features')
            if master_features is None:
                raise ValueError("No feature data found. Run feature engineering first.")

            # Prepare modeling datasets
            target_col = kwargs.get('target_column', 'churned')

            # Separate features and target
            feature_cols = [col for col in master_features.columns
                            if col not in ['customer_id', target_col]]

            X = master_features[feature_cols]
            y = master_features[target_col] if target_col in master_features.columns else None

            # Split data for modeling
            from sklearn.model_selection import train_test_split

            if y is not None:
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y,
                    test_size=0.3,
                    random_state=self.settings.data.random_state,
                    stratify=y
                )

                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp,
                    test_size=0.5,
                    random_state=self.settings.data.random_state,
                    stratify=y_temp
                )

                # Save datasets
                datasets = {
                    'X_train': X_train,
                    'X_val': X_val,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_val': y_val,
                    'y_test': y_test
                }

                # Save to disk
                for name, data in datasets.items():
                    filepath = self.settings.paths.PROCESSED_DATA_DIR / f"{name}.parquet"
                    if isinstance(data, pd.Series):
                        data.to_frame().to_parquet(filepath)
                    else:
                        data.to_parquet(filepath)
                    logger.info(f"Saved {name} to {filepath}")

                metadata = {
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'test_size': len(X_test),
                    'features': len(feature_cols),
                    'target_distribution': {
                        'train': y_train.value_counts().to_dict(),
                        'val': y_val.value_counts().to_dict(),
                        'test': y_test.value_counts().to_dict()
                    }
                }
            else:
                # No target variable - save features only
                X.to_parquet(self.settings.paths.PROCESSED_DATA_DIR / "features.parquet")

                metadata = {
                    'total_records': len(X),
                    'features': len(feature_cols)
                }

            execution_time = (datetime.now() - start_time).total_seconds()

            return PipelineResult(
                stage=PipelineStage.MODEL_PREPARATION,
                success=True,
                data=datasets if y is not None else X,
                execution_time=execution_time,
                metadata=metadata
            )

        except Exception as e:
            return PipelineResult(
                stage=PipelineStage.MODEL_PREPARATION,
                success=False,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _save_checkpoint(self, stage: PipelineStage, result: PipelineResult):
        """Save pipeline checkpoint"""
        checkpoint_dir = self.settings.paths.BASE_DIR / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_file = checkpoint_dir / f"checkpoint_{stage.value}.pkl"

        checkpoint_data = {
            'stage': stage,
            'result': result,
            'data_cache': self.data_cache,
            'timestamp': datetime.now()
        }

        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)

        logger.debug(f"Saved checkpoint for stage {stage.value}")

    def _load_checkpoint(self, stage: PipelineStage):
        """Load pipeline checkpoint"""
        checkpoint_dir = self.settings.paths.BASE_DIR / 'checkpoints'
        checkpoint_file = checkpoint_dir / f"checkpoint_{stage.value}.pkl"

        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)

            self.data_cache = checkpoint_data['data_cache']
            logger.info(f"Loaded checkpoint from {checkpoint_data['timestamp']}")
        else:
            logger.warning(f"No checkpoint found for stage {stage.value}")

    def _generate_pipeline_report(self):
        """Generate comprehensive pipeline report"""
        report_path = self.settings.paths.REPORTS_DIR / 'pipeline_report.json'

        report = {
            'execution_summary': {
                'start_time': str(self.start_time),
                'end_time': str(datetime.now()),
                'total_execution_time': (datetime.now() - self.start_time).total_seconds(),
                'environment': self.settings.env
            },
            'stage_results': {}
        }

        for stage, result in self.results.items():
            report['stage_results'][stage.value] = {
                'success': result.success,
                'execution_time': result.execution_time,
                'error': result.error,
                'metadata': result.metadata
            }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Pipeline report saved to {report_path}")

        # Generate summary text report
        summary_path = self.settings.paths.REPORTS_DIR / 'pipeline_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PIPELINE EXECUTION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Start Time: {self.start_time}\n")
            f.write(f"End Time: {datetime.now()}\n")
            f.write(f"Total Execution Time: {(datetime.now() - self.start_time).total_seconds():.2f} seconds\n\n")

            f.write("STAGE RESULTS\n")
            f.write("-" * 40 + "\n")

            for stage, result in self.results.items():
                status = "✓ SUCCESS" if result.success else "✗ FAILED"
                f.write(f"{stage.value}: {status} ({result.execution_time:.2f}s)\n")
                if result.error:
                    f.write(f"  Error: {result.error}\n")
                if result.metadata:
                    f.write(f"  Metadata: {json.dumps(result.metadata, indent=4)}\n")

            f.write("\n")

    def _cleanup(self):
        """Clean up resources"""
        try:
            # Close database connections
            if hasattr(self, 'db_manager'):
                self.db_manager.close_connections()

            # Clear large objects from memory
            if self.settings.env == 'production':
                self.data_cache.clear()

            logger.debug("Resources cleaned up")
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")

