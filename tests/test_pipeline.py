"""
================================================================================
FILE: tests/test_pipeline.py
Tests for pipeline orchestration
================================================================================
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from pathlib import Path
import json

from src.pipeline.orchestrator import PipelineOrchestrator, PipelineStage, PipelineResult
from src.data.collector import DataCollector
from src.data.cleaner import DataCleaner
from src.data.validator import DataValidator
from src.features.engineer import FeatureEngineer
from src.analysis.eda import ExploratoryDataAnalysis

class TestPipelineOrchestrator:
    """Test suite for pipeline orchestrator"""

    def test_orchestrator_initialization(self, test_settings):
        """Test orchestrator initialization"""
        orchestrator = PipelineOrchestrator(test_settings)

        assert orchestrator is not None
        assert orchestrator.settings == test_settings
        assert orchestrator.results == {}
        assert orchestrator.data_cache == {}
        assert orchestrator.pipeline_config is not None

    def test_pipeline_config_loading(self, test_settings):
        """Test pipeline configuration loading"""
        orchestrator = PipelineOrchestrator(test_settings)

        assert 'stages' in orchestrator.pipeline_config
        assert orchestrator.pipeline_config['checkpoint_enabled'] == True
        assert orchestrator.pipeline_config['parallel_processing'] == True

    def test_data_collection_stage(self, test_settings, sample_data_dict):
        """Test data collection stage execution"""
        orchestrator = PipelineOrchestrator(test_settings)

        # Mock the collector
        with patch.object(DataCollector, 'collect_all_data', return_value=sample_data_dict):
            result = orchestrator._run_data_collection()

            assert result.success == True
            assert result.stage == PipelineStage.DATA_COLLECTION
            assert result.data is not None
            assert 'raw_data' in orchestrator.data_cache
            assert len(orchestrator.data_cache['raw_data']) == 4

    def test_data_cleaning_stage(self, test_settings, sample_data_dict):
        """Test data cleaning stage execution"""
        orchestrator = PipelineOrchestrator(test_settings)
        orchestrator.data_cache['raw_data'] = sample_data_dict

        result = orchestrator._run_data_cleaning()

        assert result.success == True
        assert result.stage == PipelineStage.DATA_CLEANING
        assert 'cleaned_data' in orchestrator.data_cache

    def test_data_validation_stage(self, test_settings, sample_data_dict):
        """Test data validation stage execution"""
        orchestrator = PipelineOrchestrator(test_settings)
        orchestrator.data_cache['cleaned_data'] = sample_data_dict

        result = orchestrator._run_data_validation()

        assert result.stage == PipelineStage.DATA_VALIDATION
        assert result.data is not None
        assert isinstance(result.data, dict)

    def test_feature_engineering_stage(self, test_settings, sample_data_dict):
        """Test feature engineering stage execution"""
        orchestrator = PipelineOrchestrator(test_settings)
        orchestrator.data_cache['cleaned_data'] = sample_data_dict

        result = orchestrator._run_feature_engineering()

        assert result.success == True
        assert result.stage == PipelineStage.FEATURE_ENGINEERING
        assert 'master_features' in orchestrator.data_cache
        assert isinstance(result.data, pd.DataFrame)

    def test_full_pipeline_execution(self, test_settings, sample_data_dict):
        """Test full pipeline execution"""
        orchestrator = PipelineOrchestrator(test_settings)

        # Mock all data sources
        with patch.object(DataCollector, 'collect_all_data', return_value=sample_data_dict):
            # Run only collection and cleaning for speed
            results = orchestrator.run_pipeline(
                stages=[PipelineStage.DATA_COLLECTION, PipelineStage.DATA_CLEANING]
            )

            assert len(results) == 2
            assert PipelineStage.DATA_COLLECTION in results
            assert PipelineStage.DATA_CLEANING in results
            assert results[PipelineStage.DATA_COLLECTION].success == True

    def test_pipeline_checkpoint_saving(self, test_settings, sample_data_dict):
        """Test checkpoint saving functionality"""
        orchestrator = PipelineOrchestrator(test_settings)

        # Create a mock result
        result = PipelineResult(
            stage=PipelineStage.DATA_COLLECTION,
            success=True,
            data=sample_data_dict,
            execution_time=10.5
        )

        # Save checkpoint
        orchestrator._save_checkpoint(PipelineStage.DATA_COLLECTION, result)

        # Check checkpoint file exists
        checkpoint_file = test_settings.paths.BASE_DIR / 'checkpoints' / 'checkpoint_data_collection.pkl'
        assert checkpoint_file.exists()

    def test_pipeline_checkpoint_loading(self, test_settings, sample_data_dict):
        """Test checkpoint loading functionality"""
        orchestrator = PipelineOrchestrator(test_settings)

        # First save a checkpoint
        orchestrator.data_cache['test_data'] = sample_data_dict
        result = PipelineResult(
            stage=PipelineStage.DATA_CLEANING,
            success=True,
            data=sample_data_dict
        )
        orchestrator._save_checkpoint(PipelineStage.DATA_CLEANING, result)

        # Clear cache and reload
        orchestrator.data_cache = {}
        orchestrator._load_checkpoint(PipelineStage.DATA_CLEANING)

        # Check data was restored
        assert 'test_data' in orchestrator.data_cache

    def test_pipeline_error_handling(self, test_settings):
        """Test pipeline error handling"""
        orchestrator = PipelineOrchestrator(test_settings)

        # Run validation without data (should fail)
        result = orchestrator._run_data_validation()

        assert result.success == False
        assert result.error is not None
        assert "No cleaned data found" in result.error

    def test_pipeline_continue_on_error(self, test_settings, sample_data_dict):
        """Test continue on error functionality"""
        orchestrator = PipelineOrchestrator(test_settings)

        # Mock collector to fail
        with patch.object(DataCollector, 'collect_all_data', side_effect=Exception("Test error")):
            results = orchestrator.run_pipeline(
                stages=[PipelineStage.DATA_COLLECTION, PipelineStage.DATA_VALIDATION],
                continue_on_error=True
            )

            # Should have attempted both stages
            assert len(results) == 2
            assert results[PipelineStage.DATA_COLLECTION].success == False
            assert results[PipelineStage.DATA_VALIDATION].success == False

    def test_pipeline_report_generation(self, test_settings):
        """Test pipeline report generation"""
        orchestrator = PipelineOrchestrator(test_settings)

        # Add some mock results
        orchestrator.results[PipelineStage.DATA_COLLECTION] = PipelineResult(
            stage=PipelineStage.DATA_COLLECTION,
            success=True,
            execution_time=5.5
        )

        orchestrator._generate_pipeline_report()

        # Check report files exist
        json_report = test_settings.paths.REPORTS_DIR / 'pipeline_report.json'
        txt_report = test_settings.paths.REPORTS_DIR / 'pipeline_summary.txt'

        assert json_report.exists()
        assert txt_report.exists()

        # Verify JSON content
        with open(json_report, 'r') as f:
            report_data = json.load(f)
            assert 'execution_summary' in report_data
            assert 'stage_results' in report_data

    def test_get_pipeline_summary(self, test_settings):
        """Test pipeline summary generation"""
        orchestrator = PipelineOrchestrator(test_settings)

        # Add mock results
        orchestrator.results[PipelineStage.DATA_COLLECTION] = PipelineResult(
            stage=PipelineStage.DATA_COLLECTION,
            success=True,
            execution_time=10.0
        )
        orchestrator.results[PipelineStage.DATA_CLEANING] = PipelineResult(
            stage=PipelineStage.DATA_CLEANING,
            success=False,
            error="Test error",
            execution_time=5.0
        )

        summary = orchestrator.get_pipeline_summary()

        assert summary['total_stages'] == 2
        assert summary['successful_stages'] == 1
        assert summary['failed_stages'] == 1
        assert summary['total_execution_time'] == 15.0

class TestDataValidator:
    """Test suite for data validator"""

    def test_validator_initialization(self, test_settings):
        """Test validator initialization"""
        validator = DataValidator(test_settings)

        assert validator is not None
        assert validator.validation_results == []

    def test_validate_empty_dataframe(self, test_settings):
        """Test validation of empty DataFrame"""
        validator = DataValidator(test_settings)
        empty_df = pd.DataFrame()

        is_valid = validator.validate_data(empty_df)

        assert is_valid == False
        errors = [r for r in validator.validation_results if r.severity == 'error']
        assert len(errors) > 0

    def test_validate_valid_data(self, test_settings, sample_customer_data):
        """Test validation of valid data"""
        validator = DataValidator(test_settings)

        is_valid = validator.validate_data(sample_customer_data)

        assert is_valid == True
        errors = [r for r in validator.validation_results if r.severity == 'error' and not r.passed]
        assert len(errors) == 0

    def test_validate_missing_values(self, test_settings, sample_customer_data):
        """Test missing value validation"""
        validator = DataValidator(test_settings)

        # Add missing values
        df = sample_customer_data.copy()
        df.loc[:100, 'age'] = np.nan

        is_valid = validator.validate_data(df)

        # Check for missing value warnings
        missing_checks = [r for r in validator.validation_results
                         if 'missing' in r.check_name]
        assert len(missing_checks) > 0

    def test_validate_duplicates(self, test_settings, sample_customer_data):
        """Test duplicate validation"""
        validator = DataValidator(test_settings)

        # Add duplicates
        df = sample_customer_data.copy()
        df = pd.concat([df, df.head(50)], ignore_index=True)

        is_valid = validator.validate_data(df)

        # Check for duplicate warnings
        duplicate_checks = [r for r in validator.validation_results
                           if 'duplicate' in r.check_name]
        assert len(duplicate_checks) > 0

    def test_validate_business_rules(self, test_settings, sample_customer_data):
        """Test business rule validation"""
        validator = DataValidator(test_settings)

        # Add invalid ages
        df = sample_customer_data.copy()
        df.loc[:10, 'age'] = 150  # Invalid age

        is_valid = validator.validate_data(df)

        # Check for age validation error
        age_checks = [r for r in validator.validation_results
                     if 'age' in r.check_name]
        assert len(age_checks) > 0

class TestDataCleaner:
    """Test suite for data cleaner"""

    def test_cleaner_initialization(self, test_settings):
        """Test cleaner initialization"""
        cleaner = DataCleaner(test_settings)

        assert cleaner is not None
        assert cleaner.cleaning_reports == {}

    def test_clean_all_data(self, test_settings, sample_data_dict):
        """Test cleaning all datasets"""
        cleaner = DataCleaner(test_settings)

        cleaned_data = cleaner.clean_all_data(sample_data_dict)

        assert len(cleaned_data) == len(sample_data_dict)
        assert 'customers' in cleaned_data
        assert len(cleaner.cleaning_reports) == len(sample_data_dict)

class TestFeatureEngineer:
    """Test suite for feature engineer"""

    def test_engineer_initialization(self, test_settings):
        """Test engineer initialization"""
        engineer = FeatureEngineer(test_settings)

        assert engineer is not None
        assert engineer.encoders == {}
        assert engineer.scalers == {}

    def test_create_features(self, test_settings, sample_data_dict):
        """Test feature creation"""
        engineer = FeatureEngineer(test_settings)

        master_features = engineer.create_features(sample_data_dict)

        assert isinstance(master_features, pd.DataFrame)
        assert len(master_features) == len(sample_data_dict['customers'])
        assert len(master_features.columns) > len(sample_data_dict['customers'].columns)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])