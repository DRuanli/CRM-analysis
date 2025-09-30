"""
Fixed Data Validation Module - Properly handles foreign keys
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger
import warnings
from scipy import stats
from pathlib import Path

from config.settings import get_settings


@dataclass
class ValidationCheck:
    """Individual validation check result"""
    check_name: str
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'
    details: Optional[Dict] = None


class DataValidator:
    """Industrial data validation module with improved foreign key handling"""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.validation_results = []
        self.validation_summary = {}

    def validate_data(self, df: pd.DataFrame, dataset_name: str = "dataset") -> bool:
        """
        Run comprehensive data validation

        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset for logging

        Returns:
            True if all critical validations pass
        """
        logger.info(f"Starting validation for {dataset_name}")
        self.validation_results = []

        # Run validation checks
        self._validate_basic_properties(df, dataset_name)
        self._validate_data_types(df)
        self._validate_missing_values(df)
        self._validate_duplicates(df, dataset_name)  # Pass dataset_name
        self._validate_business_rules(df)
        self._validate_statistical_properties(df)
        self._validate_data_quality(df)
        self._validate_relationships(df)

        # Generate summary
        self._generate_validation_summary(dataset_name)

        # Count failures
        errors = [r for r in self.validation_results if r.severity == 'error' and not r.passed]
        warnings = [r for r in self.validation_results if r.severity == 'warning' and not r.passed]

        # Log summary
        logger.info(f"Validation complete for {dataset_name}: {len(errors)} errors, {len(warnings)} warnings")

        for error in errors:
            logger.error(f"❌ {error.check_name}: {error.message}")

        for warning in warnings[:5]:  # Limit warnings in log
            logger.warning(f"⚠️  {warning.check_name}: {warning.message}")

        if len(warnings) > 5:
            logger.warning(f"... and {len(warnings) - 5} more warnings")

        # Return True if no errors (warnings are acceptable)
        return len(errors) == 0

    def _validate_basic_properties(self, df: pd.DataFrame, dataset_name: str):
        """Validate basic DataFrame properties"""

        # Check if DataFrame is empty
        if len(df) == 0:
            # Empty marketing data is acceptable (warning), empty customer data is error
            if 'marketing' in dataset_name.lower():
                severity = 'warning'
            elif 'customer' in dataset_name.lower():
                severity = 'error'
            else:
                severity = 'warning'  # Be lenient with other datasets

            self.validation_results.append(ValidationCheck(
                check_name="empty_dataframe",
                passed=False,
                message=f"DataFrame {dataset_name} is empty",
                severity=severity
            ))
        else:
            self.validation_results.append(ValidationCheck(
                check_name="dataframe_size",
                passed=True,
                message=f"DataFrame has {len(df):,} rows and {len(df.columns)} columns",
                severity='info',
                details={'rows': len(df), 'columns': len(df.columns)}
            ))

        # Check memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > self.settings.data.memory_limit_gb * 1024:
            self.validation_results.append(ValidationCheck(
                check_name="memory_usage",
                passed=False,
                message=f"DataFrame uses {memory_mb:.2f}MB, exceeds limit",
                severity='warning'
            ))

    def _validate_data_types(self, df: pd.DataFrame):
        """Validate data types"""

        # Check for mixed types in columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column has mixed types
                try:
                    types = df[col].apply(type).value_counts()
                    if len(types) > 1:
                        self.validation_results.append(ValidationCheck(
                            check_name=f"mixed_types_{col}",
                            passed=False,
                            message=f"Column '{col}' has mixed data types: {dict(types)}",
                            severity='warning'
                        ))
                except:
                    pass

        # Check for expected ID columns
        id_columns = [col for col in df.columns if 'id' in col.lower()]
        for col in id_columns:
            if df[col].dtype not in ['object', 'str', 'string']:
                self.validation_results.append(ValidationCheck(
                    check_name=f"id_type_{col}",
                    passed=False,
                    message=f"ID column '{col}' should be string type, found {df[col].dtype}",
                    severity='warning'
                ))

        # Check date columns
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        for col in date_columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                self.validation_results.append(ValidationCheck(
                    check_name=f"date_type_{col}",
                    passed=False,
                    message=f"Date column '{col}' is not datetime type",
                    severity='warning'
                ))

    def _validate_missing_values(self, df: pd.DataFrame):
        """Validate missing values"""

        missing_threshold = self.settings.data.max_missing_percentage
        total_missing = df.isnull().sum().sum()
        total_cells = len(df) * len(df.columns)
        overall_missing_pct = total_missing / total_cells if total_cells > 0 else 0

        # Overall missing check
        if overall_missing_pct > missing_threshold:
            self.validation_results.append(ValidationCheck(
                check_name="overall_missing",
                passed=False,
                message=f"Overall missing rate {overall_missing_pct:.1%} exceeds threshold {missing_threshold:.0%}",
                severity='error'
            ))

        # Column-wise missing check
        high_missing_cols = []
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) if len(df) > 0 else 0

            if missing_pct > missing_threshold:
                high_missing_cols.append((col, missing_pct))
                severity = 'error' if missing_pct > 0.5 else 'warning'

                self.validation_results.append(ValidationCheck(
                    check_name=f"missing_values_{col}",
                    passed=False,
                    message=f"Column '{col}' has {missing_pct:.1%} missing values",
                    severity=severity,
                    details={'missing_count': df[col].isnull().sum(), 'missing_pct': missing_pct}
                ))

        if not high_missing_cols:
            self.validation_results.append(ValidationCheck(
                check_name="missing_values",
                passed=True,
                message=f"All columns have acceptable missing rates (<{missing_threshold:.0%})",
                severity='info'
            ))

    def _validate_duplicates(self, df: pd.DataFrame, dataset_name: str = "dataset"):
        """Validate duplicates with proper foreign key handling"""

        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            self.validation_results.append(ValidationCheck(
                check_name="duplicate_rows",
                passed=False,
                message=f"Found {duplicate_rows:,} duplicate rows ({duplicate_rows / len(df):.1%})",
                severity='warning',
                details={'duplicate_count': duplicate_rows}
            ))
        else:
            self.validation_results.append(ValidationCheck(
                check_name="duplicate_rows",
                passed=True,
                message="No duplicate rows found",
                severity='info'
            ))

        # Determine which ID columns should be unique based on dataset type
        unique_id_columns = []

        # Identify the dataset type from the name
        dataset_type = dataset_name.lower()

        for col in df.columns:
            col_lower = col.lower()

            # Only check for duplicates in actual primary keys, not foreign keys
            if 'customer' in dataset_type and col == 'customer_id':
                # customer_id should be unique only in the customers table
                unique_id_columns.append(col)
            elif 'transaction' in dataset_type and col == 'transaction_id':
                # transaction_id should be unique in transactions table
                unique_id_columns.append(col)
            elif 'interaction' in dataset_type and col == 'interaction_id':
                # interaction_id should be unique in interactions table
                unique_id_columns.append(col)
            elif 'marketing' in dataset_type and col == 'campaign_id':
                # campaign_id might be unique per customer
                pass  # Skip for now, as campaigns might be sent to multiple customers
            elif col_lower.endswith('_id'):
                # For other ID columns, check if it's likely a primary key
                if col_lower.replace('_id', '') in dataset_type:
                    # This is likely the primary key for this table
                    unique_id_columns.append(col)

        # Check for duplicates only in columns that should be unique
        for col in unique_id_columns:
            if col in df.columns:
                duplicates = df[col].duplicated().sum()
                if duplicates > 0:
                    self.validation_results.append(ValidationCheck(
                        check_name=f"duplicate_primary_key_{col}",
                        passed=False,
                        message=f"Found {duplicates:,} duplicate values in primary key column '{col}'",
                        severity='error',
                        details={'duplicate_ids': duplicates}
                    ))
                else:
                    self.validation_results.append(ValidationCheck(
                        check_name=f"unique_primary_key_{col}",
                        passed=True,
                        message=f"Primary key column '{col}' has all unique values",
                        severity='info'
                    ))

        # Check foreign key cardinality (as info, not error)
        foreign_key_candidates = ['customer_id', 'product_id', 'agent_id', 'campaign_id']
        for col in foreign_key_candidates:
            if col in df.columns and col not in unique_id_columns:
                unique_count = df[col].nunique()
                if unique_count < len(df):
                    # This is expected for foreign keys
                    self.validation_results.append(ValidationCheck(
                        check_name=f"foreign_key_cardinality_{col}",
                        passed=True,
                        message=f"Foreign key '{col}' has {unique_count:,} unique values for {len(df):,} records",
                        severity='info',
                        details={'unique_count': unique_count, 'total_records': len(df)}
                    ))

    def _validate_business_rules(self, df: pd.DataFrame):
        """Validate business rules"""

        # Age validation
        if 'age' in df.columns:
            invalid_age = ((df['age'] < 18) | (df['age'] > 100)).sum()
            if invalid_age > 0:
                self.validation_results.append(ValidationCheck(
                    check_name="age_range",
                    passed=False,
                    message=f"Found {invalid_age:,} records with invalid age (outside 18-100)",
                    severity='error',
                    details={'invalid_count': invalid_age}
                ))
            else:
                self.validation_results.append(ValidationCheck(
                    check_name="age_range",
                    passed=True,
                    message="All ages are within valid range (18-100)",
                    severity='info'
                ))

        # Transaction amount validation
        amount_columns = ['total_amount', 'amount', 'price', 'revenue', 'cost']
        for col in amount_columns:
            if col in df.columns:
                negative_amounts = (df[col] < 0).sum()
                if negative_amounts > 0:
                    self.validation_results.append(ValidationCheck(
                        check_name=f"negative_{col}",
                        passed=False,
                        message=f"Found {negative_amounts:,} records with negative {col}",
                        severity='error'
                    ))

                # Check for unrealistic amounts
                if df[col].max() > 1e9:  # 1 billion threshold
                    self.validation_results.append(ValidationCheck(
                        check_name=f"extreme_{col}",
                        passed=False,
                        message=f"Found extreme values in {col}: max={df[col].max():,.2f}",
                        severity='warning'
                    ))

        # Date validation
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        current_date = pd.Timestamp.now()
        for col in date_columns:
            if col in df.columns and pd.api.types.is_datetime64_any_dtype(df[col]):
                # Future dates
                future_dates = (df[col] > current_date).sum()
                if future_dates > 0:
                    self.validation_results.append(ValidationCheck(
                        check_name=f"future_dates_{col}",
                        passed=False,
                        message=f"Found {future_dates:,} future dates in '{col}'",
                        severity='warning' if 'end' not in col.lower() else 'info'
                    ))

                # Very old dates
                min_date = pd.Timestamp('1900-01-01')
                old_dates = (df[col] < min_date).sum()
                if old_dates > 0:
                    self.validation_results.append(ValidationCheck(
                        check_name=f"old_dates_{col}",
                        passed=False,
                        message=f"Found {old_dates:,} dates before 1900 in '{col}'",
                        severity='warning'
                    ))

        # Churned validation
        if 'churned' in df.columns:
            unique_values = df['churned'].dropna().unique()
            if not set(unique_values).issubset({0, 1, 0.0, 1.0}):
                self.validation_results.append(ValidationCheck(
                    check_name="churned_binary",
                    passed=False,
                    message=f"Churned column has non-binary values: {unique_values}",
                    severity='error'
                ))

    def _validate_statistical_properties(self, df: pd.DataFrame):
        """Validate statistical properties"""

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                # Check for extreme outliers using IQR method
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                if IQR > 0:
                    outlier_multiplier = self.settings.data.outlier_std_threshold
                    lower_bound = Q1 - outlier_multiplier * IQR
                    upper_bound = Q3 + outlier_multiplier * IQR

                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    outlier_pct = outliers / len(df)

                    if outlier_pct > 0.05:  # More than 5% outliers
                        self.validation_results.append(ValidationCheck(
                            check_name=f"outliers_{col}",
                            passed=False,
                            message=f"Column '{col}' has {outliers:,} outliers ({outlier_pct:.1%})",
                            severity='warning',
                            details={'outlier_count': outliers, 'outlier_pct': outlier_pct}
                        ))

                # Check for zero variance
                if df[col].std() < self.settings.features.variance_threshold:
                    self.validation_results.append(ValidationCheck(
                        check_name=f"zero_variance_{col}",
                        passed=False,
                        message=f"Column '{col}' has near-zero variance (std={df[col].std():.6f})",
                        severity='warning'
                    ))

                # Check for skewness
                skewness = df[col].skew()
                if abs(skewness) > 3:
                    self.validation_results.append(ValidationCheck(
                        check_name=f"high_skew_{col}",
                        passed=False,
                        message=f"Column '{col}' is highly skewed (skewness={skewness:.2f})",
                        severity='info'
                    ))

    def _validate_data_quality(self, df: pd.DataFrame):
        """Overall data quality checks"""

        # Check minimum sample size
        min_samples = 1000
        if len(df) < min_samples:
            self.validation_results.append(ValidationCheck(
                check_name="sample_size",
                passed=False,
                message=f"Dataset has only {len(df):,} records (minimum recommended: {min_samples:,})",
                severity='warning'
            ))

        # Check for class imbalance in target
        if 'churned' in df.columns:
            churn_rate = df['churned'].mean()
            if churn_rate < 0.05 or churn_rate > 0.95:
                self.validation_results.append(ValidationCheck(
                    check_name="class_imbalance",
                    passed=False,
                    message=f"Severe class imbalance detected (churn rate: {churn_rate:.1%})",
                    severity='warning',
                    details={'churn_rate': churn_rate}
                ))
            elif churn_rate < 0.1 or churn_rate > 0.9:
                self.validation_results.append(ValidationCheck(
                    check_name="class_imbalance",
                    passed=False,
                    message=f"Moderate class imbalance detected (churn rate: {churn_rate:.1%})",
                    severity='info',
                    details={'churn_rate': churn_rate}
                ))

        # Check data completeness
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns))) if (len(df) * len(df.columns)) > 0 else 1
        if completeness < 0.95:
            self.validation_results.append(ValidationCheck(
                check_name="data_completeness",
                passed=False,
                message=f"Data completeness is {completeness:.1%} (recommended: >95%)",
                severity='warning'
            ))
        else:
            self.validation_results.append(ValidationCheck(
                check_name="data_completeness",
                passed=True,
                message=f"Good data completeness ({completeness:.1%})",
                severity='info'
            ))

        # Check for high cardinality features
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            cardinality = df[col].nunique()
            cardinality_ratio = cardinality / len(df) if len(df) > 0 else 0

            if cardinality_ratio > 0.5:
                self.validation_results.append(ValidationCheck(
                    check_name=f"high_cardinality_{col}",
                    passed=False,
                    message=f"Column '{col}' has very high cardinality ({cardinality:,} unique values)",
                    severity='warning'
                ))

    def _validate_relationships(self, df: pd.DataFrame):
        """Validate relationships between columns"""

        # Check date relationships
        date_pairs = [
            ('registration_date', 'first_transaction_date'),
            ('first_transaction_date', 'last_transaction_date'),
            ('created_at', 'updated_at')
        ]

        for date1, date2 in date_pairs:
            if date1 in df.columns and date2 in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[date1]) and pd.api.types.is_datetime64_any_dtype(df[date2]):
                    invalid = (df[date1] > df[date2]).sum()
                    if invalid > 0:
                        self.validation_results.append(ValidationCheck(
                            check_name=f"date_relationship_{date1}_{date2}",
                            passed=False,
                            message=f"{date1} is after {date2} in {invalid:,} records",
                            severity='error'
                        ))

        # Check numerical relationships
        if 'total_transactions' in df.columns and 'total_spent' in df.columns:
            # Transactions without spending
            invalid = ((df['total_transactions'] > 0) & (df['total_spent'] <= 0)).sum()
            if invalid > 0:
                self.validation_results.append(ValidationCheck(
                    check_name="transaction_amount_consistency",
                    passed=False,
                    message=f"Found {invalid:,} records with transactions but no spending",
                    severity='warning'
                ))

    def _generate_validation_summary(self, dataset_name: str):
        """Generate validation summary"""

        total_checks = len(self.validation_results)
        passed_checks = sum(1 for r in self.validation_results if r.passed)
        failed_checks = total_checks - passed_checks

        errors = sum(1 for r in self.validation_results if r.severity == 'error' and not r.passed)
        warnings = sum(1 for r in self.validation_results if r.severity == 'warning' and not r.passed)
        info = sum(1 for r in self.validation_results if r.severity == 'info')

        self.validation_summary = {
            'dataset': dataset_name,
            'total_checks': total_checks,
            'passed': passed_checks,
            'failed': failed_checks,
            'errors': errors,
            'warnings': warnings,
            'info': info,
            'pass_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'is_valid': errors == 0
        }

    def get_validation_report(self) -> Dict:
        """Get detailed validation report"""

        report = {
            'summary': self.validation_summary,
            'checks': []
        }

        for check in self.validation_results:
            report['checks'].append({
                'name': check.check_name,
                'passed': check.passed,
                'severity': check.severity,
                'message': check.message,
                'details': check.details
            })

        return report

    def export_validation_report(self, filepath: Path):
        """Export validation report to file"""

        report = self.get_validation_report()

        import json
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Validation report exported to {filepath}")