import warnings
import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
warnings.filterwarnings('ignore')

from config.settings import get_settings
from src.utils.decorators import timer, memory_monitor

from dataclasses import asdict
import json

@dataclass
class CleaningReport:
    """Data cleaning report"""
    total_records_before: int
    total_records_after: int
    duplicates_removed: int
    missing_values_handled: Dict[str, int]
    outliers_handled: Dict[str, int]
    data_types_fixed: List[str]
    validation_errors: List[str]
    cleaning_time: float

class DataCleaner:
    """Industrial-grade data cleaning module"""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.cleaning_reports = {}
        self.scalers = {}
        self.imputers = {}

    @timer
    @memory_monitor
    def clean_all_data(self,
                       data_dict: Dict[str, pd.DataFrame],
                       deep_clean: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Clean all datasets with industrial-grade processing

        :param data_dict: Dictionary of raw DataFrames
        :param deep_clean: Whether to perform deep cleaning

        :return: Dictionary of cleaned DataFrames
        """
        logger.info("="*50)
        logger.info("Starting Industial Data Cleaning Pipeline")
        logger.info("="*50)

        cleaned_data = {}

        for name, df in data_dict.items():
            logger.info(f"\nCleaning {name} dataset...")
            logger.info(f"Initial shape: {df.shape}")

            # Create cleaning report
            report = CleaningReport(
                total_records_before=len(df),
                total_records_after=0,
                duplicates_removed=0,
                missing_values_handled={},
                outliers_handled={},
                data_types_fixed=[],
                validation_errors=[],
                cleaning_time=0
            )

            # Apply cleaning pipeline
            if name == 'customers':
                cleaned_df = self._clean_customers(df, report, deep_clean)
            elif name == 'transactions':
                cleaned_df = self._clean_transactions(df, report, deep_clean)
            elif name == 'interactions':
                cleaned_df = self._clean_interactions(df, report, deep_clean)
            elif name == 'marketing':
                cleaned_df = self._clean_marketing(df, report, deep_clean)
            else:
                cleaned_df = self._clean_generic(df, report, deep_clean)

            report.total_records_after = len(cleaned_df)
            self.cleaning_reports[name] = report
            cleaned_data[name] = cleaned_df

            logger.info(f"Cleaned shape: {cleaned_df.shape}")
            logger.info(f"Records removed: {report.total_records_before - report.total_records_after}")

        # Save cleaned data
        self._save_cleaned_data(cleaned_data)

        # Generate cleaning report
        self._generate_cleaning_report()

        return cleaned_data

    def _clean_customers(self, df: pd.DataFrame, report: CleaningReport, deep_clean: bool) -> pd.DataFrame:
        """Clean customer data with specific business rules"""

        # 1. Remove duplicates
        df = self._remove_duplicates(df, subset=['customer_id'], report=report)

        # 2. Fix data types
        df = self._fix_customer_dtypes(df, report=report)

        # 3. Handle missing values
        df = self._handle_customer_missing(df, report=report)

        # 4. Validate business rules
        df = self._validate_customer_rules(df, report)

        # 5. Handle outliers if deep cleaning
        if deep_clean:
            df = self._handle_customer_outliers(df, report)

        # 6. Create derived columns
        df = self._create_customer_derived_columns(df)

        # 7. Standardize categorical values
        df = self._standardize_customer_categories(df)

        return df

    def _clean_transactions(self, df: pd.DataFrame, report: CleaningReport, deep_clean: bool) -> pd.DataFrame:
        """Clean transaction data"""

        # 1. Remove duplicates
        df = self._remove_duplicates(df, subset=['transaction_id'], report=report)

        # 2. Fix data types
        dtype_mapping = {
            'customer_id': str,
            'transaction_id': str,
            'total_amount': float,
            'discount_amount': float,
            'transaction_date': 'datetime64[ns]'
        }

        df = self._fix_data_types(df, dtype_mapping, report=report)

        # 3. Remove invalid transactions
        initial_len = len(df)

        # Remove negative amounts
        df = df[df['total_amount'] > 0]

        # Remove transactions outside valid date range
        if 'transaction_date' in df.columns:
            df = df[df['transaction_date'] <= pd.Timestamp.now()]
            df = df[df['transaction_date'] >= pd.Timestamp('2020-01-01')]

        # Remove transactions with invalid status
        if 'transaction_status' in df.columns:
            valid_statuses = ['completed', 'pending', 'processing']
            df = df[df['transaction_status'].isin(valid_statuses)]

        records_removed = initial_len - len(df)
        if records_removed > 0:
            logger.info(f"Removed {records_removed} invalid transactions")

        # 4. Handle missing values
        missing_strategies = {
            'discount_amount': 0,
            'payment_method': 'Unknown',
            'channel': 'Unknown',
            'device_type': 'Unknown'
        }

        for col, strategy in missing_strategies.items():
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    df[col].fillna(strategy, inplace=True)
                    report.missing_values_handled[f'transactions.{col}'] = missing_count

        # 5. Handle outliers
        if deep_clean and 'total_amount' in df.columns:
            df = self._handle_outliers_iqr(df, 'total_amount', 3.0, report=report)

        # 6. Create transaction features
        if 'transaction_date' in df.columns:
            df['transaction_year'] = df['transaction_date'].dt.year
            df['transaction_month'] = df['transaction_date'].dt.month
            df['transaction_day'] = df['transaction_date'].dt.day
            df['transaction_hour'] = df['transaction_date'].dt.hour
            df['transaction_weekday'] = df['transaction_date'].dt.dayofweek
            df['is_weekend'] = df['transaction_weekday'].isin([5, 6]).astype(int)

        return df

    def _clean_interactions(self, df: pd.DataFrame, report: CleaningReport, deep_clean: bool) -> pd.DataFrame:
        """Clean interaction data"""

        # Remove duplicates
        df = self._remove_duplicates(df, subset=['interaction_id'], report=report)

        # Fix data types
        dtype_mapping = {
            'customer_id': str,
            'interaction_id': str,
            'duration_seconds': float,
            'satisfaction_score': float,
            'interaction_date': 'datetime64[ns]'
        }
        df = self._fix_data_types(df, dtype_mapping, report)

        # Handle missing satisfaction scores
        if 'satisfaction_score' in df.columns:
            # Use KNN imputer for satisfaction scores
            missing_count = df['satisfaction_score'].isnull().sum()
            if missing_count > 0:
                # Group by interaction type for better imputation
                if 'interaction_type' in df.columns:
                    for int_type in df['interaction_type'].unique:
                        mask = df['interaction_type'] == int_type
                        median_score = df.loc[mask, 'satisfaction_score'].median()
                        df.loc[mask, 'satisfaction_score'] = df.loc[mask, 'satisfaction_score'].fillna(median_score)
                else:
                    df['satisfaction_score'].fillna(df['satisfaction_score'].median(), inplace=True)

                report.missing_values_handled['interactions.satisfaction_score'] = missing_count

        # Validate satisfaction scores
        if 'satisfaction_score' in df.columns:
            df['satisfaction_score'] = df['satisfaction_score'].clip(1, 5)

        # Handle outliers in duration
        if deep_clean and 'duration_seconds' in df.columns:
            # Cap duration at 2 hours
            max_duration = 7200  # 2 hours in seconds
            outlier_count = (df['duration_seconds'] > max_duration).sum()
            if outlier_count > 0:
                df['duration_seconds'] = df['duration_seconds'].clip(upper=max_duration)
                report.outliers_handled['interactions.duration_seconds'] = outlier_count

        # Create derived features
        if 'interaction_date' in df.columns:
            df['interaction_hour'] = df['interaction_date'].dt.hour
            df['is_business_hours'] = df['interaction_hour'].between(9, 17).astype(int)

        return df

    def _clean_marketing(self, df: pd.DataFrame, report: CleaningReport, deep_clean: bool) -> pd.DataFrame:
        """Clean marketing data"""

        # Remove duplicates
        df = self._remove_duplicates(df, subset=['campaign_id', 'customer_id'], report=report)

        # Fix data types
        binary_cols = ['opened', 'clicked', 'converted']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Ensure logical consistency
        # If not opened, cannot be clicked or converted
        if all(col in df.columns for col in ['opened', 'clicked', 'converted']):
            df.loc[df['opened'] == 0, 'clicked'] = 0
            df.loc[df['opened'] == 0, 'converted'] = 0
            df.loc[df['clicked'] == 0, 'converted'] = 0

        return df

    def _clean_generic(self, df: pd.DataFrame, report: CleaningReport, deep_clean: bool) -> pd.DataFrame:
        """Generic cleaning for any dataset"""

        # Remove complete duplicates
        initial_duplicates = df.duplicated().sum()
        df = df.drop_duplicates()
        report.duplicates_removed = initial_duplicates

        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Numeric: fill with median
        for col in numeric_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                df[col].fillna(df[col].median(), inplace=True)
                report.missing_values_handled[col] = missing_count

        # Categorical: fill with mode or 'Unknown'
        for col in categorical_cols:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if len(df[col].mode()) > 0:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna('Unknown', inplace=True)
                report.missing_values_handled[col] = missing_count

        return df

    def _remove_duplicates(self, df: pd.DataFrame, subset: List[str], report: CleaningReport) -> pd.DataFrame:
        """Remove duplicate records"""
        initial_len = len(df)
        df = df.drop_duplicates(subset=subset, keep='first')
        duplicates_removed = initial_len - len(df)

        if duplicates_removed > 0:
            report.duplicates_removed = duplicates_removed
            logger.info(f"Removed {duplicates_removed} duplicate record(s)")

        return df

    def _fix_customer_dtypes(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Fix customer data types"""
        dtype_mapping = {
            'customer_id': str,
            'age': float,
            'registration_date': 'datetime64[ns]',
            'churned': int
        }

        return self._fix_data_types(df, dtype_mapping, report=report)

    def _handle_customer_missing(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Handle missing values in customer data"""

        # Critical fields - drop if missing
        critical_fileds = ["customer_id", "registration_date"]
        for field in critical_fileds:
            if field in df.columns:
                missing = df[field].isnull().sum()
                if missing > 0:
                    df = df.dropna(subset=[field])
                    report.missing_values_handled[f'customers.{field}'] = missing

        # Age - fill with median by segment
        if 'age' in df.columns:
            missing = df['age'].isnull().sum()
            if missing > 0:
                if 'customer_segment' in df.columns:
                    df['age'] = df.groupby('customer_segment')['age'].transform(
                        lambda x: x.fillna(x.median())
                    )
                else:
                    df['age'].fillna(df['age'].median(), inplace=True)
                report.missing_values_handled['customers.age'] = missing

        # Categorical fields - fill with mode or 'Unknown'
        categorical_fields = ['gender', 'state', 'acquisition_channel', 'customer_segment']
        for field in categorical_fields:
            if field in df.columns:
                missing = df[field].isnull().sum()
                if missing > 0:
                    df[field].fillna('Unknown', inplace=True)
                    report.missing_values_handled[f'customers.{field}'] = missing

        # Boolean fields - fill with False
        boolean_fields = ['email_verified', 'phone_verified']
        for field in boolean_fields:
            if field in df.columns:
                missing = df[field].isnull().sum()
                if missing > 0:
                    df[field].fillna(False, inplace=True)
                    report.missing_values_handled[f'customers.{field}'] = missing

        return df

    def _validate_customer_rules(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Validate customer business rules"""
        initial_len = len(df)

        # Age must be between 18 and 100
        if 'age' in df.columns:
            invalid_age = (df['age'] < 18) | (df['age'] > 100)
            if invalid_age.sum() > 0:
                df = df[~invalid_age]
                report.validation_errors.append(f"Removed {invalid_age.sum()} records with invalid age")

        # Registration date cannot be in the future
        if 'registration_date' in df.columns:
            future_dates = df['registration_date'] > pd.Timestamp.now()
            if future_dates.sum() > 0:
                df = df[~future_dates]
                report.validation_errors.append(f"Removed {future_dates.sum()} records with future registration dates")

        # Churned must be binary
        if 'churned' in df.columns:
            df['churned'] = df['churned'].fillna(0).astype(int).clip(0, 1)

        final_len = len(df)
        if initial_len > final_len:
            logger.info(f"Validation removed {initial_len - final_len} invalid records")

        return df

    def _handle_customer_outliers(self, df: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
        """Handle outliers in customer data"""

        # Age outliers
        if 'age' in df.columns:
            df = self._handle_outliers_iqr(df, 'age', factor=2.0, report=report)

        return df

    def _create_customer_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived columns for customers"""

        # Account age
        if 'registration_date' in df.columns:
            df['account_age_days'] = (pd.Timestamp.now() - df['registration_date']).dt.days
            df['account_age_months'] = df['account_age_days'] / 30.44

        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            )

        # Verification status
        if 'email_verified' in df.columns and 'phone_verified' in df.columns:
            df['fully_verified'] = (df['email_verified'] & df['phone_verified']).astype(int)
            df['verification_level'] = df['email_verified'].astype(int) + df['phone_verified'].astype(int)

        return df

    def _standardize_customer_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize categorical values"""

        # Standardize gender
        if 'gender' in df.columns:
            gender_mapping = {
                'm': 'M', 'male': 'M', 'Male': 'M', 'M': 'M',
                'f': 'F', 'female': 'F', 'Female': 'F', 'F': 'F',
                'other': 'Other', 'Other': 'Other', 'O': 'Other'
            }
            df['gender'] = df['gender'].map(lambda x: gender_mapping.get(str(x).lower(), 'Unknown'))

        # Standardize customer segment
        if 'customer_segment' in df.columns:
            df['customer_segment'] = df['customer_segment'].str.title()

        # Standardize states
        if 'state' in df.columns:
            df['state'] = df['state'].str.upper()

        return df

    def _fix_data_types(self, df: pd.DataFrame, dtype_mappings: Dict, report: CleaningReport ) -> pd.DataFrame:
        """Fix data types based on mapping"""
        for col, dtype in dtype_mappings.items():
            if col in df.columns:
                try:
                    if dtype == 'datetime64[ns]':
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    else:
                        df[col] = df[col].astype(dtype)
                    report.data_types_fixed.append(f'{col}-->{dtype}')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to {dtype}: {str(e)}")
        return df

    def _handle_outliers_iqr(self, df: pd.DataFrame, column: str, factor: float, report: CleaningReport) -> pd.DataFrame:
        if column not in df.columns:
            return df

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3-Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
        outlier_count = outliers.sum()

        if outlier_count > 0:
            # Cap outliers instead of removing
            df[column] = df[column].clip(lower_bound, upper_bound)
            report.outliers_handled[column] = outlier_count
            logger.info(f"Capped {outlier_count} outliers in {column}")

        return df

    def _save_cleaned_data(self, cleaned_data: Dict[str, pd.DataFrame]):
        """Save cleaned data to interim directory"""
        for name, df in cleaned_data.items():
            output_path = self.settings.paths.INTERIM_DATA_DIR / f"{name}_cleaned.parquet"
            df.to_parquet(output_path, index=False, compression='snappy')
            logger.info(f"Saved cleaned {name} to {output_path}")

    def _generate_cleaning_report(self):
        """Generate comprehensive cleaning report"""
        report_path = self.settings.paths.REPORTS_DIR / "cleaning_report.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DATA CLEANING REPORT\n")
            f.write("=" * 80 + "\n\n")

            for dataset_name, report in self.cleaning_reports.items():
                f.write(f"\n{dataset_name.upper()} DATASET\n")
                f.write("-" * 40 + "\n")
                f.write(f"Records before: {report.total_records_before:,}\n")
                f.write(f"Records after: {report.total_records_after:,}\n")
                f.write(f"Duplicates removed: {report.duplicates_removed:,}\n")

                if report.missing_values_handled:
                    f.write("\nMissing values handled:\n")
                    for col, count in report.missing_values_handled.items():
                        f.write(f"  - {col}: {count:,}\n")

                if report.outliers_handled:
                    f.write("\nOutliers handled:\n")
                    for col, count in report.outliers_handled.items():
                        f.write(f"  - {col}: {count:,}\n")

                if report.validation_errors:
                    f.write("\nValidation issues:\n")
                    for error in report.validation_errors:
                        f.write(f"  - {error}\n")

                f.write("\n")

        logger.info(f"Cleaning report saved to {report_path}")

        # ----- JSON report -----
        report_path_json = self.settings.paths.REPORTS_DIR / "cleaning_report.json"
        json_data = {name: asdict(report) for name, report in self.cleaning_reports.items()}

        with open(report_path_json, 'w') as f:
            json.dump(asdict(report), f, indent=4, default=str)

        logger.info(f"Cleaning JSON report saved to {report_path_json}")