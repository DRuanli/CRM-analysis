"""
================================================================================
FILE: tests/conftest.py
Pytest configuration and fixtures for testing
================================================================================
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Settings, ProjectPaths

@pytest.fixture(scope="session")
def test_settings():
    """Create test settings with temporary directories"""
    settings = Settings()
    settings.env = "test"

    # Use temporary directories for testing
    temp_dir = Path(tempfile.mkdtemp(prefix="crm_test_"))

    # Create custom ProjectPaths for testing
    settings.paths = ProjectPaths()
    settings.paths.BASE_DIR = temp_dir
    settings.paths.DATA_DIR = temp_dir / "data"
    settings.paths.RAW_DATA_DIR = temp_dir / "data" / "raw"
    settings.paths.INTERIM_DATA_DIR = temp_dir / "data" / "interim"
    settings.paths.PROCESSED_DATA_DIR = temp_dir / "data" / "processed"
    settings.paths.REPORTS_DIR = temp_dir / "reports"
    settings.paths.FIGURES_DIR = temp_dir / "reports" / "figures"
    settings.paths.LOGS_DIR = temp_dir / "logs"
    settings.paths.CONFIG_DIR = temp_dir / "config"

    # Create all directories
    for attr_name, attr_value in settings.paths.__dict__.items():
        if isinstance(attr_value, Path) and attr_name.endswith('_DIR'):
            attr_value.mkdir(parents=True, exist_ok=True)

    # Reduce data size for testing
    settings.data.chunk_size = 100
    settings.data.max_workers = 2

    yield settings

    # Cleanup after all tests
    if temp_dir.exists():
        shutil.rmtree(temp_dir)

@pytest.fixture
def sample_customer_data():
    """Create sample customer data for testing"""
    np.random.seed(42)
    n_customers = 1000

    data = {
        'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
        'registration_date': pd.date_range(
            start='2022-01-01',
            periods=n_customers,
            freq='H'
        ),
        'age': np.random.normal(35, 10, n_customers).clip(18, 80).astype(int),
        'gender': np.random.choice(['M', 'F', 'Other'], n_customers, p=[0.48, 0.48, 0.04]),
        'state': np.random.choice(['CA', 'TX', 'NY', 'FL'], n_customers),
        'customer_segment': np.random.choice(
            ['Premium', 'Standard', 'Basic', 'Trial'],
            n_customers,
            p=[0.1, 0.35, 0.4, 0.15]
        ),
        'acquisition_channel': np.random.choice(
            ['Organic', 'Paid', 'Social', 'Referral'],
            n_customers
        ),
        'email_verified': np.random.choice([True, False], n_customers, p=[0.85, 0.15]),
        'phone_verified': np.random.choice([True, False], n_customers, p=[0.7, 0.3]),
        'churned': np.random.choice([0, 1], n_customers, p=[0.75, 0.25])
    }

    df = pd.DataFrame(data)

    # Make churn correlated with segment
    df.loc[df['customer_segment'] == 'Premium', 'churned'] = np.random.choice(
        [0, 1],
        len(df[df['customer_segment'] == 'Premium']),
        p=[0.92, 0.08]
    )
    df.loc[df['customer_segment'] == 'Trial', 'churned'] = np.random.choice(
        [0, 1],
        len(df[df['customer_segment'] == 'Trial']),
        p=[0.5, 0.5]
    )

    return df

@pytest.fixture
def sample_transaction_data(sample_customer_data):
    """Create sample transaction data for testing"""
    transactions = []

    for customer_id in sample_customer_data['customer_id'].sample(500):
        n_trans = np.random.poisson(10)
        for i in range(n_trans):
            transactions.append({
                'transaction_id': f'TXN_{customer_id}_{i:03d}',
                'customer_id': customer_id,
                'transaction_date': pd.Timestamp.now() - timedelta(days=np.random.randint(0, 365)),
                'total_amount': np.random.lognormal(4, 1) * 10,
                'discount_amount': np.random.choice([0, 5, 10, 15, 20]),
                'product_category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books']),
                'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal']),
                'channel': np.random.choice(['Web', 'Mobile', 'Store'])
            })

    df = pd.DataFrame(transactions)
    df['total_amount'] = df['total_amount'].round(2)
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])

    return df

@pytest.fixture
def sample_interaction_data(sample_customer_data):
    """Create sample interaction data for testing"""
    interactions = []

    for customer_id in sample_customer_data['customer_id'].sample(600):
        n_interactions = np.random.poisson(5)
        for i in range(n_interactions):
            interactions.append({
                'interaction_id': f'INT_{customer_id}_{i:03d}',
                'customer_id': customer_id,
                'interaction_date': pd.Timestamp.now() - timedelta(days=np.random.randint(0, 365)),
                'interaction_type': np.random.choice(['Support', 'Sales', 'Complaint', 'Inquiry']),
                'channel': np.random.choice(['Email', 'Phone', 'Chat', 'Social']),
                'duration_seconds': np.random.gamma(2, 120),
                'satisfaction_score': np.random.choice([1, 2, 3, 4, 5]),
                'resolution_status': np.random.choice(['Resolved', 'Pending', 'Escalated'])
            })

    df = pd.DataFrame(interactions)
    df['duration_seconds'] = df['duration_seconds'].round(0).astype(int)
    df['interaction_date'] = pd.to_datetime(df['interaction_date'])

    return df

@pytest.fixture
def sample_marketing_data(sample_customer_data):
    """Create sample marketing data for testing"""
    campaigns = []

    for customer_id in sample_customer_data['customer_id'].sample(400):
        n_campaigns = np.random.poisson(3)
        for i in range(n_campaigns):
            campaigns.append({
                'campaign_id': f'CAMP_{customer_id}_{i:03d}',
                'customer_id': customer_id,
                'campaign_date': pd.Timestamp.now() - timedelta(days=np.random.randint(0, 365)),
                'campaign_type': np.random.choice(['Email', 'SMS', 'Push']),
                'campaign_name': np.random.choice(['Summer Sale', 'New Product', 'Win-back']),
                'opened': np.random.choice([0, 1], p=[0.6, 0.4]),
                'clicked': np.random.choice([0, 1], p=[0.8, 0.2]),
                'converted': np.random.choice([0, 1], p=[0.95, 0.05])
            })

    df = pd.DataFrame(campaigns)
    df['campaign_date'] = pd.to_datetime(df['campaign_date'])

    # Ensure logical consistency
    df.loc[df['opened'] == 0, 'clicked'] = 0
    df.loc[df['clicked'] == 0, 'converted'] = 0

    return df

@pytest.fixture
def sample_data_dict(sample_customer_data, sample_transaction_data,
                    sample_interaction_data, sample_marketing_data):
    """Create complete sample data dictionary for testing"""
    return {
        'customers': sample_customer_data,
        'transactions': sample_transaction_data,
        'interactions': sample_interaction_data,
        'marketing': sample_marketing_data
    }

@pytest.fixture
def mock_database_manager(test_settings, monkeypatch):
    """Mock database manager for testing"""
    from unittest.mock import Mock, MagicMock
    from src.utils.database import DatabaseManager

    mock_db = Mock(spec=DatabaseManager)
    mock_db.engine = MagicMock()
    mock_db.execute_query = MagicMock()
    mock_db.bulk_insert = MagicMock()
    mock_db.close_connections = MagicMock()

    return mock_db