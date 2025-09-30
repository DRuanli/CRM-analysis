from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import requests
from tqdm import tqdm
import concurrent.futures
from src.utils.decorators import timer, memory_monitor, retry
from src.utils.database import DatabaseManager
from config.settings import get_settings, Settings
from loguru import logger


class DataCollector:
    """Industrial-grade data collection module"""

    def __init__(self, settings: Settings = None):
        self.settings = settings or get_settings()
        self.db = DatabaseManager(self.settings.database)
        self.collected_data: Dict[str, pd.DataFrame] = {}

    @timer
    @memory_monitor
    def collect_all_data(self,
                         sources: List[str] = None,
                         date_range: tuple = None,
                         use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Collect data from all sources

        Args:
            sources: List of data sources to collect
            date_range: Tuple of (start_date, end_date)
            use_cache: Whether to use cached data

        Returns:
            Dictionary of DataFrames
        """
        if sources is None:
            sources = ['customers', 'transactions', 'interactions', 'marketing']

        if date_range is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)  # 2 years
            date_range = (start_date, end_date)

        logger.info(f"Collecting data from sources: {sources}")
        logger.info(f"Date range: {date_range[0].date()} to {date_range[1].date()}")

        # Parallel data collection
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.settings.data.max_workers) as executor:
            futures = {}

            for source in sources:
                # Check cache first
                if use_cache:
                    cached_df = self.db.get_cached_dataframe(f"raw_{source}")
                    if cached_df is not None:
                        self.collected_data[source] = cached_df
                        logger.info(f"Loaded {source} from cache: {len(cached_df)} records")
                        continue

                # Submit collection task
                if source == 'customers':
                    future = executor.submit(self._collect_customers, date_range)
                elif source == 'transactions':
                    future = executor.submit(self._collect_transactions, date_range)
                elif source == 'interactions':
                    future = executor.submit(self._collect_interactions, date_range)
                elif source == 'marketing':
                    future = executor.submit(self._collect_marketing, date_range)
                else:
                    logger.warning(f"Unknown source: {source}")
                    continue

                futures[future] = source

            # Collect results
            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                try:
                    df = future.result()
                    self.collected_data[source] = df

                    # Cache the data
                    if use_cache:
                        self.db.cache_dataframe(f"raw_{source}", df, expire=3600)

                    logger.success(f"Collected {source}: {len(df)} records")

                except Exception as e:
                    logger.error(f"Failed to collect {source}: {str(e)}")
                    # Use simulated data as fallback
                    self.collected_data[source] = self._simulate_data(source)

        # Save raw data
        self._save_raw_data()

        return self.collected_data

    @retry(max_attempts=3, delay=2.0)
    def _collect_customers(self, date_range: tuple) -> pd.DataFrame:
        """Collect customer data from database"""
        query = """
        SELECT 
            customer_id,
            registration_date,
            date_of_birth,
            age,
            gender,
            country,
            state,
            city,
            postal_code,
            acquisition_channel,
            acquisition_campaign,
            customer_segment,
            customer_tier,
            lifetime_value,
            preferred_contact_method,
            email_verified,
            phone_verified,
            account_status,
            churn_date,
            churned,
            created_at,
            updated_at
        FROM customers
        WHERE registration_date BETWEEN :start_date AND :end_date
            AND account_status NOT IN ('test', 'deleted')
        """

        params = {
            'start_date': date_range[0],
            'end_date': date_range[1]
        }

        try:
            df = self.db.execute_query(query, params)
            return self._validate_customers(df)
        except Exception as e:
            logger.warning(f"Database collection failed: {str(e)}. Using simulated data.")
            return self._simulate_customers(50000, date_range)

    @retry(max_attempts=3, delay=2.0)
    def _collect_transactions(self, date_range: tuple) -> pd.DataFrame:
        """Collect transaction data"""
        query = """
        SELECT 
            t.transaction_id,
            t.customer_id,
            t.transaction_date,
            t.product_id,
            p.product_name,
            p.product_category,
            p.product_subcategory,
            t.quantity,
            t.unit_price,
            t.total_amount,
            t.discount_amount,
            t.tax_amount,
            t.payment_method,
            t.payment_status,
            t.transaction_status,
            t.channel,
            t.device_type,
            t.session_id
        FROM transactions t
        LEFT JOIN products p ON t.product_id = p.product_id
        WHERE t.transaction_date BETWEEN :start_date AND :end_date
            AND t.transaction_status = 'completed'
        """

        params = {
            'start_date': date_range[0],
            'end_date': date_range[1]
        }

        try:
            df = self.db.execute_query(query, params)
            return self._validate_transactions(df)
        except Exception as e:
            logger.warning(f"Database collection failed: {str(e)}. Using simulated data.")
            return self._simulate_transactions(date_range)

    @retry(max_attempts=3, delay=2.0)
    def _collect_interactions(self, date_range: tuple) -> pd.DataFrame:
        """Collect customer interaction data"""
        query = """
        SELECT 
            interaction_id,
            customer_id,
            interaction_date,
            interaction_type,
            interaction_subtype,
            channel,
            agent_id,
            duration_seconds,
            wait_time_seconds,
            resolution_status,
            satisfaction_score,
            nps_score,
            sentiment_score,
            ticket_id,
            tags
        FROM customer_interactions
        WHERE interaction_date BETWEEN :start_date AND :end_date
        """

        params = {
            'start_date': date_range[0],
            'end_date': date_range[1]
        }

        try:
            df = self.db.execute_query(query, params)
            return self._validate_interactions(df)
        except Exception as e:
            logger.warning(f"Database collection failed: {str(e)}. Using simulated data.")
            return self._simulate_interactions(date_range)

    def _simulate_customers(self, n_customers: int, date_range: tuple) -> pd.DataFrame:
        """Simulate realistic customer data for testing"""
        np.random.seed(self.settings.data.random_state)

        # Generate registration dates
        start_ts = date_range[0].timestamp()
        end_ts = date_range[1].timestamp()
        registration_dates = pd.to_datetime(
            np.random.uniform(start_ts, end_ts, n_customers),
            unit='s'
        )

        # Generate customer data with realistic distributions
        data = {
            'customer_id': [f'CUST_{i:08d}' for i in range(1, n_customers + 1)],
            'registration_date': registration_dates,
            'age': np.random.gamma(7, 5, n_customers).clip(18, 80).astype(int),
            'gender': np.random.choice(
                ['M', 'F', 'Other'],
                n_customers,
                p=[0.48, 0.48, 0.04]
            ),
            'country': 'USA',
            'state': np.random.choice(
                ['CA', 'TX', 'NY', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI'],
                n_customers,
                p=[0.2, 0.15, 0.15, 0.1, 0.08, 0.08, 0.06, 0.06, 0.06, 0.06]
            ),
            'acquisition_channel': np.random.choice(
                ['Organic Search', 'Paid Search', 'Social Media', 'Email', 'Referral', 'Direct'],
                n_customers,
                p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
            ),
            'customer_segment': np.random.choice(
                ['Premium', 'Standard', 'Basic', 'Trial'],
                n_customers,
                p=[0.10, 0.35, 0.40, 0.15]
            ),
            'email_verified': np.random.choice([True, False], n_customers, p=[0.85, 0.15]),
            'phone_verified': np.random.choice([True, False], n_customers, p=[0.70, 0.30]),
            'account_status': np.random.choice(
                ['active', 'inactive', 'suspended'],
                n_customers,
                p=[0.80, 0.18, 0.02]
            )
        }

        df = pd.DataFrame(data)

        # Add churned column with realistic patterns
        df['churned'] = 0

        # Higher churn for trial customers
        trial_mask = df['customer_segment'] == 'Trial'
        df.loc[trial_mask, 'churned'] = np.random.choice(
            [0, 1],
            trial_mask.sum(),
            p=[0.5, 0.5]
        )

        # Lower churn for premium customers
        premium_mask = df['customer_segment'] == 'Premium'
        df.loc[premium_mask, 'churned'] = np.random.choice(
            [0, 1],
            premium_mask.sum(),
            p=[0.92, 0.08]
        )

        # Standard churn for others
        other_mask = ~(trial_mask | premium_mask) & (df['churned'] == 0)
        df.loc[other_mask, 'churned'] = np.random.choice(
            [0, 1],
            other_mask.sum(),
            p=[0.75, 0.25]
        )

        logger.info(f"Simulated {len(df)} customer records")
        return df

    def _simulate_transactions(self, date_range: tuple) -> pd.DataFrame:
        """Simulate transaction data based on collected customers"""
        if 'customers' not in self.collected_data:
            raise ValueError("Customer data must be collected first")

        customers_df = self.collected_data['customers']
        transactions = []

        for _, customer in tqdm(customers_df.iterrows(), total=len(customers_df), desc="Generating transactions"):
            # Number of transactions based on segment and churn
            if customer['churned'] == 1:
                n_trans = np.random.poisson(5)
            elif customer['customer_segment'] == 'Premium':
                n_trans = np.random.poisson(50)
            elif customer['customer_segment'] == 'Standard':
                n_trans = np.random.poisson(20)
            else:
                n_trans = np.random.poisson(10)

            # Generate transactions
            for i in range(max(1, n_trans)):
                days_since_reg = (date_range[1] - customer['registration_date']).days
                if days_since_reg > 0:
                    trans_date = customer['registration_date'] + timedelta(
                        days=np.random.randint(0, min(days_since_reg, 365))
                    )

                    transactions.append({
                        'transaction_id': f"TXN_{customer['customer_id']}_{i:04d}",
                        'customer_id': customer['customer_id'],
                        'transaction_date': trans_date,
                        'product_category': np.random.choice(
                            ['Electronics', 'Clothing', 'Food', 'Books', 'Home', 'Sports'],
                            p=[0.25, 0.20, 0.20, 0.10, 0.15, 0.10]
                        ),
                        'total_amount': np.random.lognormal(4, 1.2) * 10,
                        'discount_amount': np.random.choice([0, 5, 10, 15, 20], p=[0.5, 0.2, 0.15, 0.1, 0.05]),
                        'payment_method': np.random.choice(
                            ['Credit Card', 'Debit Card', 'PayPal', 'Apple Pay', 'Bank Transfer'],
                            p=[0.35, 0.25, 0.20, 0.15, 0.05]
                        ),
                        'channel': np.random.choice(
                            ['Web', 'Mobile App', 'Store', 'Phone'],
                            p=[0.40, 0.35, 0.20, 0.05]
                        ),
                        'transaction_status': 'completed'
                    })

        df = pd.DataFrame(transactions)
        df['total_amount'] = df['total_amount'].round(2)

        logger.info(f"Simulated {len(df)} transaction records")
        return df

    def _simulate_interactions(self, date_range: tuple) -> pd.DataFrame:
        """Simulate customer interaction data"""
        if 'customers' not in self.collected_data:
            raise ValueError("Customer data must be collected first")

        customers_df = self.collected_data['customers']
        interactions = []

        for _, customer in tqdm(customers_df.iterrows(), total=len(customers_df), desc="Generating interactions"):
            # Number of interactions based on churn
            if customer['churned'] == 1:
                n_interactions = np.random.poisson(8)  # More interactions before churn
            else:
                n_interactions = np.random.poisson(4)

            for i in range(max(1, n_interactions)):
                days_since_reg = (date_range[1] - customer['registration_date']).days
                if days_since_reg > 0:
                    interaction_date = customer['registration_date'] + timedelta(
                        days=np.random.randint(0, min(days_since_reg, 365))
                    )

                    interaction_type = np.random.choice(
                        ['Support', 'Sales', 'Complaint', 'Inquiry', 'Feedback'],
                        p=[0.35, 0.20, 0.15, 0.20, 0.10]
                    )

                    # Satisfaction score correlates with churn
                    if customer['churned'] == 1:
                        satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.30, 0.20, 0.10])
                    else:
                        satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.10, 0.20, 0.35, 0.30])

                    interactions.append({
                        'interaction_id': f"INT_{customer['customer_id']}_{i:04d}",
                        'customer_id': customer['customer_id'],
                        'interaction_date': interaction_date,
                        'interaction_type': interaction_type,
                        'channel': np.random.choice(
                            ['Email', 'Phone', 'Chat', 'Social Media', 'In-Person'],
                            p=[0.30, 0.25, 0.25, 0.15, 0.05]
                        ),
                        'duration_seconds': np.random.gamma(2, 120),
                        'satisfaction_score': satisfaction,
                        'resolution_status': np.random.choice(
                            ['Resolved', 'Pending', 'Escalated'],
                            p=[0.70, 0.20, 0.10]
                        )
                    })

        df = pd.DataFrame(interactions)
        df['duration_seconds'] = df['duration_seconds'].round(0).astype(int)

        logger.info(f"Simulated {len(df)} interaction records")
        return df

    def _validate_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate customer data"""
        # Check required columns
        required_cols = ['customer_id', 'registration_date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Remove duplicates
        df = df.drop_duplicates(subset=['customer_id'])

        # Validate data types
        df['customer_id'] = df['customer_id'].astype(str)
        df['registration_date'] = pd.to_datetime(df['registration_date'])

        # Handle missing values
        if 'churned' not in df.columns:
            df['churned'] = 0

        return df

    def _validate_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate transaction data"""
        # Remove invalid transactions
        df = df[df['total_amount'] > 0]
        df = df[df['transaction_status'] == 'completed']

        # Validate data types
        df['customer_id'] = df['customer_id'].astype(str)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')

        return df.dropna(subset=['customer_id', 'transaction_date', 'total_amount'])

    def _validate_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate interaction data"""
        # Validate data types
        df['customer_id'] = df['customer_id'].astype(str)
        df['interaction_date'] = pd.to_datetime(df['interaction_date'])

        # Handle missing satisfaction scores
        if 'satisfaction_score' in df.columns:
            df['satisfaction_score'] = df['satisfaction_score'].fillna(3)

        return df.dropna(subset=['customer_id', 'interaction_date'])

    def _save_raw_data(self):
        """Save collected raw data to disk"""
        for name, df in self.collected_data.items():
            output_path = self.settings.paths.RAW_DATA_DIR / f"{name}_raw.parquet"
            df.to_parquet(output_path, index=False, compression='snappy')
            logger.info(f"Saved {name} to {output_path}")

    def _simulate_data(self, source: str) -> pd.DataFrame:
        """Fallback simulation for any data source"""
        date_range = (datetime.now() - timedelta(days=730), datetime.now())

        if source == 'customers':
            return self._simulate_customers(50000, date_range)
        elif source == 'transactions':
            # First ensure we have customer data
            if 'customers' not in self.collected_data:
                self.collected_data['customers'] = self._simulate_customers(50000, date_range)
            return self._simulate_transactions(date_range)
        elif source == 'interactions':
            # First ensure we have customer data
            if 'customers' not in self.collected_data:
                self.collected_data['customers'] = self._simulate_customers(50000, date_range)
            return self._simulate_interactions(date_range)
        elif source == 'marketing':
            return self._simulate_marketing_data(date_range)
        else:
            return pd.DataFrame()

    def _simulate_marketing_data(self, date_range: tuple) -> pd.DataFrame:
        """Simulate marketing campaign data"""
        if 'customers' not in self.collected_data:
            return pd.DataFrame()

        customers_df = self.collected_data['customers']
        # Sample 60% of customers for marketing
        sampled_customers = customers_df.sample(frac=0.6)

        campaigns = []
        for _, customer in sampled_customers.iterrows():
            n_campaigns = np.random.poisson(5)

            for i in range(max(1, n_campaigns)):
                campaign_date = customer['registration_date'] + timedelta(
                    days=np.random.randint(0, 365)
                )

                campaigns.append({
                    'campaign_id': f"CAMP_{customer['customer_id']}_{i:04d}",
                    'customer_id': customer['customer_id'],
                    'campaign_date': campaign_date,
                    'campaign_type': np.random.choice(
                        ['Email', 'SMS', 'Push', 'Social'],
                        p=[0.40, 0.20, 0.20, 0.20]
                    ),
                    'campaign_name': np.random.choice([
                        'Summer Sale', 'New Product Launch', 'Birthday Offer',
                        'Loyalty Reward', 'Win-back Campaign'
                    ]),
                    'opened': np.random.choice([0, 1], p=[0.6, 0.4]),
                    'clicked': np.random.choice([0, 1], p=[0.8, 0.2]),
                    'converted': np.random.choice([0, 1], p=[0.95, 0.05])
                })

        return pd.DataFrame(campaigns)