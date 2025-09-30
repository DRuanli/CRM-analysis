import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from category_encoders import TargetEncoder, BinaryEncoder
from scipy import stats
from loguru import logger
from tqdm import tqdm

from src.utils.decorators import timer, memory_monitor
from config.settings import get_settings


class FeatureEngineer:
    """Industrial-grade feature engineering module"""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.encoders = {}
        self.scalers = {}
        self.feature_importance = {}

    @timer
    @memory_monitor
    def create_features(self, cleaned_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create comprehensive feature set from cleaned data

        Args:
            cleaned_data: Dictionary of cleaned DataFrames

        Returns:
            Master DataFrame with all features
        """
        logger.info("=" * 50)
        logger.info("Starting Feature Engineering Pipeline")
        logger.info("=" * 50)

        # Get base customer data
        if 'customers' not in cleaned_data:
            raise ValueError("Customer data is required for feature engineering")

        master_df = cleaned_data['customers'].copy()
        logger.info(f"Base customers: {len(master_df)}")

        # 1. Transaction features
        if 'transactions' in cleaned_data and not cleaned_data['transactions'].empty:
            transaction_features = self._create_transaction_features(cleaned_data['transactions'])
            master_df = self._merge_features(master_df, transaction_features, 'customer_id')
            logger.info(f"Added {len(transaction_features.columns) - 1} transaction features")

        # 2. Interaction features
        if 'interactions' in cleaned_data and not cleaned_data['interactions'].empty:
            interaction_features = self._create_interaction_features(cleaned_data['interactions'])
            master_df = self._merge_features(master_df, interaction_features, 'customer_id')
            logger.info(f"Added {len(interaction_features.columns) - 1} interaction features")

        # 3. Marketing features
        if 'marketing' in cleaned_data and not cleaned_data['marketing'].empty:
            marketing_features = self._create_marketing_features(cleaned_data['marketing'])
            master_df = self._merge_features(master_df, marketing_features, 'customer_id')
            logger.info(f"Added {len(marketing_features.columns) - 1} marketing features")

        # 4. RFM features
        if 'transactions' in cleaned_data:
            rfm_features = self._create_rfm_features(cleaned_data['transactions'])
            master_df = self._merge_features(master_df, rfm_features, 'customer_id')
            logger.info("Added RFM features")

        # 5. Time-based features
        master_df = self._create_time_features(master_df)

        # 6. Behavioral features
        master_df = self._create_behavioral_features(master_df)

        # 7. Engagement scores
        master_df = self._create_engagement_scores(master_df)

        # 8. Fill remaining nulls
        master_df = self._handle_feature_nulls(master_df)

        # 9. Encode categorical features
        master_df = self._encode_categorical_features(master_df)

        # 10. Feature selection (optional)
        if 'churned' in master_df.columns:
            self._calculate_feature_importance(master_df)

        logger.info(f"Final feature set: {master_df.shape}")
        logger.info(f"Total features created: {len(master_df.columns)}")

        # Save feature set
        self._save_features(master_df)

        return master_df

    def _create_transaction_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create transaction-based features"""

        features_list = []

        # Basic aggregations
        basic_agg = transactions.groupby('customer_id').agg({
            'total_amount': ['sum', 'mean', 'median', 'std', 'min', 'max', 'count'],
            'transaction_id': 'count',
            'transaction_date': ['min', 'max']
        })

        basic_agg.columns = ['_'.join(col).strip() for col in basic_agg.columns.values]
        basic_agg = basic_agg.rename(columns={
            'total_amount_sum': 'total_spent',
            'total_amount_mean': 'avg_transaction_value',
            'total_amount_median': 'median_transaction_value',
            'total_amount_std': 'transaction_value_std',
            'total_amount_min': 'min_transaction_value',
            'total_amount_max': 'max_transaction_value',
            'total_amount_count': 'transaction_count',
            'transaction_id_count': 'total_transactions',
            'transaction_date_min': 'first_transaction_date',
            'transaction_date_max': 'last_transaction_date'
        })

        # Calculate derived metrics
        basic_agg['customer_lifetime_days'] = (
                basic_agg['last_transaction_date'] - basic_agg['first_transaction_date']
        ).dt.days

        basic_agg['days_since_last_transaction'] = (
                pd.Timestamp.now() - basic_agg['last_transaction_date']
        ).dt.days

        basic_agg['purchase_frequency'] = (
                basic_agg['total_transactions'] /
                (basic_agg['customer_lifetime_days'] / 30.44)
        ).fillna(0)

        basic_agg['avg_days_between_purchases'] = (
                basic_agg['customer_lifetime_days'] /
                basic_agg['total_transactions']
        ).fillna(0)

        features_list.append(basic_agg)

        # Time-based aggregations
        for days in [30, 60, 90, 180, 365]:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            recent_trans = transactions[transactions['transaction_date'] >= cutoff_date]

            if len(recent_trans) > 0:
                recent_agg = recent_trans.groupby('customer_id').agg({
                    'total_amount': ['sum', 'count'],
                    'transaction_id': 'count'
                })

                recent_agg.columns = [f'{col[0]}_{col[1]}_last_{days}d' for col in recent_agg.columns]
                features_list.append(recent_agg)

        # Category preferences
        if 'product_category' in transactions.columns:
            category_counts = transactions.pivot_table(
                index='customer_id',
                columns='product_category',
                values='transaction_id',
                aggfunc='count',
                fill_value=0
            )
            category_counts.columns = [f'purchases_{cat.lower().replace(" ", "_")}' for cat in category_counts.columns]

            # Favorite category
            favorite_category = transactions.groupby('customer_id')['product_category'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
            ).to_frame('favorite_category')

            features_list.append(category_counts)
            features_list.append(favorite_category)

        # Payment method preferences
        if 'payment_method' in transactions.columns:
            payment_counts = transactions.pivot_table(
                index='customer_id',
                columns='payment_method',
                values='transaction_id',
                aggfunc='count',
                fill_value=0
            )
            payment_counts.columns = [f'payments_{pm.lower().replace(" ", "_")}' for pm in payment_counts.columns]
            features_list.append(payment_counts)

        # Channel preferences
        if 'channel' in transactions.columns:
            channel_counts = transactions.pivot_table(
                index='customer_id',
                columns='channel',
                values='transaction_id',
                aggfunc='count',
                fill_value=0
            )
            channel_counts.columns = [f'channel_{ch.lower().replace(" ", "_")}' for ch in channel_counts.columns]
            features_list.append(channel_counts)

        # Discount usage
        if 'discount_amount' in transactions.columns:
            discount_features = transactions.groupby('customer_id').agg({
                'discount_amount': ['sum', 'mean', 'max']
            })
            discount_features.columns = ['total_discount_received', 'avg_discount_rate', 'max_discount_received']

            # Discount usage rate
            discount_usage = (transactions['discount_amount'] > 0).groupby(
                transactions['customer_id']
            ).mean().to_frame('discount_usage_rate')

            features_list.append(discount_features)
            features_list.append(discount_usage)

        # Weekend vs weekday
        if 'is_weekend' in transactions.columns:
            weekend_features = transactions.groupby('customer_id')['is_weekend'].agg([
                'sum',
                'mean'
            ])
            weekend_features.columns = ['weekend_transactions', 'weekend_transaction_rate']
            features_list.append(weekend_features)

        # Merge all features
        result = pd.concat(features_list, axis=1)
        result = result.reset_index()

        return result

    def _create_interaction_features(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """Create interaction-based features"""

        features_list = []

        # Basic aggregations
        basic_agg = interactions.groupby('customer_id').agg({
            'interaction_id': 'count',
            'satisfaction_score': ['mean', 'median', 'std', 'min', 'max'],
            'duration_seconds': ['mean', 'median', 'sum', 'max']
        })

        basic_agg.columns = ['_'.join(col).strip() for col in basic_agg.columns.values]
        basic_agg = basic_agg.rename(columns={
            'interaction_id_count': 'total_interactions',
            'satisfaction_score_mean': 'avg_satisfaction',
            'satisfaction_score_median': 'median_satisfaction',
            'satisfaction_score_std': 'satisfaction_std',
            'satisfaction_score_min': 'min_satisfaction',
            'satisfaction_score_max': 'max_satisfaction',
            'duration_seconds_mean': 'avg_interaction_duration',
            'duration_seconds_median': 'median_interaction_duration',
            'duration_seconds_sum': 'total_interaction_time',
            'duration_seconds_max': 'max_interaction_duration'
        })

        features_list.append(basic_agg)

        # Interaction types
        if 'interaction_type' in interactions.columns:
            type_counts = interactions.pivot_table(
                index='customer_id',
                columns='interaction_type',
                values='interaction_id',
                aggfunc='count',
                fill_value=0
            )
            type_counts.columns = [f'interactions_{it.lower().replace(" ", "_")}' for it in type_counts.columns]

            # Support intensity
            if 'interactions_support' in type_counts.columns and 'interactions_complaint' in type_counts.columns:
                support_intensity = (
                        (type_counts.get('interactions_support', 0) +
                         type_counts.get('interactions_complaint', 0)) /
                        (type_counts.sum(axis=1) + 1e-10)
                ).to_frame('support_intensity')
                features_list.append(support_intensity)

            features_list.append(type_counts)

        # Channel preferences
        if 'channel' in interactions.columns:
            channel_counts = interactions.pivot_table(
                index='customer_id',
                columns='channel',
                values='interaction_id',
                aggfunc='count',
                fill_value=0
            )
            channel_counts.columns = [f'interaction_channel_{ch.lower().replace(" ", "_")}' for ch in
                                      channel_counts.columns]

            # Preferred channel
            preferred_channel = interactions.groupby('customer_id')['channel'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
            ).to_frame('preferred_interaction_channel')

            features_list.append(channel_counts)
            features_list.append(preferred_channel)

        # Resolution status
        if 'resolution_status' in interactions.columns:
            resolution_rate = (interactions['resolution_status'] == 'Resolved').groupby(
                interactions['customer_id']
            ).mean().to_frame('resolution_rate')
            features_list.append(resolution_rate)

        # Recent interactions
        for days in [30, 60, 90]:
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            recent_int = interactions[interactions['interaction_date'] >= cutoff_date]

            if len(recent_int) > 0:
                recent_agg = recent_int.groupby('customer_id').agg({
                    'interaction_id': 'count',
                    'satisfaction_score': 'mean'
                })
                recent_agg.columns = [f'{col}_last_{days}d' for col in recent_agg.columns]
                features_list.append(recent_agg)

        # Merge all features
        result = pd.concat(features_list, axis=1)
        result = result.reset_index()

        return result

    def _create_marketing_features(self, marketing: pd.DataFrame) -> pd.DataFrame:
        """Create marketing engagement features"""

        features = marketing.groupby('customer_id').agg({
            'campaign_id': 'count',
            'opened': ['sum', 'mean'],
            'clicked': ['sum', 'mean'],
            'converted': ['sum', 'mean']
        })

        features.columns = ['_'.join(col).strip() for col in features.columns.values]
        features = features.rename(columns={
            'campaign_id_count': 'total_campaigns_received',
            'opened_sum': 'total_emails_opened',
            'opened_mean': 'email_open_rate',
            'clicked_sum': 'total_emails_clicked',
            'clicked_mean': 'click_through_rate',
            'converted_sum': 'total_conversions',
            'converted_mean': 'conversion_rate'
        })

        # Marketing engagement score
        features['marketing_engagement_score'] = (
                features['email_open_rate'] * 0.3 +
                features['click_through_rate'] * 0.3 +
                features['conversion_rate'] * 0.4
        )

        # Campaign type preferences
        if 'campaign_type' in marketing.columns:
            campaign_types = marketing.pivot_table(
                index='customer_id',
                columns='campaign_type',
                values='campaign_id',
                aggfunc='count',
                fill_value=0
            )
            campaign_types.columns = [f'campaigns_{ct.lower()}' for ct in campaign_types.columns]

            features = pd.concat([features, campaign_types], axis=1)

        return features.reset_index()

    def _create_rfm_features(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Create RFM (Recency, Frequency, Monetary) features"""

        # Calculate RFM values
        rfm = transactions.groupby('customer_id').agg({
            'transaction_date': lambda x: (pd.Timestamp.now() - x.max()).days,
            'transaction_id': 'count',
            'total_amount': 'sum'
        }).reset_index()

        rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']

        # Create RFM scores (1-5)
        rfm['r_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm['m_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')

        # Convert scores to numeric
        for col in ['r_score', 'f_score', 'm_score']:
            rfm[col] = rfm[col].astype(int)

        # Combined RFM score
        rfm['rfm_score'] = (
                rfm['r_score'].astype(str) +
                rfm['f_score'].astype(str) +
                rfm['m_score'].astype(str)
        )

        # RFM segments
        def rfm_segment(row):
            score = str(row['rfm_score'])
            if score in ['555', '554', '544', '545', '454', '455', '445']:
                return 'Champions'
            elif score in ['543', '444', '435', '355', '354', '345', '344', '335']:
                return 'Loyal Customers'
            elif score in ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451']:
                return 'Potential Loyalists'
            elif score in ['512', '511', '422', '421', '412', '411', '311']:
                return 'New Customers'
            elif score in ['525', '524', '523', '522', '521', '515', '514', '513']:
                return 'Promising'
            elif score in ['535', '534', '443', '434', '343', '334', '325', '324']:
                return 'Need Attention'
            elif score in ['155', '154', '144', '214', '215', '115', '114', '113']:
                return 'At Risk'
            elif score in ['255', '254', '245', '244', '253', '252', '243', '242']:
                return 'Cannot Lose Them'
            elif score in ['111', '112', '121', '131', '141', '151']:
                return 'Lost'
            else:
                return 'Other'

        rfm['rfm_segment'] = rfm.apply(rfm_segment, axis=1)

        return rfm

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""

        # Registration features
        if 'registration_date' in df.columns:
            df['registration_year'] = df['registration_date'].dt.year
            df['registration_month'] = df['registration_date'].dt.month
            df['registration_quarter'] = df['registration_date'].dt.quarter
            df['registration_dayofweek'] = df['registration_date'].dt.dayofweek
            df['registration_is_weekend'] = df['registration_dayofweek'].isin([5, 6]).astype(int)

            # Seasonality
            df['registration_season'] = df['registration_month'].apply(
                lambda x: 'Winter' if x in [12, 1, 2] else
                'Spring' if x in [3, 4, 5] else
                'Summer' if x in [6, 7, 8] else 'Fall'
            )

        # Customer lifetime
        if 'account_age_days' in df.columns:
            df['account_age_months'] = df['account_age_days'] / 30.44
            df['account_age_years'] = df['account_age_days'] / 365.25

            # Lifetime segments
            df['lifetime_segment'] = pd.cut(
                df['account_age_days'],
                bins=[0, 30, 90, 180, 365, 730, float('inf')],
                labels=['New', 'Getting Started', 'Regular', 'Established', 'Loyal', 'Veteran']
            )

        return df

    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create behavioral pattern features"""

        # Purchase behavior
        if 'total_transactions' in df.columns and 'account_age_months' in df.columns:
            df['monthly_transaction_rate'] = df['total_transactions'] / (df['account_age_months'] + 1)

        if 'total_spent' in df.columns and 'total_transactions' in df.columns:
            df['avg_basket_size'] = df['total_spent'] / (df['total_transactions'] + 1e-10)

        # Engagement patterns
        if 'total_interactions' in df.columns and 'total_transactions' in df.columns:
            df['interaction_to_transaction_ratio'] = (
                    df['total_interactions'] / (df['total_transactions'] + 1e-10)
            )

        # Value segments
        if 'total_spent' in df.columns:
            df['value_segment'] = pd.qcut(
                df['total_spent'],
                q=4,
                labels=['Low Value', 'Medium Value', 'High Value', 'VIP'],
                duplicates='drop'
            )

        # Activity level
        if 'days_since_last_transaction' in df.columns:
            df['activity_status'] = pd.cut(
                df['days_since_last_transaction'],
                bins=[0, 30, 60, 90, 180, float('inf')],
                labels=['Very Active', 'Active', 'Moderate', 'Inactive', 'Dormant']
            )

        return df

    def _create_engagement_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite engagement scores"""

        # Overall engagement score
        engagement_components = []

        if 'purchase_frequency' in df.columns:
            purchase_score = df['purchase_frequency'].rank(pct=True)
            engagement_components.append(purchase_score * 0.3)

        if 'total_spent' in df.columns:
            monetary_score = df['total_spent'].rank(pct=True)
            engagement_components.append(monetary_score * 0.3)

        if 'days_since_last_transaction' in df.columns:
            recency_score = 1 - df['days_since_last_transaction'].rank(pct=True)
            engagement_components.append(recency_score * 0.2)

        if 'avg_satisfaction' in df.columns:
            satisfaction_score = df['avg_satisfaction'] / 5
            engagement_components.append(satisfaction_score * 0.2)

        if engagement_components:
            df['engagement_score'] = sum(engagement_components)

            # Engagement level
            df['engagement_level'] = pd.qcut(
                df['engagement_score'],
                q=4,
                labels=['Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'
            )

        # Health score (inverse of churn probability indicator)
        health_components = []

        if 'resolution_rate' in df.columns:
            health_components.append(df['resolution_rate'] * 0.2)

        if 'support_intensity' in df.columns:
            health_components.append((1 - df['support_intensity']) * 0.3)

        if 'email_open_rate' in df.columns:
            health_components.append(df['email_open_rate'] * 0.2)

        if 'fully_verified' in df.columns:
            health_components.append(df['fully_verified'] * 0.3)

        if health_components:
            df['health_score'] = sum(health_components)

        return df

    def _merge_features(self, base_df: pd.DataFrame, features_df: pd.DataFrame, on: str) -> pd.DataFrame:
        """Safely merge feature DataFrames"""
        # Ensure merge key is string type in both DataFrames
        base_df[on] = base_df[on].astype(str)
        features_df[on] = features_df[on].astype(str)

        result = base_df.merge(features_df, on=on, how='left')

        return result

    def _handle_feature_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle null values in engineered features"""

        # Numeric columns - fill with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Categorical columns - fill with 'Unknown'
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['customer_id']:
                df[col] = df[col].fillna('Unknown')

        # Specific handling for certain features
        infinity_cols = df.columns[df.isin([np.inf, -np.inf]).any()]
        for col in infinity_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0)

        return df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""

        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['customer_id']]

        for col in categorical_cols:
            unique_values = df[col].nunique()

            if unique_values == 2:
                # Binary encoding
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col])
                self.encoders[col] = le

            elif unique_values <= self.settings.features.high_cardinality_threshold:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)

            else:
                # Target encoding for high cardinality
                if 'churned' in df.columns:
                    te = TargetEncoder()
                    df[f'{col}_target_encoded'] = te.fit_transform(df[col], df['churned'])
                    self.encoders[col] = te
                else:
                    # Fallback to label encoding
                    le = LabelEncoder()
                    df[f'{col}_label_encoded'] = le.fit_transform(df[col])
                    self.encoders[col] = le

        return df

    def _calculate_feature_importance(self, df: pd.DataFrame):
        """Calculate feature importance using mutual information"""

        # Prepare features and target
        feature_cols = [col for col in df.columns if col not in ['customer_id', 'churned']]
        X = df[feature_cols].select_dtypes(include=[np.number])
        y = df['churned']

        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y, random_state=self.settings.data.random_state)

        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)

        self.feature_importance = importance_df

        # Log top features
        logger.info("Top 20 most important features:")
        for idx, row in importance_df.head(20).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    def _save_features(self, df: pd.DataFrame):
        """Save engineered features"""
        output_path = self.settings.paths.PROCESSED_DATA_DIR / "master_features.parquet"
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Saved master feature set to {output_path}")

        # Save feature importance
        if hasattr(self, 'feature_importance') and self.feature_importance is not None:
            importance_path = self.settings.paths.PROCESSED_DATA_DIR / "feature_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            logger.info(f"Saved feature importance to {importance_path}")