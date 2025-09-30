import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import json
from pathlib import Path
from loguru import logger

from src.utils.decorators import timer, memory_monitor
from config.settings import get_settings


class ExploratoryDataAnalysis:
    """Industrial-grade EDA module"""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.report = {}
        self.visualizations = []

        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    @timer
    @memory_monitor
    def perform_eda(self,
                    df: pd.DataFrame,
                    target_col: str = 'churned',
                    generate_plots: bool = True) -> Dict:
        """
        Perform comprehensive exploratory data analysis

        Args:
            df: Master feature DataFrame
            target_col: Target variable column name
            generate_plots: Whether to generate visualizations

        Returns:
            Dictionary containing EDA results
        """
        logger.info("=" * 50)
        logger.info("Starting Exploratory Data Analysis")
        logger.info("=" * 50)

        # 1. Data Overview
        self.report['data_overview'] = self._analyze_data_overview(df)

        # 2. Target Analysis
        if target_col in df.columns:
            self.report['target_analysis'] = self._analyze_target(df, target_col)

        # 3. Feature Analysis
        self.report['feature_analysis'] = self._analyze_features(df, target_col)

        # 4. Correlation Analysis
        self.report['correlation_analysis'] = self._analyze_correlations(df)

        # 5. Segmentation Analysis
        self.report['segmentation_analysis'] = self._analyze_segments(df, target_col)

        # 6. Statistical Tests
        if target_col in df.columns:
            self.report['statistical_tests'] = self._perform_statistical_tests(df, target_col)

        # 7. Generate Visualizations
        if generate_plots:
            self._generate_visualizations(df, target_col)

        # 8. Save Report
        self._save_report()

        logger.info("EDA completed successfully")
        return self.report

    def _analyze_data_overview(self, df: pd.DataFrame) -> Dict:
        """Analyze data overview and quality"""

        # Convert dtypes to string representation for JSON serialization
        dtype_counts = df.dtypes.value_counts()
        dtypes_dict = {str(k): int(v) for k, v in dtype_counts.items()}

        overview = {
            'shape': df.shape,
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'dtypes': dtypes_dict,  # Now uses string keys
            'missing_values': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            'missing_percentage': {str(k): float(v) for k, v in
                                   (df.isnull().sum() / len(df) * 100).round(2).to_dict().items()},
            'duplicate_rows': int(df.duplicated().sum()),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns),
            'unique_values': {str(col): int(df[col].nunique()) for col in df.columns},
            'zero_variance_features': [],
            'high_cardinality_features': []
        }

        # Identify problematic features
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() < 0.01:
                overview['zero_variance_features'].append(str(col))

        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() > len(df) * 0.5:
                overview['high_cardinality_features'].append(str(col))

        logger.info(f"Dataset shape: {overview['shape']}")
        logger.info(f"Memory usage: {overview['memory_usage_mb']:.2f} MB")
        logger.info(f"Missing values: {sum(overview['missing_values'].values())}")

        return overview

    def _analyze_target(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze target variable"""

        value_counts = df[target_col].value_counts()
        value_counts_dict = {str(k): int(v) for k, v in value_counts.to_dict().items()}

        percentage = df[target_col].value_counts(normalize=True) * 100
        percentage_dict = {str(k): float(v) for k, v in percentage.round(2).to_dict().items()}

        target_analysis = {
            'distribution': value_counts_dict,
            'percentage': percentage_dict,
            'class_ratio': None,
            'entropy': None
        }

        # Calculate class imbalance ratio
        if df[target_col].nunique() == 2:
            counts = df[target_col].value_counts()
            target_analysis['class_ratio'] = float(round(counts.iloc[0] / counts.iloc[1], 2))

            # Calculate entropy
            probs = df[target_col].value_counts(normalize=True)
            target_analysis['entropy'] = float(-sum(p * np.log2(p) for p in probs if p > 0))

        logger.info(f"Target distribution: {target_analysis['percentage']}")
        if target_analysis['class_ratio']:
            logger.info(f"Class imbalance ratio: {target_analysis['class_ratio']}")

        return target_analysis

    def _analyze_features(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze individual features"""

        feature_analysis = {
            'numeric_features': {},
            'categorical_features': {}
        }

        # Analyze numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        for col in numeric_cols[:50]:  # Limit to first 50 features
            # Convert all statistics to native Python types
            feature_analysis['numeric_features'][str(col)] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75)),
                'skewness': float(df[col].skew()),
                'kurtosis': float(df[col].kurtosis()),
                'zeros': int((df[col] == 0).sum()),
                'unique': int(df[col].nunique())
            }

        # Analyze categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col not in ['customer_id']]

        for col in categorical_cols[:20]:  # Limit to first 20 features
            value_counts = df[col].value_counts()
            top_values_dict = {str(k): int(v) for k, v in value_counts.head(10).to_dict().items()}

            feature_analysis['categorical_features'][str(col)] = {
                'unique_values': int(df[col].nunique()),
                'top_values': top_values_dict,
                'mode': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'missing': int(df[col].isnull().sum())
            }

        return feature_analysis

    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze feature correlations"""

        numeric_df = df.select_dtypes(include=[np.number])

        if len(numeric_df.columns) < 2:
            return {}

        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()

        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append({
                        'feature1': str(corr_matrix.columns[i]),
                        'feature2': str(corr_matrix.columns[j]),
                        'correlation': float(round(corr_matrix.iloc[i, j], 3))
                    })

        # Top correlations with target
        target_correlations = {}
        if 'churned' in numeric_df.columns:
            target_corr = numeric_df.corr()['churned'].abs().sort_values(ascending=False)
            target_correlations = {str(k): float(v) for k, v in target_corr[1:21].to_dict().items()}

        logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs")

        return {
            'high_correlations': high_corr_pairs,
            'target_correlations': target_correlations,
            'correlation_matrix_summary': {
                'mean_correlation': float(corr_matrix.abs().mean().mean()),
                'max_correlation': float(corr_matrix.abs().max().max())
            }
        }

    def _analyze_segments(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Analyze different customer segments"""

        segments = {}

        # Analyze by customer segment
        if 'customer_segment' in df.columns and target_col in df.columns:
            segment_analysis = df.groupby('customer_segment').agg({
                target_col: ['mean', 'count'],
                'customer_id': 'count'
            }).round(3)

            # Convert to nested dictionary with proper types
            segments['by_customer_segment'] = {}
            for idx in segment_analysis.index:
                segments['by_customer_segment'][str(idx)] = {
                    str(col[0]) + '_' + str(col[1]): float(val) if not pd.isna(val) else None
                    for col, val in segment_analysis.loc[idx].items()
                }

        # Analyze by age group
        if 'age_group' in df.columns and target_col in df.columns:
            age_analysis = df.groupby('age_group').agg({
                target_col: ['mean', 'count'],
                'customer_id': 'count'
            }).round(3)

            segments['by_age_group'] = {}
            for idx in age_analysis.index:
                segments['by_age_group'][str(idx)] = {
                    str(col[0]) + '_' + str(col[1]): float(val) if not pd.isna(val) else None
                    for col, val in age_analysis.loc[idx].items()
                }

        # Analyze by RFM segment
        if 'rfm_segment' in df.columns and target_col in df.columns:
            rfm_analysis = df.groupby('rfm_segment').agg({
                target_col: ['mean', 'count'],
                'customer_id': 'count'
            }).round(3)

            segments['by_rfm_segment'] = {}
            for idx in rfm_analysis.index:
                segments['by_rfm_segment'][str(idx)] = {
                    str(col[0]) + '_' + str(col[1]): float(val) if not pd.isna(val) else None
                    for col, val in rfm_analysis.loc[idx].items()
                }

        # Analyze by value segment
        if 'value_segment' in df.columns and target_col in df.columns:
            value_analysis = df.groupby('value_segment').agg({
                target_col: ['mean', 'count'],
                'customer_id': 'count'
            }).round(3)

            segments['by_value_segment'] = {}
            for idx in value_analysis.index:
                segments['by_value_segment'][str(idx)] = {
                    str(col[0]) + '_' + str(col[1]): float(val) if not pd.isna(val) else None
                    for col, val in value_analysis.loc[idx].items()
                }

        return segments

    def _perform_statistical_tests(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Perform statistical tests"""

        test_results = {}

        # Chi-square tests for categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        chi2_results = []

        for col in categorical_cols[:10]:  # Limit to first 10
            if col not in ['customer_id']:
                try:
                    contingency_table = pd.crosstab(df[col], df[target_col])
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
                    chi2_results.append({
                        'feature': str(col),
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'significant': bool(p_value < 0.05)
                    })
                except:
                    pass  # Skip if test fails

        test_results['chi_square_tests'] = chi2_results

        # T-tests for numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        t_test_results = []

        for col in numeric_cols[:10]:  # Limit to first 10
            if col != target_col:
                try:
                    group_0 = df[df[target_col] == 0][col].dropna()
                    group_1 = df[df[target_col] == 1][col].dropna()

                    if len(group_0) > 0 and len(group_1) > 0:
                        t_stat, p_value = stats.ttest_ind(group_0, group_1)
                        t_test_results.append({
                            'feature': str(col),
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'mean_diff': float(group_1.mean() - group_0.mean()),
                            'significant': bool(p_value < 0.05)
                        })
                except:
                    pass  # Skip if test fails

        test_results['t_tests'] = t_test_results

        return test_results

    def _generate_visualizations(self, df: pd.DataFrame, target_col: str):
        """Generate comprehensive visualizations"""

        # Create figure directory
        fig_dir = self.settings.paths.FIGURES_DIR
        fig_dir.mkdir(exist_ok=True)

        try:
            # 1. Target distribution
            self._plot_target_distribution(df, target_col)

            # 2. Feature distributions
            self._plot_feature_distributions(df, target_col)

            # 3. Correlation heatmap
            self._plot_correlation_heatmap(df)

            # 4. Segment analysis
            self._plot_segment_analysis(df, target_col)

            # 5. Time series analysis
            self._plot_time_series(df)

            # 6. Interactive dashboard
            self._create_interactive_dashboard(df, target_col)

            logger.info(f"Generated {len(self.visualizations)} visualizations")
        except Exception as e:
            logger.warning(f"Error generating visualizations: {str(e)}")

    def _plot_target_distribution(self, df: pd.DataFrame, target_col: str):
        """Plot target variable distribution"""
        try:
            if target_col not in df.columns:
                return

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Count plot
            df[target_col].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
            axes[0].set_title('Target Distribution (Count)')
            axes[0].set_xlabel(target_col)
            axes[0].set_ylabel('Count')

            # Pie chart
            df[target_col].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
            axes[1].set_title('Target Distribution (Percentage)')
            axes[1].set_ylabel('')

            plt.tight_layout()

            filepath = self.settings.paths.FIGURES_DIR / 'target_distribution.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            self.visualizations.append(str(filepath))
        except Exception as e:
            logger.warning(f"Error plotting target distribution: {str(e)}")
            plt.close()

    def _plot_feature_distributions(self, df: pd.DataFrame, target_col: str):
        """Plot feature distributions"""
        try:
            # Select top numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if target_col in df.columns:
                # Get top correlated features
                correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
                top_features = correlations[1:13].index.tolist()  # Top 12 excluding target
            else:
                top_features = numeric_cols[:12]

            if len(top_features) == 0:
                return

            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            axes = axes.ravel()

            for idx, col in enumerate(top_features):
                if idx < len(axes):
                    if target_col in df.columns:
                        # Plot distribution by target
                        df[df[target_col] == 0][col].hist(
                            bins=30, alpha=0.5, label='Not Churned',
                            ax=axes[idx], color='blue', density=True
                        )
                        df[df[target_col] == 1][col].hist(
                            bins=30, alpha=0.5, label='Churned',
                            ax=axes[idx], color='red', density=True
                        )
                        axes[idx].legend()
                    else:
                        df[col].hist(bins=30, ax=axes[idx], edgecolor='black')

                    axes[idx].set_title(f'{col}')
                    axes[idx].set_xlabel('Value')
                    axes[idx].set_ylabel('Frequency')

            plt.suptitle('Top Feature Distributions', fontsize=16)
            plt.tight_layout()

            filepath = self.settings.paths.FIGURES_DIR / 'feature_distributions.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            self.visualizations.append(str(filepath))
        except Exception as e:
            logger.warning(f"Error plotting feature distributions: {str(e)}")
            plt.close()

    def _plot_correlation_heatmap(self, df: pd.DataFrame):
        """Plot correlation heatmap"""
        try:
            # Select numeric features
            numeric_df = df.select_dtypes(include=[np.number])

            # Limit to top features if too many
            if len(numeric_df.columns) > 30:
                # Select top variance features
                variances = numeric_df.var().sort_values(ascending=False)
                top_features = variances[:30].index.tolist()
                numeric_df = numeric_df[top_features]

            if len(numeric_df.columns) < 2:
                return

            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()

            # Create heatmap
            plt.figure(figsize=(15, 12))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            sns.heatmap(
                corr_matrix,
                mask=mask,
                cmap='coolwarm',
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                annot=False
            )

            plt.title('Feature Correlation Heatmap', fontsize=16)
            plt.tight_layout()

            filepath = self.settings.paths.FIGURES_DIR / 'correlation_heatmap.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            self.visualizations.append(str(filepath))
        except Exception as e:
            logger.warning(f"Error plotting correlation heatmap: {str(e)}")
            plt.close()

    def _plot_segment_analysis(self, df: pd.DataFrame, target_col: str):
        """Plot segment analysis"""
        try:
            if target_col not in df.columns:
                return

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # Customer segment
            if 'customer_segment' in df.columns:
                segment_churn = df.groupby('customer_segment')[target_col].mean()
                segment_churn.plot(kind='bar', ax=axes[0, 0], color='steelblue')
                axes[0, 0].set_title('Churn Rate by Customer Segment')
                axes[0, 0].set_xlabel('Customer Segment')
                axes[0, 0].set_ylabel('Churn Rate')
                axes[0, 0].tick_params(axis='x', rotation=45)

            # Age group
            if 'age_group' in df.columns:
                age_churn = df.groupby('age_group')[target_col].mean()
                age_churn.plot(kind='bar', ax=axes[0, 1], color='coral')
                axes[0, 1].set_title('Churn Rate by Age Group')
                axes[0, 1].set_xlabel('Age Group')
                axes[0, 1].set_ylabel('Churn Rate')
                axes[0, 1].tick_params(axis='x', rotation=45)

            # RFM segment
            if 'rfm_segment' in df.columns:
                rfm_churn = df.groupby('rfm_segment')[target_col].mean().sort_values(ascending=False)[:10]
                rfm_churn.plot(kind='barh', ax=axes[1, 0], color='green')
                axes[1, 0].set_title('Churn Rate by RFM Segment (Top 10)')
                axes[1, 0].set_xlabel('Churn Rate')
                axes[1, 0].set_ylabel('RFM Segment')

            # Value segment
            if 'value_segment' in df.columns:
                value_churn = df.groupby('value_segment')[target_col].mean()
                value_churn.plot(kind='bar', ax=axes[1, 1], color='purple')
                axes[1, 1].set_title('Churn Rate by Value Segment')
                axes[1, 1].set_xlabel('Value Segment')
                axes[1, 1].set_ylabel('Churn Rate')
                axes[1, 1].tick_params(axis='x', rotation=45)

            plt.suptitle('Churn Analysis by Segments', fontsize=16)
            plt.tight_layout()

            filepath = self.settings.paths.FIGURES_DIR / 'segment_analysis.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            self.visualizations.append(str(filepath))
        except Exception as e:
            logger.warning(f"Error plotting segment analysis: {str(e)}")
            plt.close()

    def _plot_time_series(self, df: pd.DataFrame):
        """Plot time series analysis"""
        try:
            # Check for date columns
            if 'registration_date' not in df.columns:
                return

            fig, axes = plt.subplots(2, 1, figsize=(15, 10))

            # Registrations over time
            df_sorted = df.sort_values('registration_date')
            registrations = df_sorted.groupby(df_sorted['registration_date'].dt.to_period('M')).size()
            registrations.plot(ax=axes[0], kind='line', marker='o')
            axes[0].set_title('Customer Registrations Over Time')
            axes[0].set_xlabel('Month')
            axes[0].set_ylabel('Number of Registrations')
            axes[0].grid(True, alpha=0.3)

            # Churn rate over time
            if 'churned' in df.columns:
                monthly_churn = df_sorted.groupby(
                    df_sorted['registration_date'].dt.to_period('M')
                )['churned'].mean()
                monthly_churn.plot(ax=axes[1], kind='line', marker='o', color='red')
                axes[1].set_title('Churn Rate Over Time')
                axes[1].set_xlabel('Month')
                axes[1].set_ylabel('Churn Rate')
                axes[1].grid(True, alpha=0.3)

            plt.tight_layout()

            filepath = self.settings.paths.FIGURES_DIR / 'time_series_analysis.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()

            self.visualizations.append(str(filepath))
        except Exception as e:
            logger.warning(f"Error plotting time series: {str(e)}")
            plt.close()

    def _create_interactive_dashboard(self, df: pd.DataFrame, target_col: str):
        """Create interactive Plotly dashboard"""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Target Distribution', 'Top Features Importance',
                                'Segment Analysis', 'Time Series'),
                specs=[[{'type': 'pie'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'scatter'}]]
            )

            # 1. Target distribution pie chart
            if target_col in df.columns:
                target_counts = df[target_col].value_counts()
                fig.add_trace(
                    go.Pie(
                        labels=['Not Churned', 'Churned'],
                        values=target_counts.values,
                        hole=0.3,
                        marker_colors=['#2ecc71', '#e74c3c']
                    ),
                    row=1, col=1
                )

            # 2. Feature importance (if available)
            if hasattr(self, 'feature_importance') and self.feature_importance is not None:
                top_features = self.feature_importance.head(10)
                fig.add_trace(
                    go.Bar(
                        x=top_features['importance'],
                        y=top_features['feature'],
                        orientation='h',
                        marker_color='#3498db'
                    ),
                    row=1, col=2
                )

            # 3. Segment analysis
            if 'customer_segment' in df.columns and target_col in df.columns:
                segment_data = df.groupby('customer_segment')[target_col].mean().sort_values()
                fig.add_trace(
                    go.Bar(
                        x=segment_data.index,
                        y=segment_data.values,
                        marker_color='#9b59b6'
                    ),
                    row=2, col=1
                )

            # 4. Time series
            if 'registration_date' in df.columns:
                monthly_reg = df.groupby(df['registration_date'].dt.to_period('M')).size()
                fig.add_trace(
                    go.Scatter(
                        x=monthly_reg.index.astype(str),
                        y=monthly_reg.values,
                        mode='lines+markers',
                        marker_color='#e67e22'
                    ),
                    row=2, col=2
                )

            # Update layout
            fig.update_layout(
                title_text="CRM Analytics Dashboard",
                showlegend=False,
                height=800,
                template='plotly_white'
            )

            # Save interactive HTML
            filepath = self.settings.paths.FIGURES_DIR / 'interactive_dashboard.html'
            fig.write_html(str(filepath))

            self.visualizations.append(str(filepath))
            logger.info("Created interactive dashboard")
        except Exception as e:
            logger.warning(f"Error creating interactive dashboard: {str(e)}")

    def _save_report(self):
        """Save EDA report with proper JSON serialization"""

        # Convert report to be JSON serializable
        def make_json_serializable(obj):
            """Recursively convert non-serializable objects to serializable ones"""
            if isinstance(obj, dict):
                return {str(k): make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif pd.isna(obj):
                return None
            else:
                return obj

        # Convert report
        json_report = make_json_serializable(self.report)

        # Save JSON report
        report_path = self.settings.paths.REPORTS_DIR / 'eda_report.json'
        with open(report_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)

        logger.info(f"Saved EDA report to {report_path}")

        # Save summary text report
        summary_path = self.settings.paths.REPORTS_DIR / 'eda_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            if 'data_overview' in self.report:
                f.write("DATA OVERVIEW\n")
                f.write("-" * 40 + "\n")
                overview = self.report['data_overview']
                f.write(f"Dataset shape: {overview['shape']}\n")
                f.write(f"Memory usage: {overview['memory_usage_mb']:.2f} MB\n")
                f.write(f"Numeric features: {overview['numeric_features']}\n")
                f.write(f"Categorical features: {overview['categorical_features']}\n")
                f.write(f"Missing values: {sum(overview['missing_values'].values())}\n")
                f.write("\n")

            if 'target_analysis' in self.report:
                f.write("TARGET ANALYSIS\n")
                f.write("-" * 40 + "\n")
                target = self.report['target_analysis']
                f.write(f"Distribution: {target['percentage']}\n")
                if target['class_ratio']:
                    f.write(f"Class ratio: {target['class_ratio']}\n")
                f.write("\n")

            if 'correlation_analysis' in self.report:
                f.write("CORRELATION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                corr = self.report['correlation_analysis']
                if 'high_correlations' in corr:
                    f.write(f"Highly correlated pairs: {len(corr['high_correlations'])}\n")
                if 'target_correlations' in corr:
                    f.write("Top 5 features correlated with target:\n")
                    for feat, val in list(corr['target_correlations'].items())[:5]:
                        f.write(f"  - {feat}: {val:.3f}\n")
                f.write("\n")

            f.write("GENERATED VISUALIZATIONS\n")
            f.write("-" * 40 + "\n")
            for viz in self.visualizations:
                f.write(f"  - {Path(viz).name}\n")

        logger.info(f"Saved EDA summary to {summary_path}")