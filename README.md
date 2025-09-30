# CRM Customer Churn Analysis - Phase 1: Data Foundation & Analysis

## 🎯 Project Overview

This is an industrial-grade Customer Relationship Management (CRM) analytics pipeline designed to predict customer churn using advanced machine learning techniques. Phase 1 focuses on establishing a robust data foundation through comprehensive data processing, feature engineering, and exploratory analysis.

### Key Objectives
- Build a production-ready data pipeline for CRM analytics
- Engineer 160+ meaningful features from raw customer data
- Perform comprehensive exploratory data analysis
- Prepare clean, validated datasets for Phase 2 modeling

### Success Metrics
- ✅ Pipeline execution completed in ~8 minutes
- ✅ Processed 1M+ records across 4 data sources
- ✅ Generated 167 engineered features
- ✅ Created interactive visualizations and dashboards
- ✅ Achieved 27.19% baseline churn rate for balanced modeling

## 📊 Data Statistics

Based on the successful pipeline run:

| Dataset | Records | Features | Size |
|---------|---------|----------|------|
| Customers | 50,000 | 17 | 26 MB |
| Transactions | 745,328 | 15 | 364 MB |
| Interactions | 255,080 | 10 | 108 MB |
| Marketing | 0 | 0 | 0 MB |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PIPELINE ORCHESTRATOR                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │     DATA     │→ │     DATA     │→ │     DATA     │     │
│  │  COLLECTION  │  │   CLEANING   │  │  VALIDATION  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│           ↓                ↓                 ↓              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   FEATURE    │→ │ EXPLORATORY  │→ │    MODEL     │     │
│  │ ENGINEERING  │  │   ANALYSIS   │  │ PREPARATION  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- PostgreSQL (optional, will use simulated data if unavailable)
- Redis (optional, for caching)
- 8GB RAM minimum
- 5GB free disk space

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/crm-analysis.git
cd crm-analysis
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment** (optional)
```bash
cp .env.example .env
# Edit .env with your database credentials
```

### Running the Pipeline

#### Option 1: Command Line (Recommended)
```bash
# Full pipeline
python scripts/run_pipeline.py

# Specific stages only
python scripts/run_pipeline.py --stages collection cleaning validation

# Resume from checkpoint
python scripts/run_pipeline.py --resume-from engineering

# Without visualizations (faster)
python scripts/run_pipeline.py --skip-plots

# With custom configuration
python scripts/run_pipeline.py --config config/production.yaml
```

#### Option 2: Jupyter Notebook
```bash
jupyter notebook notebooks/phase1_pipeline.ipynb
```

#### Option 3: Python Script
```python
from src.pipeline.orchestrator import PipelineOrchestrator
from config.settings import get_settings

# Initialize
settings = get_settings()
orchestrator = PipelineOrchestrator(settings)

# Run pipeline
results = orchestrator.run_pipeline()
```

## 📁 Project Structure

```
001-CRM-analysis-project/
├── config/                 # Configuration files
│   ├── settings.py        # Main settings
│   └── logging_config.py  # Logging configuration
│
├── data/                  # Data storage
│   ├── raw/              # Raw collected data
│   ├── interim/          # Cleaned data
│   └── processed/        # Final features
│
├── src/                   # Source code
│   ├── data/             # Data modules
│   │   ├── collector.py  # Data collection
│   │   ├── cleaner.py    # Data cleaning
│   │   └── validator.py  # Data validation
│   │
│   ├── features/         # Feature engineering
│   │   └── engineer.py   # Feature creation
│   │
│   ├── analysis/         # Analysis modules
│   │   └── eda.py        # Exploratory analysis
│   │
│   ├── pipeline/         # Pipeline orchestration
│   │   └── orchestrator.py
│   │
│   └── utils/            # Utilities
│       ├── database.py   # Database connections
│       └── decorators.py # Custom decorators
│
├── reports/              # Generated reports
│   ├── figures/         # Visualizations
│   ├── eda_report.json  # EDA results
│   └── pipeline_report.json
│
├── notebooks/           # Jupyter notebooks
├── tests/              # Test suite
├── scripts/            # Execution scripts
└── requirements.txt    # Dependencies
```

## 🔍 Pipeline Stages

### 1. Data Collection (117s)
- Connects to PostgreSQL database or generates simulated data
- Collects customer, transaction, interaction, and marketing data
- Implements retry logic and caching
- **Output**: Raw data in `data/raw/`

### 2. Data Cleaning (4s)
- Handles missing values with smart imputation
- Removes duplicates and invalid records
- Fixes data types and formats
- Caps outliers using IQR method
- **Output**: Clean data in `data/interim/`

### 3. Data Validation (5s)
- Checks data quality and completeness
- Validates business rules
- Identifies anomalies and issues
- Ensures foreign key integrity
- **Output**: Validation report

### 4. Feature Engineering (41s)
- Creates 167 features including:
  - Transaction aggregations (51 features)
  - Interaction metrics (29 features)
  - RFM analysis (Recency, Frequency, Monetary)
  - Behavioral patterns
  - Engagement scores
- **Output**: Master feature set in `data/processed/`

### 5. Exploratory Data Analysis (14s)
- Statistical analysis and distributions
- Correlation analysis (221 highly correlated pairs found)
- Segment analysis by customer groups
- Time series patterns
- **Output**: 
  - Interactive dashboard: `reports/figures/interactive_dashboard.html`
  - Visualizations: 6 static plots
  - JSON report: `reports/eda_report.json`

### 6. Model Preparation (1s)
- Train/validation/test split (70/15/15)
- Feature scaling and encoding
- Target class distribution:
  - Not Churned: 72.81%
  - Churned: 27.19%
- **Output**: Ready-to-model datasets

## 📈 Key Findings

### Churn Analysis
- **Overall churn rate**: 27.19%
- **Class imbalance ratio**: 2.68:1
- **High-risk segments**: Trial customers (50% churn)
- **Low-risk segments**: Premium customers (8% churn)

### Top Predictive Features
1. `interaction_to_transaction_ratio`: 0.4168
2. `transaction_count`: 0.3354
3. `frequency`: 0.3341
4. `f_score` (RFM): 0.3156
5. `avg_satisfaction`: 0.2855

### Customer Segments Performance
- **Champions**: Lowest churn, highest value
- **At Risk**: High interaction rates before churn
- **Lost**: Minimal recent activity

## 📊 Generated Outputs

### Reports
- `reports/eda_report.json` - Complete statistical analysis
- `reports/eda_summary.txt` - Human-readable summary
- `reports/cleaning_report.json` - Data cleaning details
- `reports/pipeline_report.json` - Execution metrics

### Visualizations
- `target_distribution.png` - Churn distribution
- `feature_distributions.png` - Top 12 features
- `correlation_heatmap.png` - Feature correlations
- `segment_analysis.png` - Churn by segments
- `time_series_analysis.png` - Temporal patterns
- `interactive_dashboard.html` - **Interactive Plotly dashboard**

### Datasets
- `master_features.parquet` - Complete feature set (50,000 × 167)
- `X_train.parquet` - Training features (35,000 records)
- `X_val.parquet` - Validation features (7,500 records)
- `X_test.parquet` - Test features (7,500 records)
- `feature_importance.csv` - Feature rankings

## 🔧 Configuration

### Environment Variables (.env)
```env
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crm_analytics
DB_USER=your_user
DB_PASSWORD=your_password

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379

# Performance
MAX_WORKERS=4
CHUNK_SIZE=10000
MEMORY_LIMIT_GB=8
```

### Pipeline Parameters
```python
pipeline_params = {
    'use_cache': True,           # Use Redis caching
    'continue_on_error': False,  # Stop on failure
    'generate_plots': True,      # Create visualizations
    'deep_clean': True,          # Thorough cleaning
    'target_column': 'churned'   # Target variable
}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_pipeline.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📝 Logging

Logs are stored in `logs/` directory:
- `crm_analysis.log` - All logs
- `errors.log` - Errors only
- `performance.log` - Performance metrics

View logs in real-time:
```bash
tail -f logs/crm_analysis.log
```

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Solution: Pipeline automatically falls back to simulated data
   - Check PostgreSQL is running: `pg_ctl status`

2. **Memory Error**
   - Solution: Reduce chunk size in settings
   - Set `CHUNK_SIZE=5000` in .env

3. **Redis Connection Failed**
   - Solution: Pipeline works without cache
   - Start Redis: `redis-server`

4. **Module Import Error**
   - Solution: Ensure virtual environment is activated
   - Reinstall: `pip install -r requirements.txt`

## 📊 Performance Metrics

| Stage | Duration | Memory | Records |
|-------|----------|---------|---------|
| Collection | 117s | 655 MB | 1M+ |
| Cleaning | 4s | 18 MB | 1M+ |
| Validation | 5s | - | 1M+ |
| Engineering | 41s | 260 MB | 50K |
| Analysis | 14s | 253 MB | 50K |
| Preparation | 1s | - | 50K |
| **Total** | **474s** | **1.2 GB** | **1M+** |

## 🎯 Next Steps (Phase 2)

With Phase 1 complete, proceed to Phase 2 for model development:

1. **Model Selection**
   - XGBoost for gradient boosting
   - Random Forest for ensemble learning
   - Logistic Regression for baseline

2. **Target Metrics**
   - AUC-ROC > 0.85
   - Precision > 0.75
   - Recall > 0.70

3. **Deployment**
   - API endpoint for predictions
   - Real-time scoring pipeline
   - Model monitoring dashboard

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 👥 Authors

- **Your Name** - Lead Data Scientist
- Contributors welcome!

## 🙏 Acknowledgments

- Industrial-grade pipeline architecture inspired by best practices
- Feature engineering techniques from domain expertise
- Visualization design following data science standards

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Email: support@yourcompany.com
- Documentation: [Link to docs]

---

**Ready for Phase 2?** The data foundation is complete. Proceed to model development with confidence in your data quality!