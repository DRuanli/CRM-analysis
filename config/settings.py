import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from functools import lru_cache
import json
import yaml
from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator
import logging

# Load environment variables
load_dotenv()


@dataclass
class ProjectPaths:
    """Project path configuration"""
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    RAW_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "raw")
    INTERIM_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "interim")
    PROCESSED_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "processed")
    EXTERNAL_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "external")
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models")
    REPORTS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "reports")
    FIGURES_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "reports" / "figures")
    LOGS_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs")
    CONFIG_DIR: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "config")

    def __post_init__(self):
        """Create directories if they don't exist"""
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, Path) and attr_name.endswith('_DIR'):
                attr_value.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseSettings):
    """Database configuration using Pydantic"""
    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    name: str = Field(default="crm_analytics", env="DB_NAME")
    user: str = Field(default="analyst", env="DB_USER")
    password: str = Field(default="password", env="DB_PASSWORD")

    @property
    def connection_string(self) -> str:
        """Generate connection string"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    class Config:
        env_file = ".env"
        case_sensitive = False


class DataConfig(BaseSettings):
    """Data processing configuration"""
    chunk_size: int = Field(default=10000, env="CHUNK_SIZE")
    max_workers: int = Field(default=4, env="MAX_WORKERS")
    memory_limit_gb: int = Field(default=8, env="MEMORY_LIMIT_GB")
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.1

    # Data quality thresholds
    max_missing_percentage: float = 0.05
    outlier_std_threshold: float = 3.0
    min_samples_per_class: int = 100

    # Business rules
    churn_definition_days: int = 90
    min_customer_lifetime_days: int = 30
    min_transaction_amount: float = 0.01
    max_transaction_amount: float = 1000000.0

    class Config:
        env_file = ".env"


class FeatureConfig(BaseSettings):
    """Feature engineering configuration"""

    # RFM Configuration
    rfm_recency_bins: int = 5
    rfm_frequency_bins: int = 5
    rfm_monetary_bins: int = 5

    # Time windows for aggregation
    aggregation_windows: list = [7, 30, 90, 180, 365]

    # Feature selection
    correlation_threshold: float = 0.95
    variance_threshold: float = 0.01
    mutual_info_threshold: float = 0.01

    # Encoding
    high_cardinality_threshold: int = 50
    rare_category_threshold: float = 0.01


@dataclass
class ModelConfig:
    """Model configuration for Phase 2 prep"""
    model_registry_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name: str = "crm_churn_prediction"

    # Model parameters
    cv_folds: int = 5
    scoring_metric: str = "roc_auc"
    early_stopping_rounds: int = 50

    # Hyperparameter search
    n_iter_random_search: int = 100
    n_trials_optuna: int = 200


class Settings:
    """Central configuration management"""

    def __init__(self):
        self.env = os.getenv("ENV", "development")
        self.paths = ProjectPaths()
        self.database = DatabaseConfig()
        self.data = DataConfig()
        self.features = FeatureConfig()
        self.model = ModelConfig()

        # Logging configuration
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = self.paths.LOGS_DIR / "crm_analysis.log"

        # Performance settings
        self.n_jobs = -1 if self.data.max_workers == -1 else self.data.max_workers
        self.verbose = 1 if self.env == "development" else 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "env": self.env,
            "paths": {k: str(v) for k, v in self.paths.__dict__.items()},
            "database": self.database.dict(),
            "data": self.data.dict(),
            "features": self.features.dict(),
            "model": self.model.__dict__
        }

    def save_config(self, filepath: Optional[Path] = None):
        """Save configuration to file"""
        if filepath is None:
            filepath = self.paths.CONFIG_DIR / f"config_{self.env}.json"

        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load_config(cls, filepath: Path) -> 'Settings':
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        settings = cls()
        # Update settings from loaded config
        return settings


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()