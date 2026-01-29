"""
Configuration Module for Insider Trading Analysis.

Provides centralized access to configuration values from YAML files
and environment variables with validation and type hints.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database connection configuration."""
    host: str = "localhost"
    port: int = 5432
    name: str = "insider_trading_db"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    
    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def psycopg2_params(self) -> Dict[str, Any]:
        """Generate psycopg2 connection parameters."""
        return {
            "host": self.host,
            "port": self.port,
            "dbname": self.name,
            "user": self.user,
            "password": self.password
        }


@dataclass
class SECEdgarConfig:
    """SEC EDGAR API configuration."""
    base_url: str = "https://www.sec.gov"
    rate_limit_per_second: int = 10
    user_agent: str = "InsiderTradingAnalysis/1.0 (Research Project)"
    form_types: List[str] = field(default_factory=lambda: ["4", "4/A"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"


@dataclass
class PathsConfig:
    """Data and output paths configuration."""
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    external_data: str = "data/external"
    models: str = "models"
    logs: str = "logs"
    reports: str = "reports"
    
    def ensure_directories(self, base_path: Path) -> None:
        """Create all configured directories if they don't exist."""
        for path_attr in ['raw_data', 'processed_data', 'external_data', 
                          'models', 'logs', 'reports']:
            path = base_path / getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)


@dataclass 
class TemporalFeaturesConfig:
    """Configuration for temporal feature engineering."""
    window_size: int = 10
    max_days_lookback: int = 365


@dataclass
class NetworkFeaturesConfig:
    """Configuration for network feature engineering."""
    coordination_window_hours: int = 72
    monthly_snapshot: bool = True


@dataclass
class NLPFeaturesConfig:
    """Configuration for NLP feature engineering."""
    min_footnote_length: int = 10
    sentiment_model: str = "textblob"


@dataclass
class FeaturesConfig:
    """Configuration for all feature engineering."""
    temporal: TemporalFeaturesConfig = field(default_factory=TemporalFeaturesConfig)
    network: NetworkFeaturesConfig = field(default_factory=NetworkFeaturesConfig)
    nlp: NLPFeaturesConfig = field(default_factory=NLPFeaturesConfig)


@dataclass
class LSTMConfig:
    """LSTM model hyperparameters."""
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class RandomForestConfig:
    """Random Forest model hyperparameters."""
    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2


@dataclass
class GNNConfig:
    """Graph Neural Network model hyperparameters."""
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.001


@dataclass
class ModelsConfig:
    """Configuration for all models."""
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.15
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)


@dataclass
class LabelingConfig:
    """Configuration for trade labeling."""
    abnormal_return_threshold: float = 0.15
    earnings_proximity_days: int = 30
    confidence_sources_required: int = 2


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/insider_trading.log"
    max_bytes: int = 10485760
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration class that holds all configuration sections."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    sec_edgar: SECEdgarConfig = field(default_factory=SECEdgarConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tickers: Dict[str, List[str]] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Config instance populated from the YAML file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            yaml.YAMLError: If the YAML is invalid.
        """
        with open(config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        """Create Config instance from dictionary."""
        config = cls()
        
        # Database config
        if 'database' in data:
            db_data = data['database']
            # Override with environment variables if present
            config.database = DatabaseConfig(
                host=os.getenv('DB_HOST', db_data.get('host', 'localhost')),
                port=int(os.getenv('DB_PORT', db_data.get('port', 5432))),
                name=os.getenv('DB_NAME', db_data.get('name', 'insider_trading_db')),
                user=os.getenv('DB_USER', db_data.get('user', 'postgres')),
                password=os.getenv('DB_PASSWORD', db_data.get('password', '')),
                pool_size=db_data.get('pool_size', 5),
                max_overflow=db_data.get('max_overflow', 10)
            )
        
        # SEC Edgar config
        if 'sec_edgar' in data:
            sec_data = data['sec_edgar']
            date_range = sec_data.get('date_range', {})
            config.sec_edgar = SECEdgarConfig(
                base_url=sec_data.get('base_url', 'https://www.sec.gov'),
                rate_limit_per_second=sec_data.get('rate_limit_per_second', 10),
                user_agent=sec_data.get('user_agent', 'InsiderTradingAnalysis/1.0'),
                form_types=sec_data.get('form_types', ['4', '4/A']),
                start_date=date_range.get('start', '2020-01-01'),
                end_date=date_range.get('end', '2024-12-31')
            )
        
        # Paths config
        if 'paths' in data:
            paths_data = data['paths']
            config.paths = PathsConfig(**paths_data)
        
        # Features config
        if 'features' in data:
            feat_data = data['features']
            config.features = FeaturesConfig(
                temporal=TemporalFeaturesConfig(**feat_data.get('temporal', {})),
                network=NetworkFeaturesConfig(**feat_data.get('network', {})),
                nlp=NLPFeaturesConfig(**feat_data.get('nlp', {}))
            )
        
        # Models config
        if 'models' in data:
            models_data = data['models']
            config.models = ModelsConfig(
                random_seed=models_data.get('random_seed', 42),
                test_size=models_data.get('test_size', 0.2),
                validation_size=models_data.get('validation_size', 0.15),
                lstm=LSTMConfig(**models_data.get('lstm', {})),
                random_forest=RandomForestConfig(**models_data.get('random_forest', {})),
                gnn=GNNConfig(**models_data.get('gnn', {}))
            )
        
        # Labeling config
        if 'labeling' in data:
            config.labeling = LabelingConfig(**data['labeling'])
        
        # Logging config
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        # Tickers
        if 'tickers' in data:
            config.tickers = data['tickers']
        
        return config


# Global configuration instance
_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.

    Args:
        config_path: Path to configuration file. If None, uses default path.

    Returns:
        Config instance.
    """
    global _config
    
    if _config is None:
        if config_path is None:
            # Default to config/config.yaml relative to project root
            project_root = Path(__file__).parent.parent.parent
            config_path = str(project_root / "config" / "config.yaml")
        
        if Path(config_path).exists():
            _config = Config.from_yaml(config_path)
        else:
            _config = Config()
    
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
