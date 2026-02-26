import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "insider_trading_db"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    @property
    def psycopg2_params(self) -> Dict[str, Any]:
        return {"host": self.host, "port": self.port, "dbname": self.name, "user": self.user, "password": self.password}


@dataclass
class SECEdgarConfig:
    base_url: str = "https://www.sec.gov"
    rate_limit_per_second: int = 10
    user_agent: str = "InsiderTradingAnalysis/1.0 (Research Project)"
    form_types: List[str] = field(default_factory=lambda: ["4", "4/A"])
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"


@dataclass
class PathsConfig:
    raw_data: str = "data/raw"
    processed_data: str = "data/processed"
    external_data: str = "data/external"
    models: str = "models"
    logs: str = "logs"
    reports: str = "reports"

    def ensure_directories(self, base_path: Path) -> None:
        for attr in ("raw_data", "processed_data", "external_data", "models", "logs", "reports"):
            (base_path / getattr(self, attr)).mkdir(parents=True, exist_ok=True)


@dataclass
class TradeFeaturesConfig:
    log_transform_cols: list = field(default_factory=lambda: ["shares", "total_value", "shares_owned_after"])
    value_bucket_bins: list = field(default_factory=lambda: [0, 100000, 500000, 2000000, 10000000])


@dataclass
class TemporalFeaturesConfig:
    rolling_windows_days: list = field(default_factory=lambda: [7, 30, 90])
    max_days_lookback: int = 365


@dataclass
class NetworkFeaturesConfig:
    coordination_window_hours: int = 72


@dataclass
class TextFeaturesConfig:
    min_footnote_length: int = 10
    sentiment_model: str = "rule_based"


@dataclass
class FeaturesConfig:
    output_dir: str = "data/processed"
    trade: TradeFeaturesConfig = field(default_factory=TradeFeaturesConfig)
    temporal: TemporalFeaturesConfig = field(default_factory=TemporalFeaturesConfig)
    network: NetworkFeaturesConfig = field(default_factory=NetworkFeaturesConfig)
    text: TextFeaturesConfig = field(default_factory=TextFeaturesConfig)


@dataclass
class LSTMConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10


@dataclass
class RandomForestConfig:
    n_estimators: int = 200
    max_depth: int = 15
    min_samples_split: int = 5
    min_samples_leaf: int = 2


@dataclass
class GNNConfig:
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.001


@dataclass
class ModelsConfig:
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.15
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)


@dataclass
class LabelingConfig:
    abnormal_return_threshold: float = 0.15
    earnings_proximity_days: int = 30
    confidence_sources_required: int = 2


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/insider_trading.log"
    max_bytes: int = 10485760
    backup_count: int = 5


@dataclass
class Config:
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
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "Config":
        cfg = cls()
        if db := data.get("database"):
            cfg.database = DatabaseConfig(
                host=os.getenv("DB_HOST", db.get("host", "localhost")),
                port=int(os.getenv("DB_PORT", db.get("port", 5432))),
                name=os.getenv("DB_NAME", db.get("name", "insider_trading_db")),
                user=os.getenv("DB_USER", db.get("user", "postgres")),
                password=os.getenv("DB_PASSWORD", db.get("password", "")),
                pool_size=db.get("pool_size", 5),
                max_overflow=db.get("max_overflow", 10),
            )
        if sec := data.get("sec_edgar"):
            dr = sec.get("date_range", {})
            cfg.sec_edgar = SECEdgarConfig(
                base_url=sec.get("base_url", "https://www.sec.gov"),
                rate_limit_per_second=sec.get("rate_limit_per_second", 10),
                user_agent=sec.get("user_agent", "InsiderTradingAnalysis/1.0"),
                form_types=sec.get("form_types", ["4", "4/A"]),
                start_date=dr.get("start", "2020-01-01"),
                end_date=dr.get("end", "2024-12-31"),
            )
        if paths := data.get("paths"):
            cfg.paths = PathsConfig(**paths)
        if feat := data.get("features"):
            cfg.features = FeaturesConfig(
                output_dir=feat.get("output_dir", "data/processed"),
                trade=TradeFeaturesConfig(**feat.get("trade", {})),
                temporal=TemporalFeaturesConfig(**feat.get("temporal", {})),
                network=NetworkFeaturesConfig(**feat.get("network", {})),
                text=TextFeaturesConfig(**feat.get("text", {})),
            )
        if mdls := data.get("models"):
            cfg.models = ModelsConfig(
                random_seed=mdls.get("random_seed", 42),
                test_size=mdls.get("test_size", 0.2),
                validation_size=mdls.get("validation_size", 0.15),
                lstm=LSTMConfig(**mdls.get("lstm", {})),
                random_forest=RandomForestConfig(**mdls.get("random_forest", {})),
                gnn=GNNConfig(**mdls.get("gnn", {})),
            )
        if lab := data.get("labeling"):
            cfg.labeling = LabelingConfig(**lab)
        if log := data.get("logging"):
            cfg.logging = LoggingConfig(**log)
        if tickers := data.get("tickers"):
            cfg.tickers = tickers
        return cfg


_config: Optional[Config] = None


def get_config(config_path: Optional[str] = None) -> Config:
    global _config
    if _config is None:
        if config_path is None:
            config_path = str(Path(__file__).parent.parent.parent / "config" / "config.yaml")
        _config = Config.from_yaml(config_path) if Path(config_path).exists() else Config()
    return _config


def reset_config() -> None:
    global _config
    _config = None
