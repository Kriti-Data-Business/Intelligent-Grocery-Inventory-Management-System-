import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    xgboost_params: Dict = None
    ensemble_weights: Dict = None
    feature_columns: List[str] = None
    target_column: str = "demand"
    
    def __post_init__(self):
        if self.xgboost_params is None:
            self.xgboost_params = {
                'n_estimators': 1000,
                'max_depth': 8,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'xgboost': 0.4,
                'lstm': 0.3,
                'arima': 0.2,
                'prophet': 0.1
            }

@dataclass
class SparkConfig:
    app_name: str = "InventoryOptimization"
    master: str = "local[*]"
    executor_memory: str = "4g"
    driver_memory: str = "2g"
    max_result_size: str = "2g"

@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    inventory_topic: str = "inventory_updates"
    demand_topic: str = "demand_predictions"
    consumer_group: str = "inventory_optimizer"

@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "inventory_db")
    username: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "password")

class Config:
    def __init__(self):
        self.model = ModelConfig()
        self.spark = SparkConfig()
        self.kafka = KafkaConfig()
        self.database = DatabaseConfig()
        
        # Business constraints
        self.stockout_penalty = 50.0
        self.holding_cost_rate = 0.02
        self.waste_penalty = 25.0
        self.service_level = 0.95
        
        # Clustering parameters
        self.n_clusters = 8
        self.cluster_features = [
            'avg_daily_sales', 'seasonality_strength', 
            'trend_strength', 'location_density'
        ]
