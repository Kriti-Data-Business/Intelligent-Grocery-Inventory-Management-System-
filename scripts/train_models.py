import pandas as pd
import numpy as np
from src.data.feature_engineering import FeatureEngineer
from src.models.demand_forecasting import XGBoostDemandForecaster
from src.models.clustering import LocationClusterer
from config.config import Config
import joblib
import logging

def main():
    """Main training pipeline"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = Config()
    
    # Load data
    logger.info("Loading training data...")
    train_data = pd.read_csv('data/train_data.csv')
    
    # Feature engineering
    logger.info("Performing feature engineering...")
    feature_engineer = FeatureEngineer()
    
    # Create temporal features
    train_data = feature_engineer.create_temporal_features(train_data, 'date')
    
    # Create lag features
    train_data = feature_engineer.create_lag_features(train_data, 'demand')
    
    # Create rolling features
    train_data = feature_engineer.create_rolling_features(train_data, 'demand')
    
    # Seasonal decomposition
    train_data = feature_engineer.seasonal_decomposition(train_data, 'demand')
    
    # Prepare features and target
    feature_cols = [col for col in train_data.columns if col not in ['demand', 'date', 'store_id', 'product_id']]
    X = train_data[feature_cols].fillna(0)
    y = train_data['demand']
    
    # Train XGBoost model
    logger.info("Training XGBoost model...")
    xgb_model = XGBoostDemandForecaster(config.model.xgboost_params)
    training_metrics = xgb_model.train(X, y)
    
    # Save model
    xgb_model.save_model('models/xgboost_demand_model.pkl')
    logger.info(f"XGBoost model saved. Metrics: {training_metrics}")
    
    # Train clustering model
    logger.info("Training clustering model...")
    cluster_features = ['avg_daily_sales', 'seasonality_strength', 'trend_strength', 'location_density']
    
    # Aggregate data for clustering
    store_features = train_data.groupby('store_id').agg({
        'demand': ['mean', 'std', 'max'],
        'demand_seasonal': 'std',
        'demand_trend': 'std'
    }).reset_index()
    
    store_features.columns = ['store_id', 'avg_daily_sales', 'demand_std', 'max_sales', 'seasonality_strength', 'trend_strength']
    store_features['location_density'] = np.random.uniform(0.1, 1.0, len(store_features))  # Placeholder
    
    clusterer = LocationClusterer(n_clusters=config.n_clusters)
    cluster_metrics = clusterer.fit(store_features, config.cluster_features)
    
    # Save clustering model
    joblib.dump(clusterer, 'models/location_clusterer.pkl')
    logger.info(f"Clustering model saved. Metrics: {cluster_metrics}")
    
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
