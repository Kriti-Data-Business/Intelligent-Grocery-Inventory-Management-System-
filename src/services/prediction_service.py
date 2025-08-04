from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
import redis
from typing import Dict, List, Optional
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

# Prometheus metrics
REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('prediction_request_duration_seconds', 'Request latency')

class PredictionService:
    """Microservice for demand prediction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.app = Flask(__name__)
        self.models = {}
        self.redis_client = None
        self.feature_engineer = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_redis()
        self._load_models()
        self._setup_routes()
    
    def _initialize_redis(self):
        """Initialize Redis for caching"""
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=0,
                decode_responses=True
            )
            self.redis_client.ping()
            self.logger.info("Redis connection established")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            # Load XGBoost model
            self.models['xgboost'] = joblib.load('models/xgboost_demand_model.pkl')
            
            # Load TensorFlow model
            self.models['lstm'] = tf.keras.models.load_model('models/lstm_demand_model.h5')
            
            # Load ensemble weights
            with open('models/ensemble_weights.json', 'r') as f:
                import json
                self.ensemble_weights = json.load(f)
            
            self.logger.info("Models loaded successfully")
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            raise
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
        
        @self.app.route('/predict', methods=['POST'])
        @REQUEST_LATENCY.time()
        def predict():
            REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
            
            try:
                data = request.get_json()
                
                # Validate input
                required_fields = ['store_id', 'product_id', 'date', 'features']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400
                
                # Make prediction
                prediction_result = self._make_prediction(data)
                
                return jsonify(prediction_result)
                
            except Exception as e:
                self.logger.error(f"Prediction error: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/predict/batch', methods=['POST'])
        @REQUEST_LATENCY.time()
        def predict_batch():
            REQUEST_COUNT.labels(method='POST', endpoint='/predict/batch').inc()
            
            try:
                data = request.get_json()
                predictions = []
                
                for item in data['items']:
                    prediction = self._make_prediction(item)
                    predictions.append(prediction)
                
                return jsonify({'predictions': predictions})
                
            except Exception as e:
                self.logger.error(f"Batch prediction error: {e}")
                return jsonify({'error': 'Internal server error'}), 500
        
        @self.app.route('/metrics', methods=['GET'])
        def metrics():
            return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}
    
    def _make_prediction(self, data: Dict) -> Dict:
        """Make demand prediction using ensemble model"""
        
        # Check cache first
        cache_key = f"prediction:{data['store_id']}:{data['product_id']}:{data['date']}"
        if self.redis_client:
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                import json
                return json.loads(cached_result)
        
        # Prepare features
        features_df = pd.DataFrame([data['features']])
        
        # Make predictions with different models
        predictions = {}
        
        # XGBoost prediction
        if 'xgboost' in self.models:
            xgb_pred = self.models['xgboost'].predict(features_df)[0]
            predictions['xgboost'] = float(xgb_pred)
        
        # LSTM prediction (if sequence data available)
        if 'lstm' in self.models and 'sequence_features' in data:
            sequence_data = np.array(data['sequence_features']).reshape(1, -1, len(data['sequence_features'][0]))
            lstm_pred = self.models['lstm'].predict(sequence_data)[0][0]
            predictions['lstm'] = float(lstm_pred)
        
        # Ensemble prediction
        ensemble_pred = 0
        total_weight = 0
        
        for model_name, weight in self.ensemble_weights.items():
            if model_name in predictions:
                ensemble_pred += predictions[model_name] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        # Calculate prediction intervals (simplified)
        pred_std = np.std(list(predictions.values())) if len(predictions) > 1 else ensemble_pred * 0.2
        lower_bound = max(0, ensemble_pred - 1.96 * pred_std)
        upper_bound = ensemble_pred + 1.96 * pred_std
        
        result = {
            'store_id': data['store_id'],
            'product_id': data['product_id'],
            'date': data['date'],
            'predicted_demand': ensemble_pred,
            'prediction_interval': {
                'lower': lower_bound,
                'upper': upper_bound
            },
            'model_predictions': predictions,
            'timestamp': datetime.now().isoformat()
        }
        
        # Cache result
        if self.redis_client:
            import json
            self.redis_client.setex(cache_key, 3600, json.dumps(result))  # Cache for 1 hour
        
        return result
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask application"""
        self.app.run(host=host, port=port, debug=debug)
