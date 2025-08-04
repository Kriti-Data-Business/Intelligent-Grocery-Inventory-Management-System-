import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, List, Tuple, Optional
import joblib
import logging

class XGBoostDemandForecaster:
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.feature_importance = None
        self.is_trained = False
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_data: Optional[Tuple] = None) -> Dict:
        """Train XGBoost model with time series cross-validation"""
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Training parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],
            'tree_method': 'hist',
            'device': 'cuda' if self._check_gpu() else 'cpu',
            **self.config
        }
        
        # Cross-validation scores
        cv_scores = {'rmse': [], 'mae': []}
        
        for train_idx, val_idx in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create DMatrix
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_fold)
            dval = xgb.DMatrix(X_val_fold, label=y_val_fold)
            
            # Train model
            model = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=1000,
                evals=[(dtrain, 'train'), (dval, 'val')],
                early_stopping_rounds=50,
                verbose_eval=False
            )
            
            # Predictions and scores
            val_pred = model.predict(dval)
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_val_fold, val_pred)))
            cv_scores['mae'].append(mean_absolute_error(y_val_fold, val_pred))
        
        # Train final model on full dataset
        dtrain_full = xgb.DMatrix(X, label=y)
        self.model = xgb.train(
            params=params,
            dtrain=dtrain_full,
            num_boost_round=int(np.mean([m.best_iteration for m in [model]])),
            verbose_eval=False
        )
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.get_score(importance_type='weight').values()
        }).sort_values('importance', ascending=False)
        
        self.is_trained = True
        
        # Return training metrics
        training_metrics = {
            'cv_rmse_mean': np.mean(cv_scores['rmse']),
            'cv_rmse_std': np.std(cv_scores['rmse']),
            'cv_mae_mean': np.mean(cv_scores['mae']),
            'cv_mae_std': np.std(cv_scores['mae']),
            'feature_importance': self.feature_importance.head(20).to_dict('records')
        }
        
        logging.info(f"Model training completed. CV RMSE: {training_metrics['cv_rmse_mean']:.4f}")
        return training_metrics
    
    def predict(self, X: pd.DataFrame, return_std: bool = False) -> np.ndarray:
        """Make predictions with optional uncertainty estimation"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)
        
        if return_std:
            # Bootstrap for uncertainty estimation
            bootstrap_preds = []
            n_bootstrap = 100
            
            for _ in range(n_bootstrap):
                # Sample with replacement
                bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
                X_bootstrap = X.iloc[bootstrap_idx]
                dbootstrap = xgb.DMatrix(X_bootstrap)
                bootstrap_preds.append(self.model.predict(dbootstrap))
            
            pred_std = np.std(bootstrap_preds, axis=0)
            return predictions, pred_std
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_importance': self.feature_importance,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        logging.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load pre-trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_importance = model_data['feature_importance']
        self.is_trained = model_data['is_trained']
        logging.info(f"Model loaded from {filepath}")
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available for XGBoost"""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except:
            return False
