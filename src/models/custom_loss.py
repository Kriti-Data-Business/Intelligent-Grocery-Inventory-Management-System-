import numpy as np
import tensorflow as tf
from typing import Dict, Optional

class InventoryLossFunction:
    """Custom loss function incorporating business constraints"""
    
    def __init__(self, config: Dict):
        self.stockout_penalty = config.get('stockout_penalty', 50.0)
        self.holding_cost_rate = config.get('holding_cost_rate', 0.02)
        self.waste_penalty = config.get('waste_penalty', 25.0)
        self.service_level = config.get('service_level', 0.95)
    
    def inventory_cost_loss(self, y_true, y_pred, inventory_levels, costs):
        """
        Combined loss function for inventory optimization
        
        Args:
            y_true: Actual demand
            y_pred: Predicted demand
            inventory_levels: Current inventory levels
            costs: Product costs
        """
        
        # Basic prediction error
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        
        # Stockout cost (when predicted demand > inventory)
        stockout_quantity = tf.maximum(0.0, y_pred - inventory_levels)
        stockout_cost = self.stockout_penalty * stockout_quantity
        
        # Holding cost (excess inventory)
        excess_inventory = tf.maximum(0.0, inventory_levels - y_pred)
        holding_cost = self.holding_cost_rate * excess_inventory * costs
        
        # Waste cost (perishable items)
        # Assumes items expire if not sold within forecast period
        waste_quantity = tf.maximum(0.0, inventory_levels - y_true)
        waste_cost = self.waste_penalty * waste_quantity
        
        # Service level penalty
        service_penalty = self._service_level_penalty(y_true, y_pred, inventory_levels)
        
        # Total cost
        total_cost = (
            mse_loss + 
            tf.reduce_mean(stockout_cost) + 
            tf.reduce_mean(holding_cost) + 
            tf.reduce_mean(waste_cost) +
            service_penalty
        )
        
        return total_cost
    
    def _service_level_penalty(self, y_true, y_pred, inventory_levels):
        """Penalty for not meeting service level requirements"""
        
        # Service level is percentage of demand met from stock
        demand_met = tf.minimum(y_true, inventory_levels)
        service_level_achieved = demand_met / (y_true + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Penalty when service level is below target
        service_gap = tf.maximum(0.0, self.service_level - service_level_achieved)
        service_penalty = tf.reduce_mean(tf.square(service_gap)) * 100
        
        return service_penalty
    
    def quantile_loss(self, y_true, y_pred, quantile=0.95):
        """Quantile loss for probabilistic forecasting"""
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(quantile * error, (quantile - 1) * error))
    
    def asymmetric_loss(self, y_true, y_pred, over_penalty=1.0, under_penalty=2.0):
        """Asymmetric loss giving different penalties for over/under-prediction"""
        error = y_true - y_pred
        
        # Different penalties for positive and negative errors
        over_pred_loss = tf.maximum(0.0, -error) * over_penalty  # When y_pred > y_true
        under_pred_loss = tf.maximum(0.0, error) * under_penalty  # When y_pred < y_true
        
        return tf.reduce_mean(over_pred_loss + under_pred_loss)

class XGBoostInventoryObjective:
    """Custom objective function for XGBoost"""
    
    def __init__(self, stockout_penalty=50.0, holding_cost_rate=0.02):
        self.stockout_penalty = stockout_penalty
        self.holding_cost_rate = holding_cost_rate
    
    def inventory_objective(self, y_pred, dtrain):
        """Custom objective function for XGBoost inventory optimization"""
        y_true = dtrain.get_label()
        
        # Calculate residuals
        residual = y_true - y_pred
        
        # Asymmetric gradient based on over/under prediction
        grad = np.where(residual < 0, 
                       -self.stockout_penalty * residual,  # Under-prediction penalty
                       -self.holding_cost_rate * residual)  # Over-prediction penalty
        
        # Hessian (second derivative)
        hess = np.where(residual < 0, 
                       self.stockout_penalty, 
                       self.holding_cost_rate)
        
        return grad, hess
