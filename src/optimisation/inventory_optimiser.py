import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging

class InventoryOptimizer:
    """Multi-objective inventory optimization with business constraints"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stockout_penalty = config.get('stockout_penalty', 50.0)
        self.holding_cost_rate = config.get('holding_cost_rate', 0.02)
        self.waste_penalty = config.get('waste_penalty', 25.0)
        self.service_level = config.get('service_level', 0.95)
        self.max_shelf_life = config.get('max_shelf_life', 7)  # days
    
    def optimize_inventory_levels(self, 
                                demand_forecast: pd.DataFrame,
                                current_inventory: pd.DataFrame,
                                cost_data: pd.DataFrame,
                                constraints: Optional[Dict] = None) -> pd.DataFrame:
        """
        Optimize inventory levels for multiple products and locations
        
        Args:
            demand_forecast: DataFrame with columns [product_id, store_id, date, predicted_demand, demand_std]
            current_inventory: DataFrame with columns [product_id, store_id, current_stock, days_remaining]
            cost_data: DataFrame with columns [product_id, unit_cost, holding_cost, stockout_cost]
            constraints: Additional business constraints
        
        Returns:
            DataFrame with optimized inventory recommendations
        """
        
        results = []
        
        # Group by product and store for optimization
        for (product_id, store_id), group in demand_forecast.groupby(['product_id', 'store_id']):
            
            # Get current inventory and costs
            current_stock = current_inventory[
                (current_inventory['product_id'] == product_id) & 
                (current_inventory['store_id'] == store_id)
            ]['current_stock'].iloc[0] if len(current_inventory[
                (current_inventory['product_id'] == product_id) & 
                (current_inventory['store_id'] == store_id)
            ]) > 0 else 0
            
            product_cost = cost_data[cost_data['product_id'] == product_id]['unit_cost'].iloc[0]
            
            # Optimize for this product-store combination
            optimal_levels = self._optimize_single_product(
                group, current_stock, product_cost, constraints
            )
            
            optimal_levels['product_id'] = product_id
            optimal_levels['store_id'] = store_id
            results.append(optimal_levels)
        
        return pd.concat(results, ignore_index=True)
    
    def _optimize_single_product(self, 
                               demand_data: pd.DataFrame,
                               current_stock: float,
                               unit_cost: float,
                               constraints: Optional[Dict] = None) -> pd.DataFrame:
        """Optimize inventory for a single product-store combination"""
        
        # Extract demand forecast
        demand_forecast = demand_data['predicted_demand'].values
        demand_std = demand_data['demand_std'].values if 'demand_std' in demand_data.columns else np.zeros_like(demand_forecast)
        
        # Define optimization variables: order quantities for each period
        n_periods = len(demand_forecast)
        
        # Initial guess: order to meet expected demand
        x0 = np.maximum(0, demand_forecast - current_stock)
        x0[0] = max(0, sum(demand_forecast) - current_stock)  # Initial large order
        x0[1:] = 0  # No orders in subsequent periods initially
        
        # Bounds: non-negative order quantities
        bounds = [(0, None) for _ in range(n_periods)]
        
        # Constraints
        constraint_list = []
        
        # Service level constraint
        if constraints and 'min_service_level' in constraints:
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: self._service_level_constraint(x, demand_forecast, demand_std, current_stock)
            })
        
        # Storage capacity constraint
        if constraints and 'max_storage_capacity' in constraints:
            max_capacity = constraints['max_storage_capacity']
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: max_capacity - self._max_inventory_level(x, demand_forecast, current_stock)
            })
        
        # Budget constraint
        if constraints and 'max_budget' in constraints:
            max_budget = constraints['max_budget']
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x: max_budget - sum(x) * unit_cost
            })
        
        # Optimize
        result = minimize(
            fun=lambda x: self._objective_function(x, demand_forecast, demand_std, current_stock, unit_cost),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': 1000, 'ftol': 1e-6}
        )
        
        if not result.success:
            logging.warning(f"Optimization failed: {result.message}")
            # Fallback to simple reorder point policy
            optimal_orders = self._reorder_point_policy(demand_forecast, demand_std, current_stock)
        else:
            optimal_orders = result.x
        
        # Calculate recommended inventory levels
        inventory_levels = self._calculate_inventory_trajectory(
            optimal_orders, demand_forecast, current_stock
        )
        
        # Prepare results
        results_df = demand_data.copy()
        results_df['recommended_order'] = optimal_orders
        results_df['projected_inventory'] = inventory_levels
        results_df['total_cost'] = result.fun if result.success else None
        
        return results_df
    
    def _objective_function(self, order_quantities: np.ndarray,
                          demand_forecast: np.ndarray,
                          demand_std: np.ndarray,
                          current_stock: float,
                          unit_cost: float) -> float:
        """Multi-objective cost function"""
        
        n_periods = len(demand_forecast)
        total_cost = 0
        
        # Calculate inventory trajectory
        inventory_levels = self._calculate_inventory_trajectory(
            order_quantities, demand_forecast, current_stock
        )
        
        for t in range(n_periods):
            period_cost = 0
            
            # Ordering cost
            if order_quantities[t] > 0:
                period_cost += order_quantities[t] * unit_cost
            
            # Holding cost
            period_cost += self.holding_cost_rate * inventory_levels[t] * unit_cost
            
            # Expected stockout cost (using normal distribution)
            if demand_std[t] > 0:
                # Probability of stockout
                z_score = (inventory_levels[t] - demand_forecast[t]) / demand_std[t]
                stockout_prob = 1 - self._normal_cdf(z_score)
                expected_stockout = demand_std[t] * self._normal_pdf(z_score) + \
                                  (demand_forecast[t] - inventory_levels[t]) * stockout_prob
                period_cost += max(0, expected_stockout) * self.stockout_penalty
            else:
                # Deterministic case
                stockout = max(0, demand_forecast[t] - inventory_levels[t])
                period_cost += stockout * self.stockout_penalty
            
            # Waste cost (assume FIFO, items expire after max_shelf_life)
            if t >= self.max_shelf_life:
                expired_inventory = max(0, inventory_levels[t] - demand_forecast[t])
                period_cost += expired_inventory * self.waste_penalty
            
            total_cost += period_cost * (0.95 ** t)  # Discount factor
        
        return total_cost
    
    def _calculate_inventory_trajectory(self, order_quantities: np.ndarray,
                                      demand_forecast: np.ndarray,
                                      initial_stock: float) -> np.ndarray:
        """Calculate inventory levels over time"""
        n_periods = len(demand_forecast)
        inventory_levels = np.zeros(n_periods)
        
        current_inventory = initial_stock
        for t in range(n_periods):
            # Add orders
            current_inventory += order_quantities[t]
            
            # Record inventory level before demand
            inventory_levels[t] = current_inventory
            
            # Subtract demand
            current_inventory = max(0, current_inventory - demand_forecast[t])
        
        return inventory_levels
    
    def _service_level_constraint(self, order_quantities: np.ndarray,
                                demand_forecast: np.ndarray,
                                demand_std: np.ndarray,
                                current_stock: float) -> float:
        """Service level constraint function"""
        
        inventory_levels = self._calculate_inventory_trajectory(
            order_quantities, demand_forecast, current_stock
        )
        
        # Calculate average service level
        service_levels = []
        for t in range(len(demand_forecast)):
            if demand_std[t] > 0:
                z_score = (inventory_levels[t] - demand_forecast[t]) / demand_std[t]
                service_level = self._normal_cdf(z_score)
            else:
                service_level = 1.0 if inventory_levels[t] >= demand_forecast[t] else 0.0
            service_levels.append(service_level)
        
        avg_service_level = np.mean(service_levels)
        return avg_service_level - self.service_level
    
    def _max_inventory_level(self, order_quantities: np.ndarray,
                           demand_forecast: np.ndarray,
                           current_stock: float) -> float:
        """Calculate maximum inventory level"""
        inventory_levels = self._calculate_inventory_trajectory(
            order_quantities, demand_forecast, current_stock
        )
        return np.max(inventory_levels)
    
    def _reorder_point_policy(self, demand_forecast: np.ndarray,
                            demand_std: np.ndarray,
                            current_stock: float) -> np.ndarray:
        """Fallback reorder point policy"""
        
        # Simple policy: order when inventory drops below reorder point
        lead_time = 1  # Assume 1-day lead time
        safety_factor = 1.65  # For 95% service level
        
        orders = np.zeros_like(demand_forecast)
        current_inventory = current_stock
        
        for t in range(len(demand_forecast)):
            # Calculate reorder point
            lead_time_demand = demand_forecast[t] * lead_time
            safety_stock = safety_factor * demand_std[t] * np.sqrt(lead_time) if demand_std[t] > 0 else 0
            reorder_point = lead_time_demand + safety_stock
            
            # Check if we need to order
            if current_inventory < reorder_point:
                # Order enough to reach target stock level
                target_stock = reorder_point + demand_forecast[t]  # One period ahead
                orders[t] = max(0, target_stock - current_inventory)
            
            # Update inventory
            current_inventory += orders[t] - demand_forecast[t]
            current_inventory = max(0, current_inventory)
        
        return orders
    
    @staticmethod
    def _normal_cdf(x):
        """Standard normal CDF approximation"""
        return 0.5 * (1 + np.tanh(x * np.sqrt(2/np.pi)))
    
    @staticmethod
    def _normal_pdf(x):
        """Standard normal PDF"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
