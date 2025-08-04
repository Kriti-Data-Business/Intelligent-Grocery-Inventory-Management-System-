import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import logging

class LocationClusterer:
    def __init__(self, n_clusters: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_profiles = None
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, feature_cols: List[str]) -> Dict:
        """Fit K-means++ clustering model"""
        
        # Select and scale features
        X_features = X[feature_cols].copy()
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Find optimal number of clusters using elbow method
        if self.n_clusters is None:
            self.n_clusters = self._find_optimal_clusters(X_scaled)
        
        # Fit K-means with K-means++ initialization
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=self.random_state
        )
        
        cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Create cluster profiles
        X_with_clusters = X.copy()
        X_with_clusters['cluster'] = cluster_labels
        
        self.cluster_profiles = self._create_cluster_profiles(X_with_clusters, feature_cols)
        self.is_fitted = True
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
        
        metrics = {
            'n_clusters': self.n_clusters,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'inertia': self.kmeans.inertia_,
            'cluster_sizes': pd.Series(cluster_labels).value_counts().to_dict()
        }
        
        logging.info(f"Clustering completed. Silhouette score: {silhouette_avg:.4f}")
        return metrics
    
    def predict(self, X: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Predict cluster labels for new data"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_features = X[feature_cols]
        X_scaled = self.scaler.transform(X_features)
        return self.kmeans.predict(X_scaled)
    
    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 15) -> int:
        """Find optimal number of clusters using elbow method"""
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(max_clusters + 1, len(X)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=self.random_state)
            cluster_labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
        
        # Find elbow point
        # Calculate second derivative to find the point of maximum curvature
        if len(inertias) >= 3:
            second_derivatives = []
            for i in range(1, len(inertias) - 1):
                second_derivatives.append(inertias[i-1] - 2*inertias[i] + inertias[i+1])
            
            # Find the k with maximum second derivative (elbow point)
            optimal_k_idx = np.argmax(second_derivatives) + 1
            optimal_k = K_range[optimal_k_idx]
        else:
            # Fallback to maximum silhouette score
            optimal_k = K_range[np.argmax(silhouette_scores)]
        
        logging.info(f"Optimal number of clusters: {optimal_k}")
        return optimal_k
    
    def _create_cluster_profiles(self, X: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Create detailed profiles for each cluster"""
        profiles = []
        
        for cluster_id in range(self.n_clusters):
            cluster_data = X[X['cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X) * 100
            }
            
            # Statistical summary for each feature
            for feature in feature_cols:
                profile[f'{feature}_mean'] = cluster_data[feature].mean()
                profile[f'{feature}_median'] = cluster_data[feature].median()
                profile[f'{feature}_std'] = cluster_data[feature].std()
                profile[f'{feature}_min'] = cluster_data[feature].min()
                profile[f'{feature}_max'] = cluster_data[feature].max()
            
            profiles.append(profile)
        
        return pd.DataFrame(profiles)
    
    def get_cluster_insights(self) -> Dict:
        """Generate business insights from clusters"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before generating insights")
        
        insights = {}
        
        for _, cluster in self.cluster_profiles.iterrows():
            cluster_id = int(cluster['cluster_id'])
            
            # Characterize cluster based on key metrics
            avg_sales = cluster['avg_daily_sales_mean']
            seasonality = cluster['seasonality_strength_mean']
            trend = cluster['trend_strength_mean']
            
            # Create cluster characterization
            if avg_sales > self.cluster_profiles['avg_daily_sales_mean'].median():
                sales_level = "High"
            else:
                sales_level = "Low"
            
            if seasonality > self.cluster_profiles['seasonality_strength_mean'].median():
                seasonality_level = "High"
            else:
                seasonality_level = "Low"
            
            insights[f'cluster_{cluster_id}'] = {
                'name': f"{sales_level} Sales, {seasonality_level} Seasonality",
                'size': int(cluster['size']),
                'characteristics': {
                    'avg_daily_sales': round(avg_sales, 2),
                    'seasonality_strength': round(seasonality, 3),
                    'trend_strength': round(trend, 3)
                },
                'recommendations': self._generate_recommendations(cluster)
            }
        
        return insights
    
    def _generate_recommendations(self, cluster: pd.Series) -> List[str]:
        """Generate inventory management recommendations for each cluster"""
        recommendations = []
        
        avg_sales = cluster['avg_daily_sales_mean']
        seasonality = cluster['seasonality_strength_mean']
        trend = cluster['trend_strength_mean']
        
        # Sales-based recommendations
        if avg_sales > 100:
            recommendations.append("Implement frequent delivery schedule (daily/bi-daily)")
            recommendations.append("Maintain higher safety stock levels")
        else:
            recommendations.append("Weekly delivery schedule may be sufficient")
            recommendations.append("Focus on waste reduction strategies")
        
        # Seasonality-based recommendations
        if seasonality > 0.5:
            recommendations.append("Implement seasonal forecasting models")
            recommendations.append("Adjust inventory levels based on seasonal patterns")
        else:
            recommendations.append("Use simple demand forecasting methods")
        
        # Trend-based recommendations
        if trend > 0.3:
            recommendations.append("Monitor trend changes closely")
            recommendations.append("Adjust long-term inventory planning")
        
        return recommendations
