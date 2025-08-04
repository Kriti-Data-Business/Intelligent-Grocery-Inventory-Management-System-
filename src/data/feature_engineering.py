import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
from statsmodels.tsa.seasonal import STL
import holidays

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.holiday_calendar = holidays.India()
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic temporal features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['hour'] = df[date_col].dt.hour
        df['quarter'] = df[date_col].dt.quarter
        
        # Cyclical encoding
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Holiday features
        df['is_holiday'] = df[date_col].apply(lambda x: x.date() in self.holiday_calendar)
        df['days_to_holiday'] = df[date_col].apply(self._days_to_next_holiday)
        df['days_from_holiday'] = df[date_col].apply(self._days_from_last_holiday)
        
        # Weekend and special days
        df['is_weekend'] = df['dayofweek'].isin([5, 6])
        df['is_month_start'] = df[date_col].dt.is_month_start
        df['is_month_end'] = df[date_col].dt.is_month_end
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                          lags: List[int] = [1, 2, 3, 7, 14, 30]) -> pd.DataFrame:
        """Create lag features for time series"""
        df_sorted = df.sort_values('date')
        
        for lag in lags:
            df_sorted[f'{target_col}_lag_{lag}'] = df_sorted.groupby('store_id')[target_col].shift(lag)
        
        return df_sorted
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str,
                              windows: List[int] = [3, 7, 14, 30]) -> pd.DataFrame:
        """Create rolling statistical features"""
        df_sorted = df.sort_values('date')
        
        for window in windows:
            # Rolling statistics
            df_sorted[f'{target_col}_rolling_mean_{window}'] = (
                df_sorted.groupby('store_id')[target_col]
                .rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            )
            df_sorted[f'{target_col}_rolling_std_{window}'] = (
                df_sorted.groupby('store_id')[target_col]
                .rolling(window=window, min_periods=1).std().reset_index(0, drop=True)
            )
            df_sorted[f'{target_col}_rolling_max_{window}'] = (
                df_sorted.groupby('store_id')[target_col]
                .rolling(window=window, min_periods=1).max().reset_index(0, drop=True)
            )
            df_sorted[f'{target_col}_rolling_min_{window}'] = (
                df_sorted.groupby('store_id')[target_col]
                .rolling(window=window, min_periods=1).min().reset_index(0, drop=True)
            )
        
        return df_sorted
    
    def seasonal_decomposition(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Apply STL decomposition for seasonal patterns"""
        df_with_seasonal = df.copy()
        
        for store_id in df['store_id'].unique():
            store_data = df[df['store_id'] == store_id][target_col]
            
            if len(store_data) >= 14:  # Minimum data points for decomposition
                try:
                    stl = STL(store_data, seasonal=7)  # Weekly seasonality
                    result = stl.fit()
                    
                    mask = df['store_id'] == store_id
                    df_with_seasonal.loc[mask, f'{target_col}_trend'] = result.trend
                    df_with_seasonal.loc[mask, f'{target_col}_seasonal'] = result.seasonal
                    df_with_seasonal.loc[mask, f'{target_col}_residual'] = result.resid
                except:
                    # Fallback if decomposition fails
                    mask = df['store_id'] == store_id
                    df_with_seasonal.loc[mask, f'{target_col}_trend'] = store_data.mean()
                    df_with_seasonal.loc[mask, f'{target_col}_seasonal'] = 0
                    df_with_seasonal.loc[mask, f'{target_col}_residual'] = 0
        
        return df_with_seasonal
    
    def create_weather_features(self, df: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Merge and engineer weather features"""
        # Merge weather data
        df_weather = df.merge(weather_data, on=['date', 'location'], how='left')
        
        # Weather interaction features
        df_weather['temp_humidity_interaction'] = (
            df_weather['temperature'] * df_weather['humidity'] / 100
        )
        df_weather['feels_like_temp'] = (
            df_weather['temperature'] - 
            (df_weather['wind_speed'] * 0.1) + 
            (df_weather['humidity'] * 0.05)
        )
        
        # Weather categories
        df_weather['temp_category'] = pd.cut(
            df_weather['temperature'], 
            bins=[-np.inf, 15, 25, 35, np.inf],
            labels=['cold', 'cool', 'warm', 'hot']
        )
        
        return df_weather
    
    def _days_to_next_holiday(self, date):
        """Calculate days to next holiday"""
        for i in range(1, 30):
            future_date = date + pd.Timedelta(days=i)
            if future_date.date() in self.holiday_calendar:
                return i
        return 30
    
    def _days_from_last_holiday(self, date):
        """Calculate days from last holiday"""
        for i in range(1, 30):
            past_date = date - pd.Timedelta(days=i)
            if past_date.date() in self.holiday_calendar:
                return i
        return 30
