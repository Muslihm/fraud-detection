"""
Feature engineering module for fraud detection.
Creates advanced features for fraud pattern detection.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering class for fraud detection."""
    
    def __init__(self):
        self.transaction_velocity_features = []
        
    def create_time_features(self, df: pd.DataFrame, time_column: str) -> pd.DataFrame:
        """
        Create time-based features from timestamp.
        
        Args:
            df: Dataframe with timestamp column
            time_column: Name of timestamp column
            
        Returns:
            Dataframe with new time features
        """
        logger.info("Creating time-based features...")
        
        df_features = df.copy()
        
        # Extract time components
        df_features['hour_of_day'] = df_features[time_column].dt.hour
        df_features['day_of_week'] = df_features[time_column].dt.dayofweek
        df_features['month'] = df_features[time_column].dt.month
        df_features['quarter'] = df_features[time_column].dt.quarter
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Time-based risk indicators
        # Late night transactions (riskier)
        df_features['is_late_night'] = ((df_features['hour_of_day'] >= 0) & 
                                        (df_features['hour_of_day'] <= 5)).astype(int)
        
        # Early morning transactions
        df_features['is_early_morning'] = ((df_features['hour_of_day'] >= 5) & 
                                           (df_features['hour_of_day'] <= 8)).astype(int)
        
        # Business hours
        df_features['is_business_hours'] = ((df_features['hour_of_day'] >= 9) & 
                                            (df_features['hour_of_day'] <= 17)).astype(int)
        
        logger.info(f"Created {len(df_features.columns) - len(df.columns)} time features")
        return df_features
    
    def create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create transaction velocity features per user.
        
        Args:
            df: Dataframe with user_id and purchase_time
            
        Returns:
            Dataframe with velocity features
        """
        logger.info("Creating transaction velocity features...")
        
        df_features = df.copy()
        df_features = df_features.sort_values(['user_id', 'purchase_time'])
        
        # Calculate transactions in different time windows
        windows = [1, 6, 12, 24]  # hours
        
        for window in windows:
            feature_name = f'transactions_last_{window}h'
            self.transaction_velocity_features.append(feature_name)
            
            # Group by user and calculate rolling count
            df_features[feature_name] = df_features.groupby('user_id').apply(
                lambda x: x.rolling(window=f'{window}H', on='purchase_time')['purchase_value'].count()
            ).reset_index(level=0, drop=True) - 1  # Subtract current transaction
            
            # Fill NaN with 0
            df_features[feature_name] = df_features[feature_name].fillna(0)
        
        # Calculate average transaction amount over time
        for window in [6, 12, 24]:
            feature_name = f'avg_amount_last_{window}h'
            
            df_features[feature_name] = df_features.groupby('user_id').apply(
                lambda x: x.rolling(window=f'{window}H', on='purchase_time')['purchase_value'].mean()
            ).reset_index(level=0, drop=True)
            
            # Fill NaN with current transaction amount
            mask = df_features[feature_name].isna()
            df_features.loc[mask, feature_name] = df_features.loc[mask, 'purchase_value']
        
        logger.info(f"Created {len(windows)*2} velocity features")
        return df_features
    
    def create_user_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create user behavior patterns for fraud detection.
        
        Args:
            df: Dataframe with user transactions
            
        Returns:
            Dataframe with user behavior features
        """
        logger.info("Creating user behavior features...")
        
        df_features = df.copy()
        
        # Time since signup
        df_features['time_since_signup_hours'] = (
            df_features['purchase_time'] - df_features['signup_time']
        ).dt.total_seconds() / 3600
        
        # Quick transaction after signup (risk indicator)
        df_features['is_quick_transaction'] = (
            df_features['time_since_signup_hours'] < 1
        ).astype(int)
        
        # Very quick transaction (< 10 minutes)
        df_features['is_very_quick'] = (
            df_features['time_since_signup_hours'] < (10/60)
        ).astype(int)
        
        # User statistics
        user_stats = df_features.groupby('user_id').agg({
            'purchase_value': ['count', 'mean', 'std', 'max'],
            'time_since_signup_hours': 'first'
        }).round(2)
        
        user_stats.columns = ['user_transaction_count', 'user_avg_amount', 
                              'user_std_amount', 'user_max_amount', 
                              'user_time_since_signup']
        
        df_features = df_features.merge(user_stats, on='user_id', how='left')
        
        # Amount deviation from user's average
        df_features['amount_deviation_ratio'] = (
            df_features['purchase_value'] / df_features['user_avg_amount']
        )
        df_features['amount_deviation_ratio'] = df_features['amount_deviation_ratio'].replace([np.inf, -np.inf], 1)
        
        # High amount deviation (potential fraud)
        df_features['is_high_deviation'] = (
            df_features['amount_deviation_ratio'] > 3
        ).astype(int)
        
        logger.info(f"Created user behavior features")
        return df_features
    
    def create_device_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create device-based features for fraud detection.
        
        Args:
            df: Dataframe with device_id column
            
        Returns:
            Dataframe with device features
        """
        logger.info("Creating device features...")
        
        df_features = df.copy()
        
        # Device frequency (how many transactions per device)
        device_counts = df_features['device_id'].value_counts().to_dict()
        df_features['device_frequency'] = df_features['device_id'].map(device_counts)
        
        # Device risk score (higher frequency devices are more suspicious)
        df_features['device_risk_score'] = df_features['device_frequency'] / df_features['device_frequency'].max()
        
        # Multiple users per device (suspicious)
        device_user_counts = df_features.groupby('device_id')['user_id'].nunique().to_dict()
        df_features['users_per_device'] = df_features['device_id'].map(device_user_counts)
        df_features['is_shared_device'] = (df_features['users_per_device'] > 1).astype(int)
        
        return df_features
    
    def create_categorical_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create risk scores from categorical variables.
        
        Args:
            df: Dataframe with categorical columns
            
        Returns:
            Dataframe with risk features
        """
        logger.info("Creating categorical risk features...")
        
        df_features = df.copy()
        
        # Browser risk mapping (based on historical fraud rates)
        browser_risk = {
            'Chrome': 0.3, 'Firefox': 0.4, 'Safari': 0.35,
            'Edge': 0.5, 'Opera': 0.6, 'IE': 0.7
        }
        df_features['browser_risk'] = df_features['browser'].map(browser_risk).fillna(0.5)
        
        # Source risk mapping
        source_risk = {
            'SEO': 0.3, 'Ads': 0.6, 'Direct': 0.4, 
            'Email': 0.5, 'Social': 0.7, 'Referral': 0.4
        }
        df_features['source_risk'] = df_features['source'].map(source_risk).fillna(0.5)
        
        # Country risk (to be updated with actual fraud rates)
        country_risk_default = {'US': 0.3, 'UK': 0.3, 'Unknown': 0.8}
        df_features['country_risk'] = df_features.get('country', 'Unknown').map(
            country_risk_default
        ).fillna(0.6)
        
        return df_features
    
    def process_ecommerce_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline for e-commerce data.
        
        Args:
            df: Cleaned e-commerce dataframe
            
        Returns:
            Dataframe with all engineered features
        """
        logger.info("Running complete feature engineering pipeline...")
        
        # Create all features
        df_fe = self.create_time_features(df, 'purchase_time')
        df_fe = self.create_velocity_features(df_fe)
        df_fe = self.create_user_behavior_features(df_fe)
        df_fe = self.create_device_features(df_fe)
        df_fe = self.create_categorical_risk_features(df_fe)
        
        # Drop columns that are not needed for modeling
        cols_to_drop = ['signup_time', 'purchase_time', 'user_id', 'device_id', 
                       'ip_address', 'ip_int']
        cols_to_drop = [col for col in cols_to_drop if col in df_fe.columns]
        df_fe = df_fe.drop(columns=cols_to_drop)
        
        # Handle infinite values
        df_fe = df_fe.replace([np.inf, -np.inf], np.nan)
        
        # Fill remaining NaN with median
        for col in df_fe.columns:
            if df_fe[col].isnull().any():
                df_fe[col].fillna(df_fe[col].median(), inplace=True)
        
        logger.info(f"Final feature set shape: {df_fe.shape}")
        logger.info(f"Features created: {list(df_fe.columns)}")
        
        return df_fe
    
    def process_creditcard_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process credit card data (already has PCA features).
        Just scale the Amount feature and add time-based features.
        
        Args:
            df: Cleaned credit card dataframe
            
        Returns:
            Processed dataframe
        """
        logger.info("Processing credit card data...")
        
        df_processed = df.copy()
        
        # Time-based features from the Time column
        df_processed['hour_of_day'] = df_processed['Time'] % (24 * 3600) / 3600
        df_processed['day_of_week'] = (df_processed['Time'] // (24 * 3600)) % 7
        df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
        df_processed['is_late_night'] = ((df_processed['hour_of_day'] >= 0) & 
                                         (df_processed['hour_of_day'] <= 5)).astype(int)
        
        # Amount scaling (will be scaled in preprocessing)
        # Log transform for amount to handle skewness
        df_processed['Amount_log'] = np.log1p(df_processed['Amount'])
        
        # Amount category features
        df_processed['amount_category'] = pd.cut(
            df_processed['Amount'], 
            bins=[0, 50, 100, 500, 1000, float('inf')],
            labels=['micro', 'small', 'medium', 'large', 'huge']
        )
        
        logger.info(f"Processed credit card data shape: {df_processed.shape}")
        return df_processed