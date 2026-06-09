"""
Data preprocessing module for fraud detection system.
Handles data cleaning, type conversion, and initial preprocessing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Main data preprocessing class for fraud detection datasets."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def clean_ecommerce_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean e-commerce transactions dataset.
        
        Args:
            df: Raw e-commerce dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning e-commerce data...")
        
        # Make a copy to avoid warnings
        df_clean = df.copy()
        
        # Check for missing values
        logger.info(f"Missing values before cleaning: {df_clean.isnull().sum().sum()}")
        
        # Handle missing values
        # For categorical, fill with mode; for numerical, fill with median
        categorical_cols = ['source', 'browser', 'sex']
        for col in categorical_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        numerical_cols = ['purchase_value', 'age']
        for col in numerical_cols:
            if df_clean[col].isnull().any():
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Remove duplicates based on all columns except potentially class
        initial_len = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        logger.info(f"Removed {initial_len - len(df_clean)} duplicate rows")
        
        # Convert timestamp columns to datetime
        df_clean['signup_time'] = pd.to_datetime(df_clean['signup_time'])
        df_clean['purchase_time'] = pd.to_datetime(df_clean['purchase_time'])
        
        # Ensure age is within reasonable bounds (10-100)
        df_clean = df_clean[(df_clean['age'] >= 10) & (df_clean['age'] <= 100)]
        
        # Ensure purchase_value is positive
        df_clean = df_clean[df_clean['purchase_value'] > 0]
        
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def clean_creditcard_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean credit card transactions dataset.
        
        Args:
            df: Raw credit card dataframe
            
        Returns:
            Cleaned dataframe
        """
        logger.info("Cleaning credit card data...")
        
        df_clean = df.copy()
        
        # Check for missing values (creditcard data typically has none)
        logger.info(f"Missing values: {df_clean.isnull().sum().sum()}")
        
        # Remove duplicates
        initial_len = len(df_clean)
        df_clean.drop_duplicates(inplace=True)
        logger.info(f"Removed {initial_len - len(df_clean)} duplicate rows")
        
        # Remove negative Amount values if any
        df_clean = df_clean[df_clean['Amount'] >= 0]
        
        # Check for infinite values in V1-V28
        v_cols = [f'V{i}' for i in range(1, 29)]
        for col in v_cols:
            df_clean = df_clean[np.isfinite(df_clean[col])]
        
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        return df_clean
    
    def ip_to_int(self, ip_str: str) -> int:
        """
        Convert IP address string to integer for range lookup.
        
        Args:
            ip_str: IP address string
            
        Returns:
            Integer representation of IP
        """
        try:
            parts = ip_str.split('.')
            return (int(parts[0]) << 24) + (int(parts[1]) << 16) + \
                   (int(parts[2]) << 8) + int(parts[3])
        except:
            return None
    
    def enrich_with_country(self, fraud_df: pd.DataFrame, ip_df: pd.DataFrame) -> pd.DataFrame:
        """
        Enrich e-commerce data with country information from IP ranges.
        
        Args:
            fraud_df: E-commerce dataframe
            ip_df: IP to country mapping dataframe
            
        Returns:
            Enriched dataframe with country column
        """
        logger.info("Enriching data with country information...")
        
        df_enriched = fraud_df.copy()
        
        # Convert IP addresses to integer
        df_enriched['ip_int'] = df_enriched['ip_address'].apply(self.ip_to_int)
        
        # Convert IP ranges in ip_df to integers
        ip_df['lower_int'] = ip_df['lower_bound_ip_address'].apply(self.ip_to_int)
        ip_df['upper_int'] = ip_df['upper_bound_ip_address'].apply(self.ip_to_int)
        
        # Perform range-based lookup
        countries = []
        for ip in df_enriched['ip_int']:
            if pd.isna(ip):
                countries.append('Unknown')
                continue
            
            # Find matching country
            match = ip_df[(ip_df['lower_int'] <= ip) & (ip_df['upper_int'] >= ip)]
            if len(match) > 0:
                countries.append(match.iloc[0]['country'])
            else:
                countries.append('Unknown')
        
        df_enriched['country'] = countries
        
        # Analyze fraud by country
        fraud_by_country = df_enriched[df_enriched['class'] == 1]['country'].value_counts()
        logger.info(f"Top 5 countries with fraud: {fraud_by_country.head(5).to_dict()}")
        
        return df_enriched
    
    def prepare_train_test_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        test_size: float = 0.2,
        use_smote: bool = True
    ) -> Tuple:
        """
        Split data and apply SMOTE for handling class imbalance.
        
        Args:
            X: Features dataframe
            y: Target series
            test_size: Proportion for test set
            use_smote: Whether to apply SMOTE on training data
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Splitting data into train/test sets...")
        
        # First split without SMOTE
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        logger.info(f"Train class distribution:\n{y_train.value_counts(normalize=True)}")
        
        if use_smote:
            logger.info("Applying SMOTE to training data...")
            smote = SMOTE(random_state=self.random_state)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            
            logger.info(f"After SMOTE - Train shape: {X_train_resampled.shape}")
            logger.info(f"After SMOTE - Class distribution:\n{y_train_resampled.value_counts(normalize=True)}")
            
            return X_train_resampled, X_test, y_train_resampled, y_test
        
        return X_train, X_test, y_train, y_test