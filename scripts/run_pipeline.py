"""
Main pipeline script to run the entire fraud detection system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import FraudDetector
from src.utils import plot_class_distribution, plot_confusion_matrix

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load all required datasets."""
    logger.info("Loading data...")
    
    # Load e-commerce data
    fraud_data = pd.read_csv('data/raw/Fraud_Data.csv')
    ip_data = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    credit_data = pd.read_csv('data/raw/creditcard.csv')
    
    logger.info(f"E-commerce data shape: {fraud_data.shape}")
    logger.info(f"IP mapping data shape: {ip_data.shape}")
    logger.info(f"Credit card data shape: {credit_data.shape}")
    
    return fraud_data, ip_data, credit_data


def preprocess_ecommerce_data(fraud_df, ip_df):
    """Complete preprocessing pipeline for e-commerce data."""
    logger.info("\n=== Processing E-commerce Data ===")
    
    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()
    
    # Clean data
    df_clean = preprocessor.clean_ecommerce_data(fraud_df)
    
    # Enrich with country info
    df_enriched = preprocessor.enrich_with_country(df_clean, ip_df)
    
    # Create features
    df_features = engineer.process_ecommerce_data(df_enriched)
    
    # Separate features and target
    X = df_features.drop('class', axis=1)
    y = df_features['class']
    
    logger.info(f"Final features: {X.shape[1]}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def preprocess_creditcard_data(credit_df):
    """Complete preprocessing pipeline for credit card data."""
    logger.info("\n=== Processing Credit Card Data ===")
    
    preprocessor = DataPreprocessor()
    engineer = FeatureEngineer()
    
    # Clean data
    df_clean = preprocessor.clean_creditcard_data(credit_df)
    
    # Process features
    df_processed = engineer.process_creditcard_data(df_clean)
    
    # Separate features and target
    X = df_processed.drop('Class', axis=1)
    y = df_processed['Class']
    
    # One-hot encode amount_category if exists
    if 'amount_category' in X.columns:
        X = pd.get_dummies(X, columns=['amount_category'], drop_first=True)
    
    logger.info(f"Final features: {X.shape[1]}")
    logger.info(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def scale_features(X_train, X_test, scaler_type='standard'):
    """Scale numerical features."""
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    # Identify numerical columns
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns
    
    # Scale
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Save scaler
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return X_train_scaled, X_test_scaled, scaler


def main():
    """Main pipeline execution."""
    logger.info("Starting Fraud Detection Pipeline...")
    
    # Create necessary directories
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Load data
    fraud_data, ip_data, credit_data = load_data()
    
    # Process e-commerce data
    X_ecom, y_ecom = preprocess_ecommerce_data(fraud_data, ip_data)
    
    # Process credit card data
    X_credit, y_credit = preprocess_creditcard_data(credit_data)
    
    # Split and prepare data for e-commerce
    preprocessor = DataPreprocessor()
    X_train_ecom, X_test_ecom, y_train_ecom, y_test_ecom = preprocessor.prepare_train_test_split(
        X_ecom, y_ecom, test_size=0.2, use_smote=True
    )
    
    # Scale features
    X_train_ecom, X_test_ecom, scaler_ecom = scale_features(X_train_ecom, X_test_ecom)
    
    # Split and prepare data for credit card
    X_train_credit, X_test_credit, y_train_credit, y_test_credit = preprocessor.prepare_train_test_split(
        X_credit, y_credit, test_size=0.2, use_smote=True
    )
    
    # Scale features
    X_train_credit, X_test_credit, scaler_credit = scale_features(X_train_credit, X_test_credit)
    
    # Train models for e-commerce
    logger.info("\n=== Training Models for E-commerce ===")
    detector_ecom = FraudDetector()
    results_ecom = detector_ecom.train_all_models(
        X_train_ecom, y_train_ecom, X_test_ecom, y_test_ecom
    )
    
    # Train models for credit card
    logger.info("\n=== Training Models for Credit Card ===")
    detector_credit = FraudDetector()
    results_credit = detector_credit.train_all_models(
        X_train_credit, y_train_credit, X_test_credit, y_test_credit
    )
    
    # Save results
    results_ecom.to_csv('reports/ecommerce_model_results.csv', index=False)
    results_credit.to_csv('reports/creditcard_model_results.csv', index=False)
    
    # Print summary
    logger.info("\n=== Summary ===")
    logger.info("\nE-commerce Best Model:")
    logger.info(f"Model: {results_ecom.iloc[0]['model_name']}")
    logger.info(f"F1 Score: {results_ecom.iloc[0]['f1_score']:.4f}")
    logger.info(f"Cost Savings: ${results_ecom.iloc[0]['cost_savings']:.2f}")
    
    logger.info("\nCredit Card Best Model:")
    logger.info(f"Model: {results_credit.iloc[0]['model_name']}")
    logger.info(f"F1 Score: {results_credit.iloc[0]['f1_score']:.4f}")
    logger.info(f"Cost Savings: ${results_credit.iloc[0]['cost_savings']:.2f}")
    
    logger.info("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()