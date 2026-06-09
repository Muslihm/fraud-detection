# src/data_cleaning_report.py
"""
Data Cleaning Report for Fraud Detection System
"""

import pandas as pd
import numpy as np
from datetime import datetime

class DataCleaningReporter:
    """Generate comprehensive data cleaning reports."""
    
    def generate_ecommerce_report(self, raw_df, cleaned_df):
        """Generate cleaning report for e-commerce data."""
        
        report = {
            'dataset': 'E-commerce Transactions',
            'cleaning_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'initial_shape': raw_df.shape,
            'final_shape': cleaned_df.shape,
            'rows_removed': len(raw_df) - len(cleaned_df),
            'removal_percentage': ((len(raw_df) - len(cleaned_df)) / len(raw_df)) * 100,
            'missing_values_handled': {},
            'duplicates_removed': 0,
            'outliers_handled': {},
            'data_types_corrected': []
        }
        
        # Missing values analysis
        for col in raw_df.columns:
            missing_before = raw_df[col].isnull().sum()
            missing_after = cleaned_df[col].isnull().sum() if col in cleaned_df.columns else 0
            if missing_before > 0:
                report['missing_values_handled'][col] = {
                    'before': missing_before,
                    'after': missing_after,
                    'method': 'mode_imputation' if col in ['source', 'browser', 'sex'] else 'median_imputation'
                }
        
        # Duplicates
        report['duplicates_removed'] = len(raw_df) - len(raw_df.drop_duplicates())
        
        # Outliers (age and purchase_value)
        if 'age' in cleaned_df.columns:
            age_outliers = ((cleaned_df['age'] < 10) | (cleaned_df['age'] > 100)).sum()
            report['outliers_handled']['age'] = {
                'count': age_outliers,
                'method': 'removed (age < 10 or > 100)'
            }
        
        if 'purchase_value' in cleaned_df.columns:
            value_outliers = (cleaned_df['purchase_value'] <= 0).sum()
            report['outliers_handled']['purchase_value'] = {
                'count': value_outliers,
                'method': 'removed (non-positive values)'
            }
        
        return report
    
    def generate_creditcard_report(self, raw_df, cleaned_df):
        """Generate cleaning report for credit card data."""
        
        report = {
            'dataset': 'Credit Card Transactions',
            'cleaning_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'initial_shape': raw_df.shape,
            'final_shape': cleaned_df.shape,
            'rows_removed': len(raw_df) - len(cleaned_df),
            'removal_percentage': ((len(raw_df) - len(cleaned_df)) / len(raw_df)) * 100,
            'missing_values': raw_df.isnull().sum().sum(),
            'duplicates_removed': len(raw_df) - len(raw_df.drop_duplicates()),
            'infinite_values_handled': 0,
            'negative_amounts_removed': 0
        }
        
        # Check for infinite values in V1-V28
        v_cols = [f'V{i}' for i in range(1, 29)]
        infinite_count = 0
        for col in v_cols:
            if col in raw_df.columns:
                infinite_count += np.isinf(raw_df[col]).sum()
        
        report['infinite_values_handled'] = infinite_count
        
        # Negative amounts
        if 'Amount' in raw_df.columns:
            negative_amounts = (raw_df['Amount'] < 0).sum()
            report['negative_amounts_removed'] = negative_amounts
        
        return report

# Generate and save reports
def generate_all_cleaning_reports():
    """Generate all data cleaning reports."""
    reporter = DataCleaningReporter()
    
    # Load data
    raw_ecom = pd.read_csv('data/raw/Fraud_Data.csv')
    raw_credit = pd.read_csv('data/raw/creditcard.csv')
    
    # Note: In real implementation, you'd load cleaned data from processed folder
    # For now, we'll create sample reports
    
    ecom_report = reporter.generate_ecommerce_report(raw_ecom, raw_ecom)  # Replace with cleaned
    credit_report = reporter.generate_creditcard_report(raw_credit, raw_credit)  # Replace with cleaned
    
    # Save reports
    import json
    with open('reports/ecommerce_cleaning_report.json', 'w') as f:
        json.dump(ecom_report, f, indent=2)
    
    with open('reports/creditcard_cleaning_report.json', 'w') as f:
        json.dump(credit_report, f, indent=2)
    
    return ecom_report, credit_report