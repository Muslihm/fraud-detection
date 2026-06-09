# notebooks/eda_report.py
"""
Exploratory Data Analysis Report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class EDAReporter:
    """Generate comprehensive EDA reports."""
    
    def __init__(self):
        self.report = {}
    
    def analyze_class_imbalance(self, y, dataset_name):
        """Analyze and document class imbalance."""
        fraud_count = (y == 1).sum()
        legit_count = (y == 0).sum()
        fraud_pct = (fraud_count / len(y)) * 100
        
        analysis = {
            'dataset': dataset_name,
            'total_transactions': len(y),
            'legitimate_count': legit_count,
            'fraud_count': fraud_count,
            'fraud_percentage': fraud_pct,
            'imbalance_ratio': legit_count / fraud_count if fraud_count > 0 else float('inf'),
            'severity': 'extreme' if fraud_pct < 0.1 else 'high' if fraud_pct < 1 else 'moderate',
            'recommendation': 'Use SMOTE, class weights, or ensemble methods'
        }
        
        return analysis
    
    def analyze_numeric_features(self, df, feature_cols, target_col, dataset_name):
        """Analyze numeric features distribution."""
        analysis = {}
        
        for col in feature_cols:
            if col in df.columns:
                fraud_data = df[df[target_col] == 1][col]
                legit_data = df[df[target_col] == 0][col]
                
                # Statistical tests
                t_stat, p_value = stats.ttest_ind(fraud_data, legit_data, equal_var=False)
                
                analysis[col] = {
                    'mean_fraud': fraud_data.mean(),
                    'mean_legit': legit_data.mean(),
                    'std_fraud': fraud_data.std(),
                    'std_legit': legit_data.std(),
                    'median_fraud': fraud_data.median(),
                    'median_legit': legit_data.median(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant_difference': p_value < 0.05,
                    'skewness_fraud': fraud_data.skew(),
                    'skewness_legit': legit_data.skew()
                }
        
        return analysis
    
    def generate_full_eda_report(self, ecom_df, credit_df):
        """Generate complete EDA report for both datasets."""
        
        report = {
            'ecommerce_analysis': {},
            'creditcard_analysis': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # E-commerce analysis
        report['ecommerce_analysis']['class_imbalance'] = self.analyze_class_imbalance(
            ecom_df['class'], 'E-commerce'
        )
        
        # Time-based patterns
        if 'purchase_time' in ecom_df.columns:
            ecom_df['hour'] = pd.to_datetime(ecom_df['purchase_time']).dt.hour
            fraud_by_hour = ecom_df[ecom_df['class'] == 1]['hour'].value_counts().sort_index()
            legit_by_hour = ecom_df[ecom_df['class'] == 0]['hour'].value_counts().sort_index()
            
            report['ecommerce_analysis']['time_patterns'] = {
                'peak_fraud_hours': fraud_by_hour.head(3).to_dict(),
                'peak_legit_hours': legit_by_hour.head(3).to_dict(),
                'late_night_fraud_rate': fraud_by_hour[[0,1,2,3,4,5]].sum() / len(ecom_df[ecom_df['class'] == 1]) * 100
            }
        
        # Credit card analysis
        report['creditcard_analysis']['class_imbalance'] = self.analyze_class_imbalance(
            credit_df['Class'], 'Credit Card'
        )
        
        # Amount analysis
        fraud_amounts = credit_df[credit_df['Class'] == 1]['Amount']
        legit_amounts = credit_df[credit_df['Class'] == 0]['Amount']
        
        report['creditcard_analysis']['amount_analysis'] = {
            'fraud_mean_amount': fraud_amounts.mean(),
            'fraud_median_amount': fraud_amounts.median(),
            'fraud_amount_std': fraud_amounts.std(),
            'legit_mean_amount': legit_amounts.mean(),
            'fraud_amount_percentiles': fraud_amounts.quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
        }
        
        # Key findings
        report['key_findings'] = [
            f"E-commerce fraud rate: {report['ecommerce_analysis']['class_imbalance']['fraud_percentage']:.3f}%",
            f"Credit card fraud rate: {report['creditcard_analysis']['class_imbalance']['fraud_percentage']:.3f}%",
            f"E-commerce imbalance ratio: {report['ecommerce_analysis']['class_imbalance']['imbalance_ratio']:.1f}:1",
            f"Credit card imbalance ratio: {report['creditcard_analysis']['class_imbalance']['imbalance_ratio']:.1f}:1"
        ]
        
        # Recommendations
        report['recommendations'] = [
            "Use SMOTE oversampling for training data only",
            "Implement cost-sensitive learning with higher penalty for false negatives",
            "Use PR-AUC instead of ROC-AUC due to extreme imbalance",
            "Consider anomaly detection methods as baseline",
            "Implement threshold tuning based on business costs"
        ]
        
        return report

# Generate EDA report
def generate_eda_report():
    """Generate and save EDA report."""
    # Load data
    ecom_df = pd.read_csv('data/raw/Fraud_Data.csv')
    credit_df = pd.read_csv('data/raw/creditcard.csv')
    
    reporter = EDAReporter()
    report = reporter.generate_full_eda_report(ecom_df, credit_df)
    
    # Save as JSON
    import json
    with open('reports/eda_analysis_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Also save as markdown for readability
    with open('reports/eda_analysis_report.md', 'w') as f:
        f.write("# Exploratory Data Analysis Report\n\n")
        f.write("## E-commerce Fraud Analysis\n\n")
        f.write(f"- Fraud Rate: {report['ecommerce_analysis']['class_imbalance']['fraud_percentage']:.3f}%\n")
        f.write(f"- Imbalance Ratio: {report['ecommerce_analysis']['class_imbalance']['imbalance_ratio']:.1f}:1\n\n")
        
        f.write("## Credit Card Fraud Analysis\n\n")
        f.write(f"- Fraud Rate: {report['creditcard_analysis']['class_imbalance']['fraud_percentage']:.3f}%\n")
        f.write(f"- Imbalance Ratio: {report['creditcard_analysis']['class_imbalance']['imbalance_ratio']:.1f}:1\n\n")
        
        f.write("## Key Findings\n\n")
        for finding in report['key_findings']:
            f.write(f"- {finding}\n")
        
        f.write("\n## Recommendations\n\n")
        for rec in report['recommendations']:
            f.write(f"- {rec}\n")
    
    return report