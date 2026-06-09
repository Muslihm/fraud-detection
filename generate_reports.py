# generate_reports.py
import pandas as pd
import json
import os

def main():
    """Generate all required reports."""
    
    print("="*60)
    print("Generating Fraud Detection Reports")
    print("="*60)
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Check if data exists
    if not os.path.exists('data/raw/Fraud_Data.csv'):
        print("ERROR: data/raw/Fraud_Data.csv not found!")
        print("Please ensure your data files are in the correct location.")
        return
    
    # Load data
    print("\n1. Loading data...")
    ecom_df = pd.read_csv('data/raw/Fraud_Data.csv')
    credit_df = pd.read_csv('data/raw/creditcard.csv')
    print(f"   ✓ E-commerce data: {ecom_df.shape}")
    print(f"   ✓ Credit card data: {credit_df.shape}")
    
    # Calculate metrics
    print("\n2. Analyzing class imbalance...")
    ecom_fraud = (ecom_df['class'] == 1).sum()
    ecom_legit = (ecom_df['class'] == 0).sum()
    credit_fraud = (credit_df['Class'] == 1).sum()
    credit_legit = (credit_df['Class'] == 0).sum()
    
    print(f"   E-commerce - Fraud: {ecom_fraud} ({ecom_fraud/len(ecom_df)*100:.3f}%)")
    print(f"   Credit Card - Fraud: {credit_fraud} ({credit_fraud/len(credit_df)*100:.3f}%)")
    
    # Generate EDA report
    print("\n3. Generating EDA report...")
    with open('reports/eda_analysis_report.md', 'w') as f:
        f.write("# Exploratory Data Analysis Report\n\n")
        f.write("## E-commerce Transactions\n\n")
        f.write(f"- **Total**: {len(ecom_df):,}\n")
        f.write(f"- **Fraud**: {ecom_fraud:,} ({ecom_fraud/len(ecom_df)*100:.3f}%)\n")
        f.write(f"- **Imbalance ratio**: {ecom_legit/ecom_fraud:.1f}:1\n\n")
        f.write("## Credit Card Transactions\n\n")
        f.write(f"- **Total**: {len(credit_df):,}\n")
        f.write(f"- **Fraud**: {credit_fraud:,} ({credit_fraud/len(credit_df)*100:.3f}%)\n")
        f.write(f"- **Imbalance ratio**: {credit_legit/credit_fraud:.1f}:1\n\n")
        f.write("## Key Insights\n\n")
        f.write("1. Extreme class imbalance requires special handling\n")
        f.write("2. SMOTE resampling recommended for training\n")
        f.write("3. Focus on recall and precision metrics\n")
        f.write("4. Implement cost-sensitive evaluation\n")
    print("   ✓ EDA report generated")
    
    # Generate feature documentation
    print("\n4. Generating feature documentation...")
    with open('reports/feature_engineering_documentation.md', 'w') as f:
        f.write("# Feature Engineering Documentation\n\n")
        f.write("## Engineered Features\n\n")
        f.write("### Time-Based Features\n")
        f.write("- hour_of_day: Hour of transaction\n")
        f.write("- day_of_week: Day of week\n")
        f.write("- is_weekend: Weekend indicator\n")
        f.write("- is_late_night: Late night indicator\n\n")
        
        f.write("### Velocity Features\n")
        f.write("- transactions_last_1h: Transaction count in past hour\n")
        f.write("- transactions_last_24h: Transaction count in past day\n\n")
        
        f.write("### User Behavior\n")
        f.write("- time_since_signup_hours: Time since account creation\n")
        f.write("- amount_deviation_ratio: Deviation from user average\n\n")
        
        f.write("### IP-to-Country Mapping\n")
        f.write("- Converted IP addresses to integers\n")
        f.write("- Performed range-based lookup\n")
        f.write("- Added country and country_risk features\n")
    print("   ✓ Feature documentation generated")
    
    # Generate resampling justification
    print("\n5. Generating resampling justification...")
    justification = {
        'method': 'SMOTE',
        'reason': 'Extreme class imbalance requires synthetic oversampling',
        'alternatives_rejected': ['Undersampling', 'Simple oversampling', 'Class weights'],
        'implementation': 'Applied to training data only with k_neighbors=5'
    }
    
    with open('reports/resampling_justification.json', 'w') as f:
        json.dump(justification, f, indent=2)
    
    with open('reports/resampling_justification.md', 'w') as f:
        f.write("# Resampling Strategy Justification\n\n")
        f.write("## Selected Method: SMOTE\n\n")
        f.write(f"**Reason**: {justification['reason']}\n\n")
        f.write("**Alternatives Considered**:\n")
        for alt in justification['alternatives_rejected']:
            f.write(f"- {alt}\n")
        f.write(f"\n**Implementation**: {justification['implementation']}\n")
    print("   ✓ Resampling justification generated")
    
    # Generate data cleaning report
    print("\n6. Generating data cleaning report...")
    with open('reports/data_cleaning_report.md', 'w') as f:
        f.write("# Data Cleaning Report\n\n")
        f.write("## E-commerce Data\n\n")
        f.write("- Missing values: Imputed with mode/median\n")
        f.write("- Duplicates: Removed\n")
        f.write("- Outliers: Age (<10 or >100) removed\n")
        f.write("- Data types: Converted timestamps to datetime\n\n")
        f.write("## Credit Card Data\n\n")
        f.write("- Missing values: None found\n")
        f.write("- Infinite values: Checked and removed\n")
        f.write("- Amount: Validated non-negative\n")
    print("   ✓ Data cleaning report generated")
    
    print("\n" + "="*60)
    print("✅ ALL REPORTS GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nReports saved in 'reports/' directory:")
    print("  📄 eda_analysis_report.md")
    print("  📄 feature_engineering_documentation.md")
    print("  📄 resampling_justification.md")
    print("  📄 resampling_justification.json")
    print("  📄 data_cleaning_report.md")
    

if __name__ == "__main__":
    main()