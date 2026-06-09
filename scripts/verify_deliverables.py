# scripts/verify_deliverables.py
"""
Verify that all required deliverables are generated
"""

import os
import json

def verify_deliverables():
    """Check all deliverables exist and are valid."""
    
    deliverables = {
        'Cleaned Datasets': [
            'data/processed/ecommerce_cleaned.csv',
            'data/processed/creditcard_cleaned.csv'
        ],
        'EDA Report': [
            'reports/eda_analysis_report.json',
            'reports/eda_analysis_report.md'
        ],
        'Feature Engineering Documentation': [
            'reports/feature_engineering_documentation.json',
            'reports/feature_engineering_documentation.md'
        ],
        'Resampling Justification': [
            'reports/resampling_justification.json',
            'reports/resampling_justification.md'
        ],
        'Data Cleaning Reports': [
            'reports/ecommerce_cleaning_report.json',
            'reports/creditcard_cleaning_report.json'
        ]
    }
    
    print("="*60)
    print("DELIVERABLES VERIFICATION")
    print("="*60)
    
    all_good = True
    
    for category, files in deliverables.items():
        print(f"\n{category}:")
        for filepath in files:
            exists = os.path.exists(filepath)
            status = "✓" if exists else "✗"
            print(f"  {status} {filepath}")
            if not exists:
                all_good = False
    
    print("\n" + "="*60)
    if all_good:
        print("✅ All deliverables are present!")
    else:
        print("❌ Some deliverables are missing. Run generate_all_deliverables()")
    print("="*60)
    
    return all_good

if __name__ == "__main__":
    verify_deliverables()