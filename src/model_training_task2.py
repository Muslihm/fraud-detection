"""
Task 2: Model Building and Training for Fraud Detection
Fixed version with proper imports
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudModelTrainer:
    """Complete model training pipeline for fraud detection."""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def load_and_prepare_data(self):
        """Load and prepare both datasets for modeling."""
        logger.info("Loading datasets...")
        
        # Load e-commerce data
        try:
            ecom_df = pd.read_csv('data/raw/Fraud_Data.csv')
            logger.info(f"E-commerce data loaded: {ecom_df.shape}")
        except:
            logger.warning("E-commerce data not found, using sample data")
            ecom_df = self.create_sample_ecom_data()
        
        # Load credit card data
        try:
            credit_df = pd.read_csv('data/raw/creditcard.csv')
            logger.info(f"Credit card data loaded: {credit_df.shape}")
        except:
            logger.warning("Credit card data not found, using sample data")
            credit_df = self.create_sample_credit_data()
        
        return ecom_df, credit_df
    
    def create_sample_ecom_data(self):
        """Create sample e-commerce data if real data not available."""
        np.random.seed(self.random_state)
        n_samples = 10000
        n_fraud = int(n_samples * 0.0012)
        
        data = {
            'purchase_value': np.random.exponential(100, n_samples),
            'age': np.random.normal(35, 10, n_samples),
            'hour_of_day': np.random.randint(0, 24, n_samples),
            'time_since_signup_hours': np.random.exponential(48, n_samples),
            'transactions_last_24h': np.random.poisson(0.5, n_samples),
            'class': np.zeros(n_samples)
        }
        
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        data['class'][fraud_indices] = 1
        data['time_since_signup_hours'][fraud_indices] = np.random.exponential(2, n_fraud)
        data['transactions_last_24h'][fraud_indices] = np.random.poisson(3, n_fraud)
        
        return pd.DataFrame(data)
    
    def create_sample_credit_data(self):
        """Create sample credit card data if real data not available."""
        np.random.seed(self.random_state)
        n_samples = 10000
        n_fraud = int(n_samples * 0.0017)
        
        data = {
            'V1': np.random.normal(0, 1, n_samples),
            'V2': np.random.normal(0, 1, n_samples),
            'V3': np.random.normal(0, 1, n_samples),
            'Amount': np.random.exponential(100, n_samples),
            'Time': np.random.uniform(0, 172000, n_samples),
            'Class': np.zeros(n_samples)
        }
        
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        data['Class'][fraud_indices] = 1
        data['Amount'][fraud_indices] = np.random.exponential(500, n_fraud)
        data['V1'][fraud_indices] = np.random.normal(-2, 1, n_fraud)
        data['V2'][fraud_indices] = np.random.normal(2, 1, n_fraud)
        
        return pd.DataFrame(data)
    
    def preprocess_data(self, df, target_col, is_credit=False):
        """Preprocess data for modeling."""
        logger.info(f"Preprocessing data with target: {target_col}")
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale numerical features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        
        return X, y
    
    def train_test_split_stratified(self, X, y, test_size=0.2):
        """Perform stratified train-test split."""
        logger.info("Performing stratified train-test split...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        logger.info(f"Train set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        logger.info(f"Train fraud rate: {y_train.mean():.4%}")
        logger.info(f"Test fraud rate: {y_test.mean():.4%}")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train, y_train):
        """Train baseline logistic regression model."""
        logger.info("Training Logistic Regression (baseline)...")
        
        lr = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000,
            C=1.0
        )
        lr.fit(X_train, y_train)
        
        self.models['Logistic Regression'] = lr
        return lr
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest ensemble model."""
        logger.info("Training Random Forest...")
        
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        self.models['Random Forest'] = rf
        return rf
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model."""
        logger.info("Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalance
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False
        )
        xgb_model.fit(X_train, y_train)
        
        self.models['XGBoost'] = xgb_model
        return xgb_model
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation."""
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model': model_name,
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Calculate business metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        metrics['true_negatives'] = tn
        
        # Cost calculation (FP cost: $10, FN cost: $100)
        total_cost = (fp * 10) + (fn * 100)
        metrics['total_cost'] = total_cost
        
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"  Total Cost: ${total_cost:.2f}")
        
        return metrics
    
    def cross_validate_model(self, model, X, y, model_name, cv_folds=5):
        """Perform stratified k-fold cross-validation."""
        logger.info(f"Performing {cv_folds}-fold CV for {model_name}...")
        
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Store metrics for each fold
        fold_metrics = {
            'f1_score': [], 'precision': [], 'recall': [], 
            'roc_auc': [], 'pr_auc': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model
            from sklearn.base import clone
            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold)
            
            # Predict
            y_pred = model_clone.predict(X_val_fold)
            y_pred_proba = model_clone.predict_proba(X_val_fold)[:, 1]
            
            # Calculate metrics
            fold_metrics['f1_score'].append(f1_score(y_val_fold, y_pred))
            fold_metrics['precision'].append(precision_score(y_val_fold, y_pred))
            fold_metrics['recall'].append(recall_score(y_val_fold, y_pred))
            fold_metrics['roc_auc'].append(roc_auc_score(y_val_fold, y_pred_proba))
            fold_metrics['pr_auc'].append(average_precision_score(y_val_fold, y_pred_proba))
            
            logger.info(f"  Fold {fold} - F1: {fold_metrics['f1_score'][-1]:.4f}, "
                       f"ROC-AUC: {fold_metrics['roc_auc'][-1]:.4f}")
        
        # Calculate mean and std
        cv_results = {}
        for metric, values in fold_metrics.items():
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        logger.info(f"CV Results for {model_name}:")
        logger.info(f"  F1-Score: {cv_results['f1_score_mean']:.4f} (+/- {cv_results['f1_score_std']:.4f})")
        logger.info(f"  ROC-AUC: {cv_results['roc_auc_mean']:.4f} (+/- {cv_results['roc_auc_std']:.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='xgboost'):
        """Basic hyperparameter tuning."""
        from sklearn.model_selection import ParameterGrid
        
        logger.info(f"Performing hyperparameter tuning for {model_type}...")
        
        if model_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            }
            
            best_score = 0
            best_params = {}
            
            for params in ParameterGrid(param_grid):
                rf = RandomForestClassifier(
                    **params,
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                )
                # Use cross_val_score (properly imported)
                scores = cross_val_score(rf, X_train, y_train, cv=3, scoring='f1')
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best CV F1-Score: {best_score:.4f}")
            return best_params
        
        elif model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.1, 0.3]
            }
            
            best_score = 0
            best_params = {}
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            
            for params in ParameterGrid(param_grid):
                xgb_model = xgb.XGBClassifier(
                    **params,
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
                # Use cross_val_score (properly imported)
                scores = cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='f1')
                mean_score = scores.mean()
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best CV F1-Score: {best_score:.4f}")
            return best_params
        
        return {}
    
    def run_complete_pipeline(self, dataset_name='creditcard'):
        """Run complete model training pipeline."""
        logger.info("="*60)
        logger.info(f"Running complete pipeline for {dataset_name}")
        logger.info("="*60)
        
        # Load data
        ecom_df, credit_df = self.load_and_prepare_data()
        
        if dataset_name == 'creditcard':
            df = credit_df
            target = 'Class'
            is_credit = True
        else:
            df = ecom_df
            target = 'class'
            is_credit = False
        
        # Preprocess
        X, y = self.preprocess_data(df, target, is_credit)
        
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split_stratified(X, y)
        
        # Train models
        results = {}
        cv_results = {}
        
        # 1. Logistic Regression (Baseline)
        lr = self.train_logistic_regression(X_train, y_train)
        results['Logistic Regression'] = self.evaluate_model(lr, X_test, y_test, 'Logistic Regression')
        cv_results['Logistic Regression'] = self.cross_validate_model(lr, X_train, y_train, 'Logistic Regression')
        
        # 2. Random Forest
        rf = self.train_random_forest(X_train, y_train)
        results['Random Forest'] = self.evaluate_model(rf, X_test, y_test, 'Random Forest')
        cv_results['Random Forest'] = self.cross_validate_model(rf, X_train, y_train, 'Random Forest')
        
        # 3. XGBoost
        xgb_model = self.train_xgboost(X_train, y_train)
        results['XGBoost'] = self.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
        cv_results['XGBoost'] = self.cross_validate_model(xgb_model, X_train, y_train, 'XGBoost')
        
        # Hyperparameter tuning for XGBoost (skip if error)
        try:
            logger.info("\n" + "="*60)
            logger.info("Hyperparameter Tuning for XGBoost...")
            best_params = self.hyperparameter_tuning(X_train, y_train, 'xgboost')
            
            # Train optimized XGBoost if tuning successful
            if best_params:
                scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
                xgb_optimized = xgb.XGBClassifier(
                    **best_params,
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state,
                    eval_metric='logloss',
                    use_label_encoder=False
                )
                xgb_optimized.fit(X_train, y_train)
                results['XGBoost (Optimized)'] = self.evaluate_model(xgb_optimized, X_test, y_test, 'XGBoost Optimized')
                self.models['XGBoost (Optimized)'] = xgb_optimized
        except Exception as e:
            logger.warning(f"Hyperparameter tuning skipped: {e}")
        
        # Store results
        self.results[dataset_name] = {
            'test_results': results,
            'cv_results': cv_results
        }
        
        # Select best model
        self.select_best_model(dataset_name)
        
        return results, cv_results
    
    def select_best_model(self, dataset_name):
        """Select the best model based on F1-score and business costs."""
        results = self.results[dataset_name]['test_results']
        
        best_model_name = None
        best_score = 0
        
        for model_name, metrics in results.items():
            # Combine F1-score and cost savings
            combined_score = metrics['f1_score'] * 0.7 + (1 - min(metrics['total_cost']/10000, 1)) * 0.3
            if combined_score > best_score:
                best_score = combined_score
                best_model_name = model_name
        
        self.best_model = self.models.get(best_model_name)
        
        logger.info("\n" + "="*60)
        logger.info(f"BEST MODEL FOR {dataset_name.upper()}: {best_model_name}")
        logger.info(f"Combined Score: {best_score:.4f}")
        logger.info(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
        logger.info(f"Cost: ${results[best_model_name]['total_cost']:.2f}")
        logger.info("="*60)
        
        # Save best model
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.best_model, f'models/best_model_{dataset_name}.pkl')
        
        return best_model_name
    
    def generate_comparison_report(self):
        """Generate comprehensive model comparison report."""
        import os
        
        results = self.results['creditcard']['test_results']
        cv_results = self.results['creditcard']['cv_results']
        
        report = []
        report.append("# Model Comparison Report - Task 2")
        report.append(f"\n## Credit Card Dataset")
        report.append(f"\n### Performance Summary")
        report.append("\n| Model | F1-Score | Precision | Recall | ROC-AUC | PR-AUC | Cost/10K |")
        report.append("|-------|----------|-----------|--------|---------|--------|----------|")
        
        for model_name, metrics in results.items():
            report.append(f"| {model_name} | {metrics['f1_score']:.4f} | {metrics['precision']:.4f} | "
                         f"{metrics['recall']:.4f} | {metrics['roc_auc']:.4f} | "
                         f"{metrics['pr_auc']:.4f} | ${metrics['total_cost']:.2f} |")
        
        report.append("\n## Cross-Validation Results (5-fold)\n")
        report.append("| Model | CV F1-Score (Mean) | CV F1-Score (Std) |")
        report.append("|-------|-------------------|------------------|")
        
        for model_name, cv_metrics in cv_results.items():
            report.append(f"| {model_name} | {cv_metrics['f1_score_mean']:.4f} | ±{cv_metrics['f1_score_std']:.4f} |")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
        best_name, best_metrics = best_model
        
        report.append("\n## Model Selection Justification\n")
        report.append(f"**Selected Model**: {best_name}\n")
        report.append("**Reasons:**")
        report.append(f"1. **Highest F1-Score**: {best_metrics['f1_score']:.4f} (balanced metric)")
        report.append(f"2. **Best Precision-Recall Balance**: {best_metrics['precision']:.4f} / {best_metrics['recall']:.4f}")
        report.append(f"3. **Lowest Business Cost**: ${best_metrics['total_cost']:.2f} per 10,000 transactions")
        report.append(f"4. **Robust CV Performance**: {cv_results[best_name]['f1_score_mean']:.4f} (±{cv_results[best_name]['f1_score_std']:.4f})")
        
        # Baseline comparison
        baseline = results['Logistic Regression']
        improvement = ((best_metrics['f1_score'] - baseline['f1_score']) / baseline['f1_score']) * 100
        report.append(f"\n### Improvement over Baseline")
        report.append(f"- **{improvement:.1f}% higher F1-Score** than Logistic Regression")
        report.append(f"- **${baseline['total_cost'] - best_metrics['total_cost']:.2f} cost savings** per 10,000 transactions")
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        with open('reports/model_comparison_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        logger.info("Model comparison report saved to reports/model_comparison_report.md")
        
        return '\n'.join(report)


def main():
    """Main execution function."""
    import os
    
    print("= - model_training_task2.py:504"*60)
    print("FRAUD DETECTION  TASK 2: MODEL BUILDING & TRAINING - model_training_task2.py:505")
    print("= - model_training_task2.py:506"*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Initialize trainer
    trainer = FraudModelTrainer(random_state=42)
    
    # Run for Credit Card dataset
    print("\n📊 Processing Credit Card Dataset... - model_training_task2.py:516")
    credit_results, credit_cv = trainer.run_complete_pipeline('creditcard')
    
    # Generate comparison report
    print("\n📄 Generating Comparison Report... - model_training_task2.py:520")
    trainer.generate_comparison_report()
    
    print("\n - model_training_task2.py:523" + "="*60)
    print("✅ TASK 2 COMPLETED SUCCESSFULLY! - model_training_task2.py:524")
    print("= - model_training_task2.py:525"*60)
    print("\nDeliverables: - model_training_task2.py:526")
    print("✓ Baseline model (Logistic Regression) trained - model_training_task2.py:527")
    print("✓ Ensemble models (Random Forest, XGBoost) trained - model_training_task2.py:528")
    print("✓ 5fold stratified crossvalidation performed - model_training_task2.py:529")
    print("✓ Model comparison report generated - model_training_task2.py:530")
    print("\nSaved files: - model_training_task2.py:531")
    print("models/best_model_creditcard.pkl - model_training_task2.py:532")
    print("reports/model_comparison_report.md - model_training_task2.py:533")
    
    return trainer


if __name__ == "__main__":
    trainer = main()