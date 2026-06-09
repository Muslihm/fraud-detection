"""
Model training and evaluation module for fraud detection.
Includes multiple algorithms and comprehensive evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, matthews_corrcoef
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    IsolationForest
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from typing import Dict, Any, Tuple, List
import joblib
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetector:
    """Main fraud detection model training and evaluation class."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.metrics_history = {}
        
    def get_models(self) -> Dict[str, Any]:
        """
        Define dictionary of models to train.
        
        Returns:
            Dictionary of model instances
        """
        return {
            'LogisticRegression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                scale_pos_weight=10,  # Adjust for imbalance
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            ),
            'CatBoost': CatBoostClassifier(
                iterations=100,
                random_seed=self.random_state,
                verbose=False,
                auto_class_weights='Balanced'
            )
        }
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   costs: Dict[str, float] = None) -> Dict[str, float]:
        """
        Calculate business-specific metrics including cost-based metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            costs: Dictionary with 'false_positive_cost' and 'false_negative_cost'
            
        Returns:
            Dictionary of business metrics
        """
        if costs is None:
            # Default costs (FP: $10, FN: $100)
            costs = {'false_positive_cost': 10, 'false_negative_cost': 100}
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate costs
        total_cost = (fp * costs['false_positive_cost'] + 
                     fn * costs['false_negative_cost'])
        
        # Cost savings compared to baseline (predict all as non-fraud)
        baseline_cost = (y_true.sum() * costs['false_negative_cost'])
        cost_savings = max(0, baseline_cost - total_cost)
        cost_savings_pct = (cost_savings / baseline_cost) * 100 if baseline_cost > 0 else 0
        
        # Additional business metrics
        metrics = {
            'total_cost': total_cost,
            'cost_savings': cost_savings,
            'cost_savings_percentage': cost_savings_pct,
            'fraud_caught_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'precision_at_10_percent_fpr': self.calculate_precision_at_fpr(y_true, y_pred, 0.1)
        }
        
        return metrics
    
    def calculate_precision_at_fpr(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                                  max_fpr: float = 0.1) -> float:
        """
        Calculate precision at a given false positive rate threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            max_fpr: Maximum false positive rate allowed
            
        Returns:
            Precision at specified FPR
        """
        from sklearn.metrics import precision_recall_curve
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # Find threshold that gives FPR <= max_fpr
        fpr = 1 - precision  # Approximate relationship
        mask = fpr <= max_fpr
        
        if np.any(mask):
            return precision[mask][-1]
        else:
            return 0.0
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> Dict[str, Any]:
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = y_pred
        
        # Calculate standard metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # Calculate business metrics
        business_metrics = self.calculate_business_metrics(
            y_test.values if hasattr(y_test, 'values') else y_test,
            y_pred
        )
        metrics.update(business_metrics)
        
        logger.info(f"{model_name} - F1: {metrics['f1_score']:.4f}, "
                   f"ROC-AUC: {metrics['roc_auc']:.4f}, "
                   f"Cost: ${metrics['total_cost']:.2f}")
        
        return metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Train all models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with all model metrics
        """
        logger.info("Training all models...")
        
        models_dict = self.get_models()
        results = []
        
        for name, model in models_dict.items():
            logger.info(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                self.models[name] = model
                
                # Evaluate
                metrics = self.evaluate_model(model, X_test, y_test, name)
                results.append(metrics)
                
                # Save model
                self.save_model(model, name)
                
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                continue
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by F1 score
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        # Select best model
        self.best_model = self.models[results_df.iloc[0]['model_name']]
        logger.info(f"Best model: {results_df.iloc[0]['model_name']} "
                   f"with F1: {results_df.iloc[0]['f1_score']:.4f}")
        
        return results_df
    
    def save_model(self, model: Any, name: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            name: Model name
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"models/{name}_{timestamp}.joblib"
        joblib.dump(model, filename)
        logger.info(f"Saved model to {filename}")
    
    def load_model(self, filepath: str) -> Any:
        """
        Load saved model from disk.
        
        Args:
            filepath: Path to saved model
            
        Returns:
            Loaded model
        """
        return joblib.load(filepath)
    
    def create_ensemble_model(self, models: List[Any], weights: List[float] = None) -> Any:
        """
        Create a weighted ensemble of multiple models.
        
        Args:
            models: List of trained models
            weights: List of weights for each model
            
        Returns:
            Ensemble model wrapper
        """
        if weights is None:
            weights = [1/len(models)] * len(models)
        
        class EnsembleModel:
            def __init__(self, models, weights):
                self.models = models
                self.weights = weights
            
            def predict_proba(self, X):
                probas = np.zeros((X.shape[0], 2))
                for model, weight in zip(self.models, self.weights):
                    probas += weight * model.predict_proba(X)
                return probas
            
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        
        return EnsembleModel(models, weights)
    
    def optimize_threshold(self, model: Any, X_val: pd.DataFrame, y_val: pd.Series,
                          cost_ratio: float = 10.0) -> float:
        """
        Optimize classification threshold based on business costs.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation labels
            cost_ratio: Cost of false negative / cost of false positive
            
        Returns:
            Optimal threshold
        """
        logger.info("Optimizing classification threshold...")
        
        # Get predicted probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Try different thresholds
        thresholds = np.linspace(0, 1, 100)
        best_threshold = 0.5
        best_cost = float('inf')
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate total cost
            cm = confusion_matrix(y_val, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # FP cost = 1, FN cost = cost_ratio
            total_cost = fp + (fn * cost_ratio)
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.3f} with cost: {best_cost:.2f}")
        return best_threshold