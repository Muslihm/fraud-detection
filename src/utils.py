"""
Utility functions for fraud detection system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution") -> None:
    """
    Plot class distribution of target variable.
    
    Args:
        y: Target series
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count plot
    y.value_counts().plot(kind='bar', ax=ax1)
    ax1.set_title(f'{title} - Count')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Count')
    
    # Percentage plot
    percentages = y.value_counts(normalize=True) * 100
    percentages.plot(kind='bar', ax=ax2)
    ax2.set_title(f'{title} - Percentage')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Percentage (%)')
    
    # Add percentage labels
    for i, p in enumerate(percentages):
        ax2.text(i, p + 0.5, f'{p:.2f}%', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Class distribution:\n{y.value_counts()}")
    logger.info(f"Imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.2f}:1")


def plot_confusion_matrix(cm: np.ndarray, model_name: str = "Model") -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        model_name: Name of the model
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                   model_name: str = "Model") -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = np.trapz(tpr, fpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                                model_name: str = "Model") -> None:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = np.trapz(precision, recall)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'{model_name} (PR-AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names: list, top_n: int = 20) -> None:
    """
    Plot feature importance from tree-based models.
    
    Args:
        model: Trained tree-based model
        feature_names: List of feature names
        top_n: Number of top features to display
    """
    # Extract feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        logger.warning("Model doesn't have feature_importances_ or coef_ attribute")
        return
    
    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='importance', y='feature')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5) -> pd.Series:
    """
    Detect outliers using IQR method.
    
    Args:
        df: DataFrame
        column: Column name to check
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        Boolean Series indicating outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    logger.info(f"Column {column}: Found {outliers.sum()} outliers "
               f"({outliers.sum()/len(df)*100:.2f}%)")
    
    return outliers


def calculate_psd(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Percentage of Successful Detection for fraud.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        PSD score
    """
    fraud_cases = (y_true == 1).sum()
    detected_fraud = ((y_true == 1) & (y_pred == 1)).sum()
    
    if fraud_cases == 0:
        return 1.0
    
    return detected_fraud / fraud_cases


def calculate_far(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate False Alarm Rate.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        FAR score
    """
    legitimate_cases = (y_true == 0).sum()
    false_alarms = ((y_true == 0) & (y_pred == 1)).sum()
    
    if legitimate_cases == 0:
        return 0.0
    
    return false_alarms / legitimate_cases