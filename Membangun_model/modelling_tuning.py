"""
MLflow Advanced Model Training with Hyperparameter Tuning
=========================================================

This module provides advanced model training with hyperparameter tuning,
manual logging, and advanced metrics for skilled/advanced MLflow usage.

Author: SHAH FIRIZKI AZMI  
Date: September 17, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from pathlib import Path
import sys
import os
import time
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class AdvancedMLflowTrainer:
    """
    Advanced MLflow model trainer with hyperparameter tuning and manual logging.
    
    Features:
    - Hyperparameter tuning with GridSearch/RandomizedSearch
    - Manual logging with comprehensive metrics
    - Multiple model support
    - Advanced performance metrics
    - Artifact management
    - Cross-validation tracking
    """
    
    def __init__(self, experiment_name: str = "Telco_Churn_Advanced"):
        """
        Initialize the advanced trainer.
        
        Args:
            experiment_name (str): MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.best_model = None
        self.scaler = None
        self.models_config = self._get_models_config()
        
        # Setup MLflow
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking with advanced configuration."""
        try:
            # For Advanced: DagsHub integration (uncomment when ready)
            # import dagshub
            # dagshub.init(repo_owner='your_username', repo_name='your_repo', mlflow=True)
            
            # Local tracking for now
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment set: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Using default tracking.")
            mlflow.set_experiment(self.experiment_name)
    
    def _get_models_config(self) -> dict:
        """
        Get configuration for different models and their hyperparameters.
        
        Returns:
            dict: Models configuration
        """
        return {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            }
        }
    
    def load_data(self) -> tuple:
        """
        Load and prepare data for training.
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        try:
            # Load preprocessed data
            df = pd.read_csv('telco_churn_preprocessing.csv')
            
            # Separate features and target
            X = df.drop('Churn', axis=1)
            y = df['Churn']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            logger.info(f"Data loaded. Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_proba: np.ndarray) -> dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            dict: Comprehensive metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        # Additional advanced metrics
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        metrics['pr_auc'] = auc(recall_curve, precision_curve)
        
        # Class-specific metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def _create_visualizations(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              y_pred_proba: np.ndarray) -> dict:
        """
        Create visualization artifacts.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            y_pred_proba: Prediction probabilities
            
        Returns:
            dict: Paths to saved plots
        """
        plots = {}
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        confusion_path = 'confusion_matrix.png'
        plt.savefig(confusion_path)
        plt.close()
        plots['confusion_matrix'] = confusion_path
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_true, y_pred_proba):.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = 'roc_curve.png'
        plt.savefig(roc_path)
        plt.close()
        plots['roc_curve'] = roc_path
        
        # Precision-Recall Curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_curve, precision_curve, label=f'PR Curve (AUC = {auc(recall_curve, precision_curve):.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.tight_layout()
        pr_path = 'precision_recall_curve.png'
        plt.savefig(pr_path)
        plt.close()
        plots['pr_curve'] = pr_path
        
        return plots
    
    def train_with_tuning(self, model_name: str = 'random_forest', 
                         search_type: str = 'grid') -> None:
        """
        Train model with hyperparameter tuning and comprehensive logging.
        
        Args:
            model_name (str): Model to train ('random_forest' or 'gradient_boosting')
            search_type (str): Search strategy ('grid' or 'random')
        """
        logger.info(f"Starting {model_name} training with {search_type} search...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Get model configuration
        if model_name not in self.models_config:
            raise ValueError(f"Model {model_name} not supported")
        
        model_config = self.models_config[model_name]
        base_model = model_config['model']
        param_grid = model_config['params']
        
        # Start MLflow run
        run_name = f"{model_name}_{search_type}_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            start_time = time.time()
            
            # Hyperparameter search
            if search_type == 'grid':
                search = GridSearchCV(
                    base_model, param_grid, cv=5, scoring='f1',
                    n_jobs=-1, verbose=1
                )
            elif search_type == 'random':
                search = RandomizedSearchCV(
                    base_model, param_grid, cv=5, scoring='f1',
                    n_jobs=-1, verbose=1, n_iter=50, random_state=42
                )
            else:
                raise ValueError(f"Search type {search_type} not supported")
            
            # Fit search
            search.fit(X_train, y_train)
            
            # Best model
            self.best_model = search.best_estimator_
            
            # Predictions
            y_pred = self.best_model.predict(X_test)
            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
            
            # Training time
            training_time = time.time() - start_time
            
            # Calculate comprehensive metrics
            metrics = self._calculate_advanced_metrics(y_test, y_pred, y_pred_proba)
            
            # Create visualizations
            plots = self._create_visualizations(y_test, y_pred, y_pred_proba)
            
            # Manual Logging - Parameters
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("search_type", search_type)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_param("scoring_metric", "f1")
            mlflow.log_param("training_time_seconds", training_time)
            
            # Log best hyperparameters
            for param, value in search.best_params_.items():
                mlflow.log_param(f"best_{param}", value)
            
            # Manual Logging - Metrics
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Cross-validation metrics
            mlflow.log_metric("cv_score_mean", search.best_score_)
            mlflow.log_metric("cv_score_std", search.cv_results_['std_test_score'][search.best_index_])
            
            # Additional metrics for Advanced level
            mlflow.log_metric("n_support_vectors", len(y_train))  # Dataset size info
            mlflow.log_metric("feature_count", X_train.shape[1])
            mlflow.log_metric("training_samples", X_train.shape[0])
            mlflow.log_metric("test_samples", X_test.shape[0])
            
            # Log model artifacts
            mlflow.sklearn.log_model(self.best_model, "best_model")
            mlflow.sklearn.log_model(self.scaler, "scaler")
            
            # Log visualization artifacts
            for plot_name, plot_path in plots.items():
                mlflow.log_artifact(plot_path)
                os.remove(plot_path)  # Cleanup
            
            # Log additional artifacts
            # Classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_path = "classification_report.csv"
            report_df.to_csv(report_path)
            mlflow.log_artifact(report_path)
            os.remove(report_path)
            
            # Feature importance (if available)
            if hasattr(self.best_model, 'feature_importances_'):
                # Load feature names
                df = pd.read_csv('telco_churn_preprocessing.csv')
                feature_names = df.drop('Churn', axis=1).columns.tolist()
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': self.best_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = "feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path)
                os.remove(importance_path)
            
            # Print comprehensive results
            print(f"\n{'='*70}")
            print(f"ADVANCED MODEL TRAINING RESULTS - {model_name.upper()}")
            print(f"{'='*70}")
            print(f"Best Parameters: {search.best_params_}")
            print(f"Cross-validation Score: {search.best_score_:.4f}")
            print(f"Training Time: {training_time:.2f} seconds")
            print("\nTest Set Performance:")
            print(f"  Accuracy:     {metrics['accuracy']:.4f}")
            print(f"  Precision:    {metrics['precision']:.4f}")
            print(f"  Recall:       {metrics['recall']:.4f}")
            print(f"  F1-score:     {metrics['f1_score']:.4f}")
            print(f"  ROC-AUC:      {metrics['roc_auc']:.4f}")
            print(f"  PR-AUC:       {metrics['pr_auc']:.4f}")
            print(f"  Specificity:  {metrics['specificity']:.4f}")
            print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
            print(f"{'='*70}")
            
            logger.info("Advanced model training completed successfully")


def main():
    """Main function to run advanced model training."""
    try:
        # Check if preprocessed data exists
        if not Path('telco_churn_preprocessing.csv').exists():
            logger.error("Preprocessed data not found. Please run preprocessing first.")
            return
        
        # Initialize trainer
        trainer = AdvancedMLflowTrainer()
        
        # Train multiple models for comparison
        models_to_train = ['random_forest', 'gradient_boosting']
        
        for model_name in models_to_train:
            print(f"\n{'*'*50}")
            print(f"Training {model_name.replace('_', ' ').title()}")
            print(f"{'*'*50}")
            
            # Train with grid search
            trainer.train_with_tuning(model_name, 'grid')
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()