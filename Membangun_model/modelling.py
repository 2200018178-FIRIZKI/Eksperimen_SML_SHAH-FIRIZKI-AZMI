"""
MLflow Basic Model Training
===========================

This module provides basic model training with MLflow tracking.
Follows clean code principles for scalability and maintainability.

Author: SHAH FIRIZKI AZMI
Date: September 17, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import logging
from pathlib import Path
import sys
import os

# Add preprocessing module to path
sys.path.append('../preprocessing')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class BasicMLflowTrainer:
    """
    Basic MLflow model trainer with clean architecture.
    
    Follows SOLID principles:
    - Single Responsibility: Only handles basic model training
    - Open/Closed: Extensible for different models
    - Liskov Substitution: Can be substituted with other trainers
    - Interface Segregation: Minimal required interface
    - Dependency Inversion: Depends on abstractions
    """
    
    def __init__(self, experiment_name: str = "Telco_Churn_Basic"):
        """
        Initialize the basic trainer.
        
        Args:
            experiment_name (str): MLflow experiment name
        """
        self.experiment_name = experiment_name
        self.model = None
        self.scaler = None
        
        # Security: Set tracking URI safely
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking with proper configuration."""
        try:
            # Portability: Use local tracking by default
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment(self.experiment_name)
            logger.info(f"MLflow experiment set: {self.experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}. Using default tracking.")
            mlflow.set_experiment(self.experiment_name)
    
    def load_data(self) -> tuple:
        """
        Load preprocessed data.
        
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
    
    def train_model(self) -> None:
        """
        Train the model with MLflow tracking.
        """
        logger.info("Starting model training...")
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Start MLflow run
        with mlflow.start_run(run_name="basic_random_forest"):
            # Enable autolog for automatic tracking
            mlflow.sklearn.autolog()
            
            # Initialize model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log additional information
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_param("preprocessing", "StandardScaler")
            
            # Print results
            print(f"\n{'='*50}")
            print("BASIC MODEL TRAINING RESULTS")
            print(f"{'='*50}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-score:  {f1:.4f}")
            print(f"{'='*50}")
            
            logger.info("Model training completed successfully")


def main():
    """Main function to run basic model training."""
    try:
        # Check if preprocessed data exists
        if not Path('telco_churn_preprocessing.csv').exists():
            logger.error("Preprocessed data not found. Please run preprocessing first.")
            return
        
        # Initialize trainer
        trainer = BasicMLflowTrainer()
        
        # Train model
        trainer.train_model()
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()