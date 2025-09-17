"""
ML Model Inference Service
=========================

This module provides a comprehensive inference service for the trained
Telco Churn model with monitoring capabilities and clean architecture.

Author: SHAH FIRIZKI AZMI
Date: September 17, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import joblib
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Union
import sys
import os

# Add preprocessing module to path
sys.path.append('../preprocessing')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelcoChurnInference:
    """
    Telco Churn Model Inference Service.
    
    Provides:
    - Model loading and validation
    - Data preprocessing for inference
    - Batch and single predictions
    - Performance monitoring
    - Error handling and logging
    """
    
    def __init__(self, model_path: Optional[str] = None, artifacts_dir: str = '../preprocessing'):
        """
        Initialize the inference service.
        
        Args:
            model_path (str): Path to the trained model
            artifacts_dir (str): Directory containing preprocessing artifacts
        """
        self.model_path = model_path
        self.artifacts_dir = Path(artifacts_dir)
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
        
        # Performance tracking
        self.prediction_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        # Load model and preprocessing artifacts
        self._load_model_and_artifacts()
        
        logger.info("TelcoChurnInference service initialized successfully")
    
    def _load_model_and_artifacts(self) -> None:
        """Load model and preprocessing artifacts."""
        try:
            # Load preprocessing artifacts
            self._load_preprocessing_artifacts()
            
            # Load model (placeholder - in practice, load from MLflow)
            if self.model_path and Path(self.model_path).exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                # Fallback: Load a basic model for demonstration
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                
                # Train on preprocessed data if available
                self._train_fallback_model()
                logger.warning("Using fallback model - load actual trained model for production")
        
        except Exception as e:
            logger.error(f"Error loading model and artifacts: {e}")
            raise
    
    def _load_preprocessing_artifacts(self) -> None:
        """Load preprocessing artifacts."""
        try:
            # Load scaler
            scaler_path = self.artifacts_dir / 'scaler.pkl'
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Scaler loaded successfully")
            
            # Load label encoders
            encoders_path = self.artifacts_dir / 'label_encoders.pkl'
            if encoders_path.exists():
                self.label_encoders = joblib.load(encoders_path)
                logger.info(f"Loaded {len(self.label_encoders)} label encoders")
            
            # Load feature names
            features_path = self.artifacts_dir / 'feature_names.pkl'
            if features_path.exists():
                self.feature_names = joblib.load(features_path)
                logger.info(f"Loaded {len(self.feature_names)} feature names")
                
        except Exception as e:
            logger.error(f"Error loading preprocessing artifacts: {e}")
            raise
    
    def _train_fallback_model(self) -> None:
        """Train a fallback model using preprocessed data."""
        try:
            processed_data_path = self.artifacts_dir / 'telco_churn_preprocessing.csv'
            if processed_data_path.exists():
                df = pd.read_csv(processed_data_path)
                X = df.drop('Churn', axis=1)
                y = df['Churn']
                
                if self.scaler:
                    X_scaled = self.scaler.transform(X)
                    self.model.fit(X_scaled, y)
                else:
                    self.model.fit(X, y)
                
                logger.info("Fallback model trained successfully")
                
        except Exception as e:
            logger.warning(f"Could not train fallback model: {e}")
    
    def _preprocess_input(self, data: Union[Dict, pd.DataFrame]) -> np.ndarray:
        """
        Preprocess input data for inference.
        
        Args:
            data: Input data (dict or DataFrame)
            
        Returns:
            np.ndarray: Preprocessed data ready for prediction
        """
        try:
            # Convert dict to DataFrame
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                raise ValueError("Input data must be dict or DataFrame")
            
            # Handle missing values (same as training)
            if 'TotalCharges' in df.columns:
                df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
                df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
            
            # Remove ID column if present
            if 'customerID' in df.columns:
                df = df.drop('customerID', axis=1)
            
            # Encode categorical variables
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError as e:
                        logger.warning(f"Unknown category in {col}: {e}")
                        # Handle unknown categories by using most frequent class
                        df[col] = encoder.transform([encoder.classes_[0]])[0]
            
            # Ensure all expected features are present
            if self.feature_names:
                missing_features = set(self.feature_names) - set(df.columns)
                if missing_features:
                    logger.warning(f"Missing features: {missing_features}")
                    # Add missing features with default values
                    for feature in missing_features:
                        df[feature] = 0
                
                # Reorder columns to match training data
                df = df[self.feature_names]
            
            # Scale features
            if self.scaler:
                X_scaled = self.scaler.transform(df)
            else:
                X_scaled = df.values
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            raise
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single prediction.
        
        Args:
            customer_data: Customer data dictionary
            
        Returns:
            Dict containing prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess input
            X = self._preprocess_input(customer_data)
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            prediction_proba = self.model.predict_proba(X)[0]
            
            # Prepare result
            result = {
                'customer_id': customer_data.get('customerID', 'unknown'),
                'prediction': int(prediction),
                'prediction_label': 'Churn' if prediction == 1 else 'No Churn',
                'confidence': float(max(prediction_proba)),
                'churn_probability': float(prediction_proba[1]),
                'no_churn_probability': float(prediction_proba[0]),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }
            
            # Update metrics
            self.prediction_count += 1
            self.total_latency += time.time() - start_time
            
            logger.info(f"Prediction made for customer {result['customer_id']}: {result['prediction_label']}")
            
            return result
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, customers_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make batch predictions.
        
        Args:
            customers_data: List of customer data dictionaries
            
        Returns:
            List of prediction results
        """
        start_time = time.time()
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(customers_data)
            
            # Preprocess batch
            X = self._preprocess_input(df)
            
            # Make predictions
            predictions = self.model.predict(X)
            predictions_proba = self.model.predict_proba(X)
            
            # Prepare results
            results = []
            for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
                result = {
                    'customer_id': customers_data[i].get('customerID', f'customer_{i}'),
                    'prediction': int(pred),
                    'prediction_label': 'Churn' if pred == 1 else 'No Churn',
                    'confidence': float(max(proba)),
                    'churn_probability': float(proba[1]),
                    'no_churn_probability': float(proba[0])
                }
                results.append(result)
            
            # Add batch metadata
            batch_info = {
                'batch_size': len(customers_data),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat(),
                'results': results
            }
            
            # Update metrics
            self.prediction_count += len(customers_data)
            self.total_latency += time.time() - start_time
            
            logger.info(f"Batch prediction completed for {len(customers_data)} customers")
            
            return batch_info
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error making batch prediction: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and performance metrics.
        
        Returns:
            Dict containing model information
        """
        avg_latency = self.total_latency / max(1, self.prediction_count)
        
        return {
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_names),
            'preprocessing_artifacts': {
                'scaler_loaded': self.scaler is not None,
                'encoders_loaded': len(self.label_encoders),
                'features_loaded': len(self.feature_names)
            },
            'performance_metrics': {
                'total_predictions': self.prediction_count,
                'total_errors': self.error_count,
                'error_rate': self.error_count / max(1, self.prediction_count),
                'average_latency_ms': round(avg_latency * 1000, 2),
                'total_latency_seconds': round(self.total_latency, 2)
            },
            'service_status': 'healthy' if self.error_count / max(1, self.prediction_count) < 0.1 else 'degraded'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of the inference service.
        
        Returns:
            Dict containing health status
        """
        try:
            # Test prediction with dummy data
            test_data = {
                'gender': 'Female',
                'SeniorCitizen': 0,
                'Partner': 'Yes',
                'Dependents': 'No',
                'tenure': 1,
                'PhoneService': 'No',
                'MultipleLines': 'No phone service',
                'InternetService': 'DSL',
                'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes',
                'DeviceProtection': 'No',
                'TechSupport': 'No',
                'StreamingTV': 'No',
                'StreamingMovies': 'No',
                'Contract': 'Month-to-month',
                'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 29.85,
                'TotalCharges': '29.85'
            }
            
            # Make test prediction
            start_time = time.time()
            result = self.predict_single(test_data)
            response_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'response_time_ms': round(response_time * 1000, 2),
                'test_prediction': result['prediction_label'],
                'model_loaded': self.model is not None,
                'artifacts_loaded': {
                    'scaler': self.scaler is not None,
                    'encoders': len(self.label_encoders) > 0,
                    'features': len(self.feature_names) > 0
                }
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'model_loaded': self.model is not None
            }


def main():
    """Main function to demonstrate inference service."""
    try:
        # Initialize inference service
        inference_service = TelcoChurnInference()
        
        # Sample customer data
        sample_customer = {
            'customerID': 'DEMO-001',
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'No',
            'tenure': 12,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 85.0,
            'TotalCharges': '1020.0'
        }
        
        print(f"\n{'='*60}")
        print("TELCO CHURN INFERENCE SERVICE DEMO")
        print(f"{'='*60}")
        
        # Health check
        health = inference_service.health_check()
        print(f"\n🏥 Health Check: {health['status'].upper()}")
        
        # Single prediction
        print(f"\n🔮 Making prediction for customer {sample_customer['customerID']}...")
        result = inference_service.predict_single(sample_customer)
        
        print(f"\n📊 PREDICTION RESULTS:")
        print(f"   Customer ID: {result['customer_id']}")
        print(f"   Prediction: {result['prediction_label']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print(f"   Churn Probability: {result['churn_probability']:.3f}")
        print(f"   Processing Time: {result['processing_time_ms']} ms")
        
        # Model info
        model_info = inference_service.get_model_info()
        print(f"\n📈 MODEL INFORMATION:")
        print(f"   Model Type: {model_info['model_type']}")
        print(f"   Feature Count: {model_info['feature_count']}")
        print(f"   Total Predictions: {model_info['performance_metrics']['total_predictions']}")
        print(f"   Average Latency: {model_info['performance_metrics']['average_latency_ms']} ms")
        print(f"   Service Status: {model_info['service_status'].upper()}")
        print(f"\n{'='*60}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()