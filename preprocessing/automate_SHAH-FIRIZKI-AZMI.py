"""
Telco Customer Churn Data Preprocessing Automation
=================================================

This module provides automated data preprocessing for Telco Customer Churn dataset.
Designed with clean code principles: Scalability, Readability, Maintainability, 
Reusability, Performance, Testability, Security, Portability, and Documentation.

Author: SHAH FIRIZKI AZMI
Date: September 17, 2025
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import logging
from typing import Tuple, Dict, Any, Optional
import warnings
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class TelcoChurnPreprocessor:
    """
    A comprehensive preprocessor for Telco Customer Churn dataset.
    
    This class follows SOLID principles and provides:
    - Scalable data processing pipeline
    - Configurable preprocessing parameters
    - Reusable components
    - Comprehensive error handling
    - Performance optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config (Dict): Configuration parameters for preprocessing
        """
        self.config = config or self._get_default_config()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[list] = None
        
        # Security: Validate paths
        self._validate_config()
        
        logger.info("TelcoChurnPreprocessor initialized successfully")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration for preprocessing.
        
        Returns:
            Dict: Default configuration parameters
        """
        return {
            'test_size': 0.2,
            'random_state': 42,
            'stratify': True,
            'scale_features': True,
            'handle_missing': True,
            'save_artifacts': True,
            'output_dir': 'preprocessing',
            'target_column': 'Churn',
            'id_column': 'customerID'
        }
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters for security and correctness.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_keys = ['test_size', 'random_state', 'target_column']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        if not 0 < self.config['test_size'] < 1:
            raise ValueError("test_size must be between 0 and 1")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV file with comprehensive error handling.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
            Exception: For other data loading errors
        """
        try:
            # Security: Validate file path
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")
            
            # Performance: Use efficient data types
            df = pd.read_csv(file_path, low_memory=False)
            
            if df.empty:
                raise pd.errors.EmptyDataError("Dataset is empty")
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with appropriate strategies.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with handled missing values
        """
        df_clean = df.copy()
        
        # Handle TotalCharges column (contains spaces instead of NaN)
        if 'TotalCharges' in df_clean.columns:
            df_clean['TotalCharges'] = pd.to_numeric(
                df_clean['TotalCharges'], 
                errors='coerce'
            )
            
            # Fill with median for numerical stability
            median_total_charges = df_clean['TotalCharges'].median()
            df_clean['TotalCharges'].fillna(median_total_charges, inplace=True)
            
            logger.info("TotalCharges missing values handled")
        
        # Check for other missing values
        missing_counts = df_clean.isnull().sum()
        if missing_counts.any():
            logger.warning(f"Additional missing values found: {missing_counts[missing_counts > 0].to_dict()}")
        
        return df_clean
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Dataframe with encoded categorical features
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        # Remove ID column from encoding
        if self.config['id_column'] in categorical_cols:
            categorical_cols = categorical_cols.drop(self.config['id_column'])
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            self.label_encoders[col] = le
            
            logger.debug(f"Encoded column: {col}")
        
        logger.info(f"Encoded {len(categorical_cols)} categorical columns")
        return df_encoded
    
    def _prepare_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target variable.
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Remove unnecessary columns
        columns_to_drop = [self.config['id_column'], self.config['target_column']]
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        X = df.drop(columns=columns_to_drop)
        y = df[self.config['target_column']]
        
        self.feature_names = list(X.columns)
        
        logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
        return X, y
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        stratify = y if self.config['stratify'] else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=stratify
        )
        
        logger.info(f"Data split completed. Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def _scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Testing features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Scaled training and testing features
        """
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled
    
    def _save_artifacts(self, df_processed: pd.DataFrame) -> None:
        """
        Save preprocessing artifacts for reusability.
        
        Args:
            df_processed (pd.DataFrame): Processed dataframe
        """
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Save processed data
            processed_file = output_dir / 'telco_churn_preprocessing.csv'
            df_processed.to_csv(processed_file, index=False)
            
            # Save preprocessing artifacts
            if self.scaler:
                scaler_file = output_dir / 'scaler.pkl'
                joblib.dump(self.scaler, scaler_file)
            
            if self.label_encoders:
                encoders_file = output_dir / 'label_encoders.pkl'
                joblib.dump(self.label_encoders, encoders_file)
            
            # Save feature names
            if self.feature_names:
                features_file = output_dir / 'feature_names.pkl'
                joblib.dump(self.feature_names, features_file)
            
            # Save configuration
            config_file = output_dir / 'preprocessing_config.pkl'
            joblib.dump(self.config, config_file)
            
            logger.info(f"Artifacts saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving artifacts: {str(e)}")
            raise
    
    def preprocess(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Complete preprocessing pipeline.
        
        Args:
            file_path (str): Path to the dataset
            
        Returns:
            Tuple: X_train_scaled, X_test_scaled, y_train, y_test
        """
        logger.info("Starting preprocessing pipeline")
        
        # Load data
        df = self.load_data(file_path)
        
        # Handle missing values
        if self.config['handle_missing']:
            df = self._handle_missing_values(df)
        
        # Encode categorical features
        df_encoded = self._encode_categorical_features(df)
        
        # Prepare features and target
        X, y = self._prepare_features_target(df_encoded)
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Scale features
        if self.config['scale_features']:
            X_train_scaled, X_test_scaled = self._scale_features(X_train, X_test)
        else:
            X_train_scaled, X_test_scaled = X_train.values, X_test.values
        
        # Save artifacts
        if self.config['save_artifacts']:
            self._save_artifacts(df_encoded)
        
        logger.info("Preprocessing pipeline completed successfully")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    @classmethod
    def load_artifacts(cls, artifacts_dir: str = 'preprocessing') -> 'TelcoChurnPreprocessor':
        """
        Load saved preprocessing artifacts for reuse.
        
        Args:
            artifacts_dir (str): Directory containing artifacts
            
        Returns:
            TelcoChurnPreprocessor: Configured preprocessor instance
        """
        artifacts_path = Path(artifacts_dir)
        
        # Load configuration
        config_file = artifacts_path / 'preprocessing_config.pkl'
        config = joblib.load(config_file) if config_file.exists() else None
        
        # Create instance
        preprocessor = cls(config)
        
        # Load artifacts
        scaler_file = artifacts_path / 'scaler.pkl'
        if scaler_file.exists():
            preprocessor.scaler = joblib.load(scaler_file)
        
        encoders_file = artifacts_path / 'label_encoders.pkl'
        if encoders_file.exists():
            preprocessor.label_encoders = joblib.load(encoders_file)
        
        features_file = artifacts_path / 'feature_names.pkl'
        if features_file.exists():
            preprocessor.feature_names = joblib.load(features_file)
        
        logger.info(f"Artifacts loaded from {artifacts_path}")
        return preprocessor


def main():
    """
    Main function to demonstrate usage.
    """
    try:
        # Configuration for different environments (Portability)
        config = {
            'test_size': 0.2,
            'random_state': 42,
            'stratify': True,
            'scale_features': True,
            'handle_missing': True,
            'save_artifacts': True,
            'output_dir': 'preprocessing',
            'target_column': 'Churn',
            'id_column': 'customerID'
        }
        
        # Initialize preprocessor
        preprocessor = TelcoChurnPreprocessor(config)
        
        # Run preprocessing
        X_train, X_test, y_train, y_test = preprocessor.preprocess(
            '../WA_Fn-UseC_-Telco-Customer-Churn.csv'
        )
        
        # Performance metrics
        print(f"\n{'='*50}")
        print("PREPROCESSING COMPLETED SUCCESSFULLY")
        print(f"{'='*50}")
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Target distribution - Train: {np.bincount(y_train)}")
        print(f"Target distribution - Test: {np.bincount(y_test)}")
        print(f"Features: {len(preprocessor.feature_names)}")
        print(f"Categorical encoders: {len(preprocessor.label_encoders)}")
        print(f"Artifacts saved to: {config['output_dir']}/")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()