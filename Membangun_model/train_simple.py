#!/usr/bin/env python3
"""
Simple MLflow Training Script for Telco Customer Churn
Author: SHAH FIRIZKI AZMI
Date: September 17, 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import warnings
warnings.filterwarnings('ignore')

def simple_preprocessing(df):
    """
    Simple preprocessing for telco churn data
    """
    print("🔧 Starting simple preprocessing...")
    
    # Drop customer ID
    df = df.drop(['customerID'], axis=1)
    print("✅ Dropped customerID column")
    
    # Fix TotalCharges - convert to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print("✅ Fixed TotalCharges column")
    
    # Encode categorical variables
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
    
    le = LabelEncoder()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
            print(f"✅ Encoded {col}")
    
    # Encode target variable
    df['Churn'] = le.fit_transform(df['Churn'])
    print("✅ Encoded Churn target")
    
    print(f"✅ Preprocessing completed: {df.shape}")
    return df

def train_simple_model():
    """
    Simple model training with manual MLflow logging
    """
    print("\n" + "="*80)
    print("🚀 SIMPLE MLFLOW TRAINING")
    print("="*80)
    
    # Load data
    print("📊 Loading dataset...")
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    print(f"✅ Dataset loaded: {df.shape}")
    
    # Preprocess data
    df_processed = simple_preprocessing(df)
    
    # Prepare features and target
    X = df_processed.drop(['Churn'], axis=1)
    y = df_processed['Churn']
    
    print(f"📊 Features shape: {X.shape}")
    print(f"📊 Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📊 Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Telco_Churn_Basic")
    
    # Start MLflow run
    with mlflow.start_run(run_name="telco_churn_simple_training"):
        print("🎯 MLflow run started")
        
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        print("🤖 Training model...")
        model.fit(X_train_scaled, y_train)
        print("✅ Model trained")
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"📊 Precision: {precision:.4f}")
        print(f"📊 Recall: {recall:.4f}")
        print(f"📊 F1-Score: {f1:.4f}")
        print(f"📊 AUC: {auc:.4f}")
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 5)
        mlflow.log_param("min_samples_leaf", 2)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="TelcoChurnClassifier"
        )
        
        print("✅ All parameters, metrics, and model logged to MLflow")
        
        # Get run info
        run = mlflow.active_run()
        print(f"🏃 Run ID: {run.info.run_id}")
        print(f"🧪 Experiment ID: {run.info.experiment_id}")
        print(f"🌐 View run: http://127.0.0.1:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    train_simple_model()