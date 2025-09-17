"""
Prometheus Metrics Exporter for ML Model Monitoring
==================================================

This module provides comprehensive metrics collection for machine learning
model monitoring using Prometheus. Designed with scalability and clean
architecture principles.

Author: SHAH FIRIZKI AZMI
Date: September 17, 2025
Version: 1.0.0
"""

from prometheus_client import (
    start_http_server, Gauge, Counter, Histogram, Info, Enum
)
import time
import requests
import json
import random
import numpy as np
import psutil
import threading
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModelMetricsExporter:
    """
    Comprehensive ML model metrics exporter for Prometheus.
    
    Provides scalable, maintainable metrics collection with:
    - System performance metrics
    - Model performance metrics  
    - Business metrics
    - Custom application metrics
    """
    
    def __init__(self, port: int = 8000, model_endpoint: Optional[str] = None):
        """
        Initialize metrics exporter.
        
        Args:
            port (int): Port for metrics server
            model_endpoint (str): MLflow model serving endpoint
        """
        self.port = port
        self.model_endpoint = model_endpoint or "http://localhost:1234"
        self.running = False
        
        # Initialize metrics
        self._init_metrics()
        
        logger.info(f"MLModelMetricsExporter initialized on port {port}")
    
    def _init_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        
        # Model Performance Metrics
        self.model_accuracy = Gauge(
            'model_accuracy_score', 
            'Current model accuracy score'
        )
        
        self.model_precision = Gauge(
            'model_precision_score', 
            'Current model precision score'
        )
        
        self.model_recall = Gauge(
            'model_recall_score', 
            'Current model recall score'
        )
        
        self.model_f1_score = Gauge(
            'model_f1_score', 
            'Current model F1 score'
        )
        
        self.model_roc_auc = Gauge(
            'model_roc_auc_score', 
            'Current model ROC AUC score'
        )
        
        # Prediction Metrics
        self.prediction_requests_total = Counter(
            'prediction_requests_total', 
            'Total number of prediction requests'
        )
        
        self.prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Prediction request latency in seconds',
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0]
        )
        
        self.prediction_errors_total = Counter(
            'prediction_errors_total',
            'Total number of prediction errors',
            ['error_type']
        )
        
        # System Metrics
        self.cpu_usage_percent = Gauge(
            'system_cpu_usage_percent', 
            'CPU usage percentage'
        )
        
        self.memory_usage_percent = Gauge(
            'system_memory_usage_percent', 
            'Memory usage percentage'
        )
        
        self.disk_usage_percent = Gauge(
            'system_disk_usage_percent', 
            'Disk usage percentage'
        )
        
        self.network_bytes_sent = Gauge(
            'system_network_bytes_sent_total', 
            'Total network bytes sent'
        )
        
        self.network_bytes_recv = Gauge(
            'system_network_bytes_received_total', 
            'Total network bytes received'
        )
        
        # Application Metrics
        self.active_connections = Gauge(
            'app_active_connections', 
            'Number of active connections'
        )
        
        self.response_time_ms = Gauge(
            'app_response_time_milliseconds', 
            'Average response time in milliseconds'
        )
        
        self.error_rate_percent = Gauge(
            'app_error_rate_percent', 
            'Application error rate percentage'
        )
        
        self.throughput_rps = Gauge(
            'app_throughput_requests_per_second', 
            'Application throughput in requests per second'
        )
        
        # Business Metrics
        self.churn_predictions_total = Counter(
            'business_churn_predictions_total',
            'Total churn predictions made',
            ['prediction_class']
        )
        
        self.model_drift_score = Gauge(
            'business_model_drift_score',
            'Model drift detection score'
        )
        
        self.data_quality_score = Gauge(
            'business_data_quality_score',
            'Data quality assessment score'
        )
        
        # Model Info
        self.model_info = Info(
            'model_information',
            'Information about the deployed model'
        )
        
        # Model Status
        self.model_status = Enum(
            'model_status',
            'Current status of the model',
            states=['healthy', 'degraded', 'failed', 'unknown']
        )
        
        # Set initial model info
        self.model_info.info({
            'model_name': 'telco_churn_classifier',
            'model_version': '1.0.0',
            'framework': 'scikit-learn',
            'deployment_time': datetime.now().isoformat()
        })
        
        self.model_status.state('healthy')
    
    def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage_percent.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage_percent.set(memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage_percent.set(disk_percent)
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.network_bytes_sent.set(net_io.bytes_sent)
            self.network_bytes_recv.set(net_io.bytes_recv)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_model_metrics(self) -> None:
        """Collect simulated model performance metrics."""
        try:
            # Simulate model performance metrics with realistic variations
            base_accuracy = 0.85
            accuracy_variation = random.uniform(-0.05, 0.03)
            current_accuracy = max(0.7, min(0.95, base_accuracy + accuracy_variation))
            self.model_accuracy.set(current_accuracy)
            
            # Derive other metrics from accuracy with realistic relationships
            precision = max(0.6, min(0.9, current_accuracy + random.uniform(-0.1, 0.05)))
            recall = max(0.5, min(0.9, current_accuracy + random.uniform(-0.15, 0.1)))
            f1 = 2 * (precision * recall) / (precision + recall)
            roc_auc = max(0.6, min(0.95, current_accuracy + random.uniform(-0.05, 0.1)))
            
            self.model_precision.set(precision)
            self.model_recall.set(recall)
            self.model_f1_score.set(f1)
            self.model_roc_auc.set(roc_auc)
            
            # Model drift simulation
            drift_score = random.uniform(0.0, 0.3)  # 0 = no drift, 1 = high drift
            self.model_drift_score.set(drift_score)
            
            # Data quality simulation
            quality_score = random.uniform(0.8, 1.0)  # 0 = poor, 1 = excellent
            self.data_quality_score.set(quality_score)
            
            # Update model status based on performance
            if current_accuracy < 0.75 or drift_score > 0.25:
                self.model_status.state('degraded')
            elif current_accuracy < 0.65 or drift_score > 0.4:
                self.model_status.state('failed')
            else:
                self.model_status.state('healthy')
                
        except Exception as e:
            logger.error(f"Error collecting model metrics: {e}")
            self.model_status.state('unknown')
    
    def _collect_application_metrics(self) -> None:
        """Collect application-level metrics."""
        try:
            # Simulate application metrics
            self.active_connections.set(random.randint(10, 100))
            self.response_time_ms.set(random.uniform(50, 500))
            self.error_rate_percent.set(random.uniform(0, 5))
            self.throughput_rps.set(random.uniform(5, 50))
            
            # Simulate prediction requests
            if random.random() < 0.8:  # 80% chance of new request
                self.prediction_requests_total.inc()
                
                # Simulate latency
                latency = random.uniform(0.1, 2.0)
                self.prediction_latency.observe(latency)
                
                # Simulate predictions
                churn_prediction = random.choice(['churn', 'no_churn'])
                self.churn_predictions_total.labels(prediction_class=churn_prediction).inc()
                
                # Simulate occasional errors
                if random.random() < 0.02:  # 2% error rate
                    error_type = random.choice(['timeout', 'invalid_input', 'model_error'])
                    self.prediction_errors_total.labels(error_type=error_type).inc()
                    
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    def _test_model_endpoint(self) -> bool:
        """Test if model endpoint is available."""
        try:
            response = requests.get(f"{self.model_endpoint}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def _metrics_collection_loop(self) -> None:
        """Main metrics collection loop."""
        logger.info("Starting metrics collection loop")
        
        while self.running:
            try:
                # Collect all metrics
                self._collect_system_metrics()
                self._collect_model_metrics()
                self._collect_application_metrics()
                
                # Test model endpoint availability
                endpoint_available = self._test_model_endpoint()
                if not endpoint_available:
                    logger.warning(f"Model endpoint {self.model_endpoint} not available")
                
                # Wait before next collection
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(10)  # Wait longer on error
    
    def start(self) -> None:
        """Start the metrics exporter."""
        try:
            # Start Prometheus metrics server
            start_http_server(self.port)
            logger.info(f"Prometheus metrics server started on port {self.port}")
            
            # Start metrics collection
            self.running = True
            collection_thread = threading.Thread(target=self._metrics_collection_loop)
            collection_thread.daemon = True
            collection_thread.start()
            
            logger.info("Metrics collection started")
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                self.stop()
                
        except Exception as e:
            logger.error(f"Failed to start metrics exporter: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the metrics exporter."""
        self.running = False
        logger.info("Metrics exporter stopped")


def main():
    """Main function to start the metrics exporter."""
    try:
        # Configuration
        port = int(os.getenv('METRICS_PORT', 8000))
        model_endpoint = os.getenv('MODEL_ENDPOINT', 'http://localhost:1234')
        
        print(f"\n{'='*60}")
        print("ML MODEL METRICS EXPORTER")
        print(f"{'='*60}")
        print(f"📊 Metrics Server Port: {port}")
        print(f"🤖 Model Endpoint: {model_endpoint}")
        print(f"🔗 Metrics URL: http://localhost:{port}/metrics")
        print(f"{'='*60}")
        print("Press Ctrl+C to stop")
        print(f"{'='*60}\n")
        
        # Create and start exporter
        exporter = MLModelMetricsExporter(port=port, model_endpoint=model_endpoint)
        exporter.start()
        
    except Exception as e:
        logger.error(f"Failed to start metrics exporter: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()