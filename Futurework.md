
# AIOps Guardian Architecture

## Overview

The AIOps Guardian system collects system telemetry, exports metrics to Prometheus,
and applies machine learning anomaly detection models.

## Data Flow

System Metrics
      ↓
aiops-watchdog exporters
      ↓
Prometheus
      ↓
Grafana dashboards
      ↓
ML anomaly detection models

## Components

aiops-watchdog-knn.py  
Prometheus exporter with KNN anomaly detection

aiops-watchdog-ml.py  
Telemetry collection pipeline for training datasets

train_knn_final.py  
KNN model training

train_iforest.py  
Isolation Forest training
