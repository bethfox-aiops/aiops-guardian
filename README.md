# AIOps Guardian

A local AIOps experimentation platform combining:

- Prometheus metrics exporters
- Grafana dashboards
- machine learning anomaly detection (KNN + Isolation Forest)

## Components

aiops-watchdog-knn.py  
Prometheus exporter that collects system telemetry and runs anomaly detection.

aiops-watchdog-ml.py  
Data collection pipeline for training datasets.

train_knn_final.py  
KNN anomaly detection training script.

train_iforest.py  
Isolation Forest training script.

## Stack

Prometheus → metrics collection  
Grafana → visualization dashboards  
Python → exporters and ML models
