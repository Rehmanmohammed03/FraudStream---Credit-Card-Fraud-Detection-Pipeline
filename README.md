# FraudStream
### Real-Time Credit Card Fraud Detection Pipeline

FraudStream is a **production-style real-time fraud detection system** designed to simulate how financial institutions monitor and detect fraudulent transactions at scale. The platform processes streaming transaction data, trains machine learning models, and performs real-time inference to identify suspicious activity.

The project demonstrates how **distributed data pipelines, machine learning, and stream processing** can be combined to build systems similar to those used in modern fintech infrastructure.

---

# Overview

Financial institutions process **millions of transactions every minute**, making it critical to detect fraud quickly and reliably. FraudStream simulates this environment by streaming transaction data through a distributed pipeline where machine learning models analyze transactions in real time.

The system continuously retrains fraud detection models as new transaction data arrives and uses the latest model to classify incoming transactions as **fraudulent or legitimate**.

---

# Architecture
Transaction Generator
↓
Apache Kafka (Transaction Stream)
↓
Apache Airflow (Model Training Pipeline)
↓
MLflow (Model Tracking & Versioning)
↓
Apache Spark Streaming (Real-Time Inference)
↓
Fraud Alerts / Monitoring


### Pipeline Components

**Transaction Generator**  
Simulates financial transaction data and publishes events to Kafka topics.

**Apache Kafka**  
Acts as the real-time streaming backbone, ingesting and distributing transaction data across the system.

**Apache Airflow**  
Schedules and orchestrates machine learning pipelines that train fraud detection models from transaction data.

**MLflow**  
Tracks model performance metrics and manages model versioning to ensure only the best-performing model is deployed.

**Apache Spark Streaming**  
Consumes streaming transactions from Kafka and performs real-time inference to detect fraudulent activity.

**Fraud Alert Output**  
Fraud predictions can be written to alert systems, dashboards, or monitoring services.

---

# Tech Stack

## Data Infrastructure
- Apache Kafka
- Apache Spark
- Apache Airflow
- PostgreSQL
- Redis
- MinIO (S3-compatible storage)

## Machine Learning
- Python
- Scikit-learn
- XGBoost
- Pandas

## MLOps
- MLflow
- Docker
- Docker Compose

---

# Key Features

### Real-Time Fraud Detection
Streaming architecture that detects fraudulent transactions as data flows through the system.

### Distributed Data Pipeline
Built using Kafka, Spark, and Airflow to mimic real-world data infrastructure.

### Automated Model Retraining
Models are periodically retrained as new transaction data arrives.

### Model Versioning
MLflow tracks model performance and promotes only higher-performing models to production.

### Production-Style Architecture
Designed to resemble real fintech transaction monitoring systems.

---

# Running the System

## Requirements

- Python 3.10+
- Docker
- Docker Compose

Recommended:
- 16GB RAM (system can run with reduced resources in development mode)

---


Start infrastructure services:
docker-compose up -d


Launch the transaction producer:

python producer/produce_transactions.py


Run model training pipeline:
airflow dags trigger fraud_training_pipeline


Start real-time inference:
python spark/streaming_inference.py


---

# Example Workflow

1. Transactions are generated and streamed into Kafka.
2. Airflow periodically trains fraud detection models using historical transaction data.
3. MLflow evaluates and tracks model performance.
4. Spark Streaming loads the latest model and classifies incoming transactions.
5. Fraudulent transactions are flagged and written to a monitoring topic.

---

# Project Goals

This project demonstrates how modern fintech systems combine:

- Distributed streaming infrastructure
- Machine learning pipelines
- Real-time inference systems
- Model lifecycle management

to detect fraud at scale.

---

# Future Improvements

- Real-time fraud alert dashboard
- Feature store integration
- Advanced fraud detection models
- Kubernetes deployment
- Online learning for continuous model updates

---

Credits to: Yusuf Ganiyu's Playlist
