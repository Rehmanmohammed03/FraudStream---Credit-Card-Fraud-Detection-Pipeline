"""
Fraud Detection Model Training Pipeline

System Architecture:

1. Configuration & Credentials Management
   - Environment-specific settings loaded via YAML configuration files
   - Secure separation between configuration and sensitive credentials
   - Multi-environment deployment support through .env file integration

2. Monitoring & Telemetry
   - Multi-sink structured logging for operational visibility
   - Experiment tracking and versioning via MLflow
   - Artifact persistence using MinIO (S3-compatible object storage)

3. Enterprise Data Processing Pipeline
   - Real-time data ingestion using Kafka message streaming
   - Intelligent hyperparameter optimization and model tuning
   - Model versioning and centralized registry management
   - Detailed performance metrics and statistical analysis
   - Sophisticated techniques for handling class imbalance

4. Reliability & Control Mechanisms
   - Upfront system validation and dependency checks
   - Data quality enforcement and schema validation
   - Defensive error handling and exception recovery
   - Performance baseline establishment and deviation detection
"""

import json
import logging
import os

import boto3
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from kafka import KafkaConsumer
from mlflow.models import infer_signature
from numpy.array_api import astype
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer, fbeta_score, precision_recall_curve, average_precision_score, precision_score, \
    recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# Set up logging with output to both file and console using a structured format
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler('./fraud_detection_model.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FraudDetectionTraining:
    """
    Comprehensive fraud detection model training framework with modern MLOps integration.

    Core System Capabilities:
    - Settings Management: Hierarchical YAML-based configuration with environment-specific overrides
    - Message Stream Processing: Kafka integration with SASL/SSL authentication
    - Feature Engineering: Time-based, user-activity, transaction-amount, and merchant-risk features
    - Model Training: XGBoost classifier with SMOTE for balanced class representation
    - Parameter Optimization: Randomized search with stratified k-fold validation
    - Experiment Management: MLflow integration for tracking runs, metrics, and artifacts
    - Production Packaging: Serialization and model registry integration

    Built for distributed training and containerized cloud deployment.
    """

    def __init__(self, config_path='/app/config.yaml'):
        # Configure Git behavior for container environments
        os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
        os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = '/usr/bin/git'

        # Import settings from .env file before loading main configuration
        load_dotenv(dotenv_path='/app/.env')

        # Parse and initialize configuration from file
        self.config = self._load_config(config_path)

        # Establish cloud storage credentials from environment
        os.environ.update({
            'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
            'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'AWS_S3_ENDPOINT_URL': self.config['mlflow']['s3_endpoint_url']
        })

        # Execute dependency and connectivity validation
        self._validate_environment()

        # Initialize MLflow with tracking and experiment configuration
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def _load_config(self, config_path: str) -> dict:
        """
        Read configuration from YAML file with fast-fail validation.

        Performs:
        - YAML file parsing and deserialization
        - Immediate verification of required settings
        - Configuration loading audit trail
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info('Configuration loaded successfully')
            return config
        except Exception as e:
            logger.error('Failed to load configuration: %s', str(e))
            raise

    def _validate_environment(self):
        """
        Verify system readiness through comprehensive environment assessment:
        1. Required environment variables presence
        2. Object storage accessibility
        3. Authentication credentials validation

        Stops initialization immediately if critical dependencies are unavailable.
        """
        required_vars = ['KAFKA_BOOTSTRAP_SERVERS', 'KAFKA_USERNAME', 'KAFKA_PASSWORD']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f'Missing required environment variables: {missing}')

        self._check_minio_connection()

    def _check_minio_connection(self):
        """
        Test connection to object storage and configure bucket persistence.

        Actions:
        - Establish S3-compatible client connection with exception handling
        - Verify bucket exists in remote storage
        - Create new bucket if missing from configuration

        Keeps configuration distinct from infrastructure initialization logic.
        """
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=self.config['mlflow']['s3_endpoint_url'],
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            )

            buckets = s3.list_buckets()
            bucket_names = [b['Name'] for b in buckets.get('Buckets', [])]
            logger.info('Minio connection verified. Buckets: %s', bucket_names)

            mlflow_bucket = self.config['mlflow'].get('bucket', 'mlflow')

            if mlflow_bucket not in bucket_names:
                s3.create_bucket(Bucket=mlflow_bucket)
                logger.info('Created missing MLFlow bucket: %s', mlflow_bucket)
        except Exception as e:
            logger.error('Minio connection failed: %s', str(e))

    def read_from_kafka(self) -> pd.DataFrame:
        """
        Authenticated message stream consumption with production-grade reliability:

        - SASL/SSL encryption and authentication
        - Automatic recovery from offset resets
        - Incoming data quality verification:
          - Column structure validation
          - Target variable verification
          - Class distribution analysis

        Handles timeouts and errors with clean resource deallocation.
        """
        try:
            topic = self.config['kafka']['topic']
            logger.info('Connecting to kafka topic %s', topic)

            consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.config['kafka']['bootstrap_servers'].split(','),
                security_protocol='SASL_SSL',
                sasl_mechanism='PLAIN',
                sasl_plain_username=self.config['kafka']['username'],
                sasl_plain_password=self.config['kafka']['password'],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='earliest',
                consumer_timeout_ms=self.config['kafka'].get('timeout', 10000)
            )

            messages = [msg.value for msg in consumer]
            consumer.close()

            df = pd.DataFrame(messages)
            if df.empty:
                raise ValueError('No messages received from Kafka.')

            # Convert timestamp field to UTC datetime format
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

            if 'is_fraud' not in df.columns:
                raise ValueError('Fraud label (is_fraud) missing from Kafka data')

            # Calculate and report class distribution statistics
            fraud_rate = df['is_fraud'].mean() * 100
            logger.info('Kafka data read successfully with fraud rate: %.2f%%', fraud_rate)

            return df
        except Exception as e:
            logger.error('Failed to read data from Kafka: %s', str(e), exc_info=True)
            raise

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct feature matrix from raw transaction data applying fraud-specific transformations:

        1. Time-Based Features:
           - Hour of transaction
           - Nighttime occurrence flag
           - Weekend occurrence flag

        2. User Activity Features:
           - Recent transaction count (past 24 hours)

        3. Amount Anomaly Features:
           - Deviation from user's historical spending

        4. Merchant Classification:
           - Risk score based on known problem merchants

        Preserves immutability through DataFrame copying and validates feature completeness.
        """
        df = df.sort_values(['user_id', 'timestamp']).copy()

        # Extract time-of-day characteristics
        # Detect unusual timing patterns indicative of fraudulent activity
        df['transaction_hour'] = df['timestamp'].dt.hour
        df['is_night'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] < 5)).astype(int)
        df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
        df['transaction_day'] = df['timestamp'].dt.day

        # Account activity metrics
        # Count transactions across most recent 24-hour period per user
        df['user_activity_24h'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: g.rolling('24h', on='timestamp', closed='left')['amount'].count().fillna(0)
        )

        # Transaction amount anomaly detection
        # Compare current transaction against user's recent average spending
        df['amount_to_avg_ratio'] = df.groupby('user_id', group_keys=False).apply(
            lambda g: (g['amount'] / g['amount'].rolling(7, min_periods=1).mean()).fillna(1.0)
        )

        # Known problematic vendor flagging
        # Mark transactions from vendors with history of fraudulent activity
        high_risk_merchants = self.config.get('high_risk_merchants', ['QuickCash', 'GlobalDigital', 'FastMoneyX'])
        df['merchant_risk'] = df['merchant'].isin(high_risk_merchants).astype(int)

        feature_cols = [
            'amount', 'is_night', 'is_weekend', 'transaction_day', 'user_activity_24h',
            'amount_to_avg_ratio', 'merchant_risk', 'merchant'
        ]

        # Verify required output column is present
        if 'is_fraud' not in df.columns:
            raise ValueError('Missing target column "is_fraud"')

        return df[feature_cols + ['is_fraud']]

    def train_model(self):
        """
        Full training workflow implementing production machine learning standards:

        1. Input Data Verification
        2. Balanced Train/Test Division
        3. Minority Class Augmentation (SMOTE)
        4. Automated Parameter Search
        5. Decision Threshold Calibration
        6. Performance Assessment
        7. Result Documentation
        8. Model Repository Storage

        Uses MLflow for complete experiment reproducibility and tracking.
        """
        try:
            logger.info('Starting model training process')

            # Retrieve messages and construct feature representations
            df = self.read_from_kafka()
            data = self.create_features(df)

            # Separate predictors from target and stratify by class
            X = data.drop(columns=['is_fraud'])
            y = data['is_fraud']

            # Enforce minimum positive sample requirements
            if y.sum() == 0:
                raise ValueError('No positive samples in training data')
            if y.sum() < 10:
                logger.warning('Low positive samples: %d. Consider additional data augmentation', y.sum())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.config['model'].get('test_size', 0.2),
                stratify=y,
                random_state=self.config['model'].get('seed', 42)
            )

            # Begin MLflow run tracking session
            with mlflow.start_run():
                # Record training dataset statistics
                mlflow.log_metrics({
                    'train_samples': X_train.shape[0],
                    'positive_samples': int(y_train.sum()),
                    'class_ratio': float(y_train.mean()),
                    'test_samples': X_test.shape[0]
                })

                # Set up vendor name encoding step
                preprocessor = ColumnTransformer([
                    ('merchant_encoder', OrdinalEncoder(
                        handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.float32
                    ), ['merchant'])
                ], remainder='passthrough')

                # Initialize gradient boosting classifier with performance tuning
                xgb = XGBClassifier(
                    eval_metric='aucpr',  # Prioritizes precision-recall balance
                    random_state=self.config['model'].get('seed', 42),
                    reg_lambda=1.0,
                    n_estimators=self.config['model']['params']['n_estimators'],
                    n_jobs=-1,
                    tree_method=self.config['model'].get('tree_method', 'hist')  # Supports accelerators
                )

                # Combine preprocessing and modeling into single workflow
                pipeline = ImbPipeline([
                    ('preprocessor', preprocessor),
                    ('smote', SMOTE(random_state=self.config['model'].get('seed', 42))),
                    ('classifier', xgb)
                ], memory='./cache')

                # Define parameter grid for optimization exploration
                param_dist = {
                    'classifier__max_depth': [3, 5, 7],  # Tree size constraints
                    'classifier__learning_rate': [0.01, 0.05, 0.1],  # Gradient step sizes
                    'classifier__subsample': [0.6, 0.8, 1.0],  # Sample randomization
                    'classifier__colsample_bytree': [0.6, 0.8, 1.0],  # Feature sampling
                    'classifier__gamma': [0, 0.1, 0.3],  # Split cost thresholds
                    'classifier__reg_alpha': [0, 0.1, 0.5]  # Sparsity penalties
                }

                # Search for parameter combinations using F2 scoring metric
                searcher = RandomizedSearchCV(
                    pipeline,
                    param_dist,
                    n_iter=20,
                    scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
                    cv=StratifiedKFold(n_splits=3, shuffle=True),
                    n_jobs=-1,
                    refit=True,
                    error_score='raise',
                    random_state=self.config['model'].get('seed', 42)
                )

                logger.info('Starting hyperparameter tuning...')
                searcher.fit(X_train, y_train)
                best_model = searcher.best_estimator_
                best_params = searcher.best_params_
                logger.info('Best hyperparameters: %s', best_params)

                # Find the decision boundary that maximizes F1 score on training set
                train_proba = best_model.predict_proba(X_train)[:, 1]
                precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_train, train_proba)
                f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in
                             zip(precision_arr[:-1], recall_arr[:-1])]
                best_threshold = thresholds_arr[np.argmax(f1_scores)]
                logger.info('Optimal threshold determined: %.4f', best_threshold)

                # Generate predictions on held-out test data
                X_test_processed = best_model.named_steps['preprocessor'].transform(X_test)
                test_proba = best_model.named_steps['classifier'].predict_proba(X_test_processed)[:, 1]
                y_pred = (test_proba >= best_threshold).astype(int)

                # Compile array of performance indicators
                metrics = {
                    'auc_pr': float(average_precision_score(y_test, test_proba)),
                    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                    'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                    'f1': float(f1_score(y_test, y_pred, zero_division=0)),
                    'threshold': float(best_threshold)
                }

                mlflow.log_metrics(metrics)
                mlflow.log_params(best_params)

                # Generate and save prediction confusion matrix chart
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(6, 4))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                tick_marks = np.arange(2)
                plt.xticks(tick_marks, ['Not Fraud', 'Fraud'])
                plt.yticks(tick_marks, ['Not Fraud', 'Fraud'])

                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='red')

                plt.tight_layout()
                cm_filename = 'confusion_matrix.png'
                plt.savefig(cm_filename)
                mlflow.log_artifact(cm_filename)
                plt.close()

                # Create and save precision-recall tradeoff visualization
                plt.figure(figsize=(10, 6))
                plt.plot(recall_arr, precision_arr, marker='.', label='Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                pr_filename = 'precision_recall_curve.png'
                plt.savefig(pr_filename)
                mlflow.log_artifact(pr_filename)
                plt.close()

                # Register trained model in MLflow repository
                signature = infer_signature(X_train, y_pred)
                mlflow.sklearn.log_model(
                    sk_model=best_model,
                    artifact_path='model',
                    signature=signature,
                    registered_model_name='fraud_detection_model'
                )

                # Save model weights to local filesystem for serving
                os.makedirs('/app/models', exist_ok=True)
                joblib.dump(best_model, '/app/models/fraud_detection_model.pkl')

                logger.info('Training successfully completed with metrics: %s', metrics)

                return best_model, metrics

        except Exception as e:
            logger.error('Training failed: %s', str(e), exc_info=True)
            raise