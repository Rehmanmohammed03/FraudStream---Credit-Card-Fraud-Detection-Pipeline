import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowException
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    'depends_on_past': False,
    'owner': 'datamasterylab.com',
    'start_date': datetime(2025, 3, 3),
    'max_active_runs': 1,
}

def _train_model(**context):
    """Airflow wrapper for training task"""
    from fraud_detection_training import FraudDetectionTraining
    try:
        logger.info('Initializing fraud detection training')
        trainer = FraudDetectionTraining()
        model, precision = trainer.train_model()
        return { 'status': 'success', 'precision': precision }
    except Exception as e:
        logger.error('Training failed: %s', str(e), exc_info=True)
        raise AirflowException(f'Model training failed: {str(e)}')

with DAG(
    'fraud_detection_training',
    schedule_interval='0 3 * * *',
    default_args=default_args,
    description='Fraud detection model training pipeline',
    catchup=False,
    tags=['fraud', 'ML']
) as dag:

    validate_environment = BashOperator(
        task_id='validate_environment',
        bash_command='''
        echo "Validating environment..."
        test -f /app/config.yaml &&
        test -f /app/.env &&
        echo "Environment is valid!"
        '''
    )

    training_task = PythonOperator(
        task_id='execute_training',
        provide_context=True,
        python_callable=_train_model,
    )

    cleanup_task = BashOperator(
        task_id='cleanup_resources',
        trigger_rule='all_done',
        bash_command='rm -f /app/tmp/*.pkl',
    )

    validate_environment >> training_task >> cleanup_task