from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def test_task():
    print("✅ Airflow task running successfully!")

with DAG(
    dag_id='airflow_migration_test',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,  
    catchup=False,
    tags=["migration_test"]
) as dag:
    t1 = PythonOperator(
        task_id='test_task',
        python_callable=test_task,
    )
