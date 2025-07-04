from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

# import utils
from utils.data_utils import run_incremental, run_incremental_feature_generation, merge_features
from utils.model_utils import load_config, search_best_params, train_xgboost
from utils.s3_utils import upload_model_to_s3

# import env
from dotenv import load_dotenv
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = os.getenv("REPO_NAME", "huggingface/transformers")
DATA_DIR = os.getenv("DATA_BASE_DIR", "./data")
FEATURE_DIR = os.path.join(DATA_DIR, "features")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
PARAM_DIR = os.path.join(DATA_DIR, "params")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")
MODEL_FILE = os.path.join(MODEL_DIR, "latest_model.json")
BUCKET_NAME = os.getenv("MODEL_BUCKET")
S3_KEY = "model/latest_model.json"
os.makedirs(FEATURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAM_DIR, exist_ok=True)

# define tasks

def fetch_closed_issues_task(date=None):
    run_incremental(GITHUB_TOKEN, REPO_NAME, DATA_DIR, date)

def generate_features_task(date=None):
    run_incremental_feature_generation(DATA_DIR, FEATURE_DIR, date)

def merge_features_task(feature_dir=FEATURE_DIR, output_name="issues_features_full_plus_increment.parquet"):
    merge_features(feature_dir, output_name=output_name)

def search_best_params_task(features_path=os.path.join(DATA_DIR, "features/issues_features_full_plus_increment.parquet"),
                           config_path=CONFIG_PATH, model_dir=MODEL_DIR, param_dir=PARAM_DIR, auc_alert_threshold=0.6):
    config = load_config(config_path)
    n_trials = config.get("n_trials", 30)
    auc = search_best_params(features_path, n_trials, model_dir, param_dir)
    if auc < auc_alert_threshold:
        raise ValueError(f"Best AUC dropped below {auc_alert_threshold}: {auc}")
    return auc

def train_xgboost_task(features_path=os.path.join(DATA_DIR, "features/issues_features_full_plus_increment.parquet"),
                      params_path=os.path.join(DATA_DIR, "params/best_params.json"),
                      model_out=os.path.join(MODEL_DIR, "latest_model.json")):
    acc = train_xgboost(features_path, params_path, model_out)
    print(f"[MODEL] Train accuracy: {acc:.4f}")
    return acc

def upload_model_to_s3_task(local_model_file=MODEL_FILE, bucket_name=BUCKET_NAME, s3_key=S3_KEY, with_history=True):
    upload_model_to_s3(local_model_file, bucket_name, s3_key, with_history)

# define Airflow DAG 
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

with DAG(
    dag_id='github_issues_mlops_pipeline',
    default_args=default_args,
    schedule=None,  
    catchup=False,
    tags=["github", "mlops"],
) as dag:

    t1 = PythonOperator(
        task_id='fetch_closed_issues',
        python_callable=fetch_closed_issues_task
    )

    t2 = PythonOperator(
        task_id='generate_features',
        python_callable=generate_features_task
    )

    t3 = PythonOperator(
        task_id='merge_features',
        python_callable=merge_features_task
    )

    t4 = PythonOperator(
        task_id='search_best_params',
        python_callable=search_best_params_task
    )

    t5 = PythonOperator(
        task_id='train_xgboost',
        python_callable=train_xgboost_task
    )

    t6 = PythonOperator(
        task_id='upload_model_to_s3',
        python_callable=upload_model_to_s3_task
    )

    # tasks
    t1 >> t2 >> t3 >> t4 >> t5 >> t6
