from prefect import flow, task, get_run_logger
from utils.data_utils import run_incremental, run_incremental_feature_generation, merge_features
from utils.model_utils import load_config, search_best_params, train_xgboost
from utils.s3_utils import upload_model_to_s3
import os
from dotenv import load_dotenv
import time 

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

@task
def fetch_closed_issues_task(date=None):
    """
    Fetch closed issues for the specified date (defaults to yesterday).
    """
    run_incremental(GITHUB_TOKEN, REPO_NAME, DATA_DIR, date)

@task
def generate_features_task(date=None):
    """
    Incremental feature engineering (defaults to yesterday).
    """
    run_incremental_feature_generation(DATA_DIR, FEATURE_DIR, date)

@task
def merge_features_task(
    feature_dir=FEATURE_DIR,
    output_name="issues_features_full_plus_increment.parquet"
):
    """
    Merge all feature files: prioritize the full feature file, supplement with incrementals, 
    and output the final deduplicated feature set.
    """
    merge_features(feature_dir, output_name=output_name)

@task
def search_best_params_task(
    features_path=os.path.join(DATA_DIR, "features/issues_features_full_plus_increment.parquet"),
    config_path=CONFIG_PATH,
    model_dir=MODEL_DIR,
    param_dir=PARAM_DIR,
    auc_alert_threshold=0.6
):
    """
    Search for the best XGBoost parameters and save them to model_dir and param_dir.
    """
    config = load_config(config_path)
    n_trials = config.get("n_trials", 30)
    auc = search_best_params(features_path, n_trials, model_dir, param_dir)
    logger = get_run_logger()
    if auc < auc_alert_threshold:
        logger.error(f"[ALERT] Best AUC dropped below threshold! Current: {auc}")
        raise ValueError(f"Best AUC dropped below {auc_alert_threshold}: {auc}")
    else:
        logger.info(f"Best AUC: {auc}")
    return auc

@task
def train_xgboost_task(
    features_path=os.path.join(DATA_DIR, "features/issues_features_full_plus_increment.parquet"),
    params_path=os.path.join(DATA_DIR, "params/best_params.json"),
    model_out=os.path.join(MODEL_DIR, "latest_model.json")
):
    """
    Train the XGBoost model and save both the latest and historical models. 
    Returns training accuracy/AUC for logging reference.
    """
    acc = train_xgboost(features_path, params_path, model_out)
    logger = get_run_logger()
    logger.info(f"[MODEL] Train accuracy: {acc:.4f}")

    return acc

@task
def upload_model_to_s3_task(
    local_model_file=MODEL_FILE,
    bucket_name=BUCKET_NAME,
    s3_key=S3_KEY,
    with_history=True
):
    upload_model_to_s3(local_model_file, bucket_name, s3_key, with_history)

@flow
def main_flow(date=None, flow_latency_threshold=900):  
    logger = get_run_logger()
    start = time.time()

    fetch_closed_issues_task(date)
    generate_features_task(date)
    merge_features_task()
    search_best_params_task()
    train_xgboost_task()
    upload_model_to_s3_task()

    duration = time.time() - start
    logger.info(f"[FLOW-TIMING] main_flow total duration: {duration:.2f} seconds")

    if duration > flow_latency_threshold:
        logger.warning(f"[FLOW-ALERT] Total flow duration exceeded {flow_latency_threshold}s: {duration:.2f}s")
