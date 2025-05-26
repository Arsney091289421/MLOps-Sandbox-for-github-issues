import os
import boto3
from datetime import datetime

def download_model_from_s3(bucket_name, s3_key, local_dir, local_filename="latest_model.json"):
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, local_filename)
    s3 = boto3.client("s3")
    s3.download_file(bucket_name, s3_key, local_path)
    print(f"[DONE] Downloaded s3://{bucket_name}/{s3_key} → {local_path}")
    return local_path

def upload_model_to_s3(local_model_file, bucket_name, s3_key, with_history=True):
    s3 = boto3.client("s3")
    # Upload main file
    s3.upload_file(local_model_file, bucket_name, s3_key)
    print(f"[UPLOAD] {local_model_file} → s3://{bucket_name}/{s3_key}")

    # Snapshot archive
    if with_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_key = f"model/history/model_{timestamp}.json"
        s3.upload_file(local_model_file, bucket_name, history_key)
        print(f"[SNAPSHOT] Archived as s3://{bucket_name}/{history_key}")
