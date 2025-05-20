import os
import boto3
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./model")
MODEL_FILE = os.path.join(MODEL_DIR, "latest_model.json")
BUCKET_NAME = os.getenv("MODEL_BUCKET")
S3_KEY = "model/latest_model.json"  # Path to store the model on S3

def upload_model(with_history=True):
    s3 = boto3.client("s3")  # AWS credentials are automatically injected via environment variables

    # Upload the latest model (overwrite if exists)
    s3.upload_file(MODEL_FILE, BUCKET_NAME, S3_KEY)
    print(f"[UPLOAD] {MODEL_FILE} → s3://{BUCKET_NAME}/{S3_KEY}")

    # Optionally: save a historical snapshot
    if with_history:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_key = f"model/history/model_{timestamp}.json"
        s3.upload_file(MODEL_FILE, BUCKET_NAME, history_key)
        print(f"[SNAPSHOT] Archived as s3://{BUCKET_NAME}/{history_key}")

if __name__ == "__main__":
    upload_model()
