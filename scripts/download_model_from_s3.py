import os
from dotenv import load_dotenv
from utils.s3_utils import download_model_from_s3

# Load .env variables
load_dotenv()

if __name__ == "__main__":
    BUCKET_NAME = os.getenv("MODEL_BUCKET")
    MODEL_S3_KEY = "model/latest_model.json"
    MODEL_DIR = os.getenv("MODEL_DIR", "/home/ec2-user/mlops-api/model")
    LOCAL_FILENAME = "latest_model.json"

    download_model_from_s3(BUCKET_NAME, MODEL_S3_KEY, MODEL_DIR, LOCAL_FILENAME)
