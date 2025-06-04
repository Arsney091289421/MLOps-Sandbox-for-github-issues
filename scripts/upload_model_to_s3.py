import os
from dotenv import load_dotenv
from utils.s3_utils import upload_model_to_s3

# Load env
load_dotenv()

MODEL_DIR = os.getenv("MODEL_DIR", "./models")
MODEL_FILE = os.path.join(MODEL_DIR, "latest_model.json")
BUCKET_NAME = os.getenv("MODEL_BUCKET")
S3_KEY = "model/latest_model.json"

if __name__ == "__main__":
    upload_model_to_s3(MODEL_FILE, BUCKET_NAME, S3_KEY, with_history=True)
    