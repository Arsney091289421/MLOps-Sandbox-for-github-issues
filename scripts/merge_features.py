import os
from dotenv import load_dotenv
from utils.data_utils import merge_features

load_dotenv()
DATA_DIR = os.getenv("DATA_BASE_DIR", "./data")
FEATURE_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

if __name__ == "__main__":
    merge_features(FEATURE_DIR, "issues_features_full_plus_increment.parquet")
