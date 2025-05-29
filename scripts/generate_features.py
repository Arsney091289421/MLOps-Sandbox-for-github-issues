import os
import argparse
from dotenv import load_dotenv
from utils.data_utils import run_full_feature_generation, run_incremental_feature_generation

load_dotenv()
DATA_DIR = os.getenv("DATA_BASE_DIR", "./data")
RAW_DIR = DATA_DIR
FEATURE_DIR = os.path.join(DATA_DIR, "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental", help="Run mode")
    parser.add_argument("--date", type=str, help="Target date in YYYY-MM-DD format (only used in incremental mode)")
    args = parser.parse_args()

    if args.mode == "full":
        run_full_feature_generation(RAW_DIR, FEATURE_DIR)
    else:
        run_incremental_feature_generation(RAW_DIR, FEATURE_DIR, args.date)
