import os
from dotenv import load_dotenv
import argparse
from utils.data_utils import run_incremental, run_full_backfill

# Load .env file
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "huggingface/transformers"
DATA_DIR = os.getenv("DATA_BASE_DIR", "./data")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental", help="Fetch full dataset or incremental update")
    parser.add_argument("--date", type=str, help="Target date in YYYY-MM-DD format (only used in incremental mode)")
    args = parser.parse_args()

    if args.mode == "full":
        run_full_backfill(GITHUB_TOKEN, REPO_NAME, DATA_DIR)
    else:
        run_incremental(GITHUB_TOKEN, REPO_NAME, DATA_DIR, args.date)
