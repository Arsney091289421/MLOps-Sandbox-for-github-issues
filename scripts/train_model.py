import os
import argparse
from dotenv import load_dotenv
from utils.model_utils import train_xgboost

load_dotenv()

DATA_BASE_DIR = os.getenv("DATA_BASE_DIR", "./data")
FEATURE_DIR = os.path.join(DATA_BASE_DIR, "features")
MODEL_DIR = os.getenv("MODEL_DIR", "./model")
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_PARAMS_PATH = os.path.join(DATA_BASE_DIR, "params", "best_params.json")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.json")
DEFAULT_FEATURES_PATH = os.path.join(FEATURE_DIR, "issues_features_full_plus_increment.parquet")
FALLBACK_FEATURES_PATH = os.path.join(FEATURE_DIR, "issues_features_full.parquet")

if not os.path.exists(DEFAULT_FEATURES_PATH):
    if os.path.exists(FALLBACK_FEATURES_PATH):
        print(f"[WARN] {DEFAULT_FEATURES_PATH} not found. Using {FALLBACK_FEATURES_PATH} instead.")
        DEFAULT_FEATURES_PATH = FALLBACK_FEATURES_PATH
    else:
        raise FileNotFoundError(f"Neither {DEFAULT_FEATURES_PATH} nor {FALLBACK_FEATURES_PATH} exists!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=DEFAULT_FEATURES_PATH, help=f"Features file (default: {DEFAULT_FEATURES_PATH})")
    parser.add_argument("--params", type=str, default=BEST_PARAMS_PATH, help=f"Best params JSON (default: {BEST_PARAMS_PATH})")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_PATH, help=f"Output model file (default: {DEFAULT_MODEL_PATH})")
    args = parser.parse_args()

    train_xgboost(
        features_path=args.features,
        params_path=args.params,
        model_out=args.output
    )
