import os
import argparse
from dotenv import load_dotenv
from utils.model_utils import load_config, search_best_params

load_dotenv()

DATA_BASE_DIR = os.getenv("DATA_BASE_DIR", "./data")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
PARAM_DIR = os.path.join(DATA_BASE_DIR, "params")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAM_DIR, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=os.path.join(DATA_BASE_DIR, "features/issues_features_full.parquet"), help="Path to features parquet")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Config JSON for n_trials")
    args = parser.parse_args()

    config = load_config(args.config)
    n_trials = config.get("n_trials", 30)

    search_best_params(args.features, n_trials, MODEL_DIR, PARAM_DIR)
