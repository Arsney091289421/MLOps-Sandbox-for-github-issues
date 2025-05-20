import os
import pandas as pd
import xgboost as xgb
import json
from dotenv import load_dotenv
import argparse
from datetime import datetime

def train_xgboost(features_path, params_path, model_out):
    # Load features
    df = pd.read_parquet(features_path)
    print(f"[INFO] Loaded features: {features_path}, shape: {df.shape}")

    # Load best params
    with open(params_path, "r") as f:
        best_params = json.load(f)
    print(f"[INFO] Using best params from: {params_path}")
    # Remove n_estimators from xgb params if exists
    n_estimators = best_params.pop("n_estimators", 200)

    # Prepare data
    X = df.drop(["closed_within_7_days", "number"], axis=1)
    y = df["closed_within_7_days"]

    # Train XGBoost
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="auc",
        n_estimators=n_estimators,
        **best_params
    )
    model.fit(X, y)
    print(f"[DONE] Model trained, AUC on train: {model.score(X, y):.4f}")

    # Save latest model
    model.save_model(model_out)  

    # Save history snapshot
    history_dir = os.path.join(os.path.dirname(model_out), "history")
    os.makedirs(history_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_model_path = os.path.join(history_dir, f"model_{timestamp}.json")
    model.save_model(history_model_path)

    print(f"[DONE] Model saved to: {model_out}")
    print(f"[SNAPSHOT] Historical model saved to: {history_model_path}")

if __name__ == "__main__":
    load_dotenv()

    # Paths
    DATA_BASE_DIR = os.getenv("DATA_BASE_DIR", "./data")
    FEATURE_DIR = os.path.join(DATA_BASE_DIR, "features")
    MODEL_DIR = os.getenv("MODEL_DIR", "./model")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # File defaults
    BEST_PARAMS_PATH = os.path.join(DATA_BASE_DIR, "params", "best_params.json")
    DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "latest_model.json")
    DEFAULT_FEATURES_PATH = os.path.join(FEATURE_DIR, "issues_features_full_plus_increment.parquet")
    FALLBACK_FEATURES_PATH = os.path.join(FEATURE_DIR, "issues_features_full.parquet")

    # Automatically fallback to the full features file
    if not os.path.exists(DEFAULT_FEATURES_PATH):
        if os.path.exists(FALLBACK_FEATURES_PATH):
            print(f"[WARN] {DEFAULT_FEATURES_PATH} not found. Using {FALLBACK_FEATURES_PATH} instead.")
            DEFAULT_FEATURES_PATH = FALLBACK_FEATURES_PATH
        else:
            raise FileNotFoundError(f"Neither {DEFAULT_FEATURES_PATH} nor {FALLBACK_FEATURES_PATH} exists!")

    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=DEFAULT_FEATURES_PATH,
                        help=f"Features file (default: {DEFAULT_FEATURES_PATH})")
    parser.add_argument("--params", type=str, default=BEST_PARAMS_PATH,
                        help=f"Best params JSON (default: {BEST_PARAMS_PATH})")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_PATH,
                        help=f"Output model file (default: {DEFAULT_MODEL_PATH})")
    args = parser.parse_args()

    train_xgboost(
        features_path=args.features,
        params_path=args.params,
        model_out=args.output
    )
