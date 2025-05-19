import os
import json
import argparse
from datetime import datetime

import pandas as pd
import optuna
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

# Load .env file

DATA_BASE_DIR = os.getenv("DATA_BASE_DIR", "./data")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")
PARAM_DIR = os.path.join(DATA_BASE_DIR, "params")
CONFIG_PATH = os.getenv("CONFIG_PATH", "config.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PARAM_DIR, exist_ok=True)

def load_config(config_path=CONFIG_PATH):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return json.load(f)
    return {}

def load_data(feature_path):
    df = pd.read_parquet(feature_path)
    X = df.drop(columns=["closed_within_7_days", "number"], errors="ignore")
    y = df["closed_within_7_days"]
    return X, y

def objective(trial, X, y):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'tree_method': 'hist',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
        'n_estimators': trial.suggest_int('n_estimators', 100, 600),
        'random_state': 42,
    }
    dtrain = xgb.DMatrix(X, label=y)
    cv_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=param["n_estimators"],
        nfold=3,
        stratified=True,
        early_stopping_rounds=10,
        metrics="auc",
        seed=42,
        verbose_eval=False,
        shuffle=True
    )
    best_auc = cv_results["test-auc-mean"].max()
    return 1.0 - best_auc  # minimize (Optuna tries to minimize objective)

def main(feature_path, n_trials):
    X, y = load_data(feature_path)
    study = optuna.create_study(direction="minimize")
    print(f"[INFO] Hyperparameter search with {n_trials} trials ...")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("Best params:", study.best_params)
    print("Best AUC:", 1.0 - study.best_value)

    # Save best params
    best_params_path = os.path.join(MODEL_DIR, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"[SAVE] models/best_params.json saved: {best_params_path}")

    # Save to params archive
    param_save_path = os.path.join(PARAM_DIR, "best_params.json")
    with open(param_save_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"[SAVE] data/params/best_params.json saved: {param_save_path}")

    # Save snapshot with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_path = os.path.join(PARAM_DIR, f"best_params_{ts}.json")
    with open(hist_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"[SNAPSHOT] params snapshot: {hist_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=os.path.join(DATA_BASE_DIR, "features/issues_features_full.parquet"), help="Path to features parquet")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Config JSON for n_trials")
    args = parser.parse_args()

    # Use config.json to adjust n_trials
    config = load_config(args.config)
    n_trials = config.get("n_trials", 30)

    main(args.features, n_trials)
