import os
import json
import pandas as pd
from datetime import datetime
import optuna
import xgboost as xgb

# search_best_params.py

def load_config(config_path):
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
    return 1.0 - best_auc

def search_best_params(feature_path, n_trials, model_dir, param_dir):
    X, y = load_data(feature_path)
    study = optuna.create_study(direction="minimize")
    print(f"[INFO] Hyperparameter search with {n_trials} trials ...")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("Best params:", study.best_params)
    auc = 1.0 - study.best_value
    print("Best AUC:", auc)

    # Save best params
    best_params_path = os.path.join(model_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"[SAVE] models/best_params.json saved: {best_params_path}")

    # Save to params archive
    param_save_path = os.path.join(param_dir, "best_params.json")
    with open(param_save_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"[SAVE] data/params/best_params.json saved: {param_save_path}")

    # Save snapshot with timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_path = os.path.join(param_dir, f"best_params_{ts}.json")
    with open(hist_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"[SNAPSHOT] params snapshot: {hist_path}")

    return auc

# train_model.py

def train_xgboost(features_path, params_path, model_out):
    # Load features
    df = pd.read_parquet(features_path)
    print(f"[INFO] Loaded features: {features_path}, shape: {df.shape}")

    # Load best params
    with open(params_path, "r") as f:
        best_params = json.load(f)
    print(f"[INFO] Using best params from: {params_path}")
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
    acc = model.score(X, y)
    print(f"[DONE] Model trained, ACC on train: {acc:.4f}")

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

    return acc
