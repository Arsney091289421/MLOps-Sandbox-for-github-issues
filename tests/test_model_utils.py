import os
import json
import pandas as pd
import pytest

from utils import model_utils

@pytest.fixture
def mock_features_parquet(tmp_path):
    df = pd.DataFrame({
        "title_len": [10, 20, 12, 25, 18, 14],
        "body_len": [30, 40, 35, 50, 31, 44],
        "num_labels": [2, 1, 2, 3, 1, 2],
        "has_bug_label": [1, 0, 1, 0, 1, 0],
        "hour_created": [12, 15, 10, 13, 9, 8],
        "comments": [0, 3, 2, 1, 0, 4],
        "closed_within_7_days": [1, 0, 1, 1, 0, 0],
        "number": [101, 102, 103, 104, 105, 106]
    })
    features_path = tmp_path / "mock_features.parquet"
    df.to_parquet(features_path)
    return str(features_path)


@pytest.fixture
def mock_params_json(tmp_path):
    params = {
        "learning_rate": 0.1,
        "max_depth": 3,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "n_estimators": 10,
        "random_state": 42
    }
    params_path = tmp_path / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(params, f)
    return str(params_path)

def test_train_xgboost(tmp_path, mock_features_parquet, mock_params_json):
    model_out = str(tmp_path / "model.json")
    model_utils.train_xgboost(
        features_path=mock_features_parquet,
        params_path=mock_params_json,
        model_out=model_out
    )
    # Check if the main model file is written
    assert os.path.exists(model_out)
    # Check if at least one history file exists
    hist_dir = tmp_path / "history"
    hist_files = list(hist_dir.glob("model_*.json"))
    assert len(hist_files) > 0

def test_search_best_params(tmp_path, mock_features_parquet):
    model_dir = tmp_path / "model_dir"
    param_dir = tmp_path / "param_dir"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(param_dir, exist_ok=True)
    # Use n_trials=2 to speed up the test
    model_utils.search_best_params(
        feature_path=mock_features_parquet,
        n_trials=2,
        model_dir=str(model_dir),
        param_dir=str(param_dir)
    )
    # Check if the main parameter file and history snapshots are written
    assert (model_dir / "best_params.json").exists()
    assert (param_dir / "best_params.json").exists()
    snap_files = list(param_dir.glob("best_params_*.json"))
    assert len(snap_files) > 0
