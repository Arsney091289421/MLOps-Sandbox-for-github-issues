import os
import pandas as pd
import pytest

from utils import data_utils

@pytest.fixture
def mock_raw_parquet(tmp_path):
    df = pd.DataFrame([
        {
            "number": 1,
            "body": "first bug",
            "title": "fix needed",
            "labels": ["bug"],
            "created_at": pd.Timestamp("2024-05-01T01:00:00Z"),
            "closed_at": pd.Timestamp("2024-05-03T12:00:00Z"),
            "comments": 1,
        },
        {
            "number": 2,
            "body": "",
            "title": "add feature",
            "labels": ["enhancement"],
            "created_at": pd.Timestamp("2024-05-02T10:00:00Z"),
            "closed_at": pd.Timestamp("2024-05-15T11:00:00Z"),
            "comments": 2,
        },
    ])
    p = tmp_path / "issues_closed_full.parquet"
    df.to_parquet(p)
    return str(p)

def test_extract_features_basic():
    row = {
        "body": "test",
        "title": "hello",
        "labels": ["bug", "urgent"],
        "created_at": pd.Timestamp("2024-05-01T01:00:00Z"),
        "closed_at": pd.Timestamp("2024-05-05T01:00:00Z"),
        "comments": 3,
    }
    feats = data_utils.extract_features(row)
    assert feats["title_len"] == 5
    assert feats["body_len"] == 4
    assert feats["num_labels"] == 2
    assert feats["has_bug_label"] == 1
    assert feats["closed_within_7_days"] == 1

def test_generate_features(tmp_path, mock_raw_parquet):
    out_dir = tmp_path / "features"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / "issues_features_full.parquet"

    data_utils.generate_features(str(mock_raw_parquet), str(out_path))
    feats_df = pd.read_parquet(out_path)
    assert feats_df.shape[0] == 2
    assert "title_len" in feats_df.columns
    # Check that the 'number' column is preserved
    assert set(feats_df["number"]) == {1, 2}

def test_merge_features(tmp_path, mock_raw_parquet):
    # Generate two feature files, with some overlapping rows
    out_dir = tmp_path / "features"
    os.makedirs(out_dir, exist_ok=True)
    full_path = out_dir / "issues_features_full.parquet"
    data_utils.generate_features(str(mock_raw_parquet), str(full_path))

    # Copy one as an incremental file and modify the 'number' column
    inc_path = out_dir / "issues_features_2024-05-03.parquet"
    feats_df = pd.read_parquet(full_path)
    feats_df2 = feats_df.copy()
    feats_df2["number"] = [2, 3]  # 2 is duplicate, 3 is new
    feats_df2.to_parquet(inc_path)

    # Merge the feature files
    out_merge = out_dir / "merged.parquet"
    data_utils.merge_features(str(out_dir), output_name="merged.parquet")
    merged_df = pd.read_parquet(out_merge)

    # The result should contain 3 unique 'number' values
    assert set(merged_df["number"]) == {1, 2, 3}
