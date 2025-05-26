# utils/data_utils.py

import os
import pandas as pd
from github import Github
from datetime import datetime, timedelta, timezone

#fetch_closed_issues.py

def fetch_closed_issues(github_token, repo_name, since=None, until=None, save_path=None):
    """
    Fetch closed issues from a GitHub repository.
    Supports full extraction and time window filtering.
    Uses UTC timezone for all timestamps. Shows progress during fetch.
    """
    g = Github(github_token)
    repo = g.get_repo(repo_name)

    if since is not None:
        issues = repo.get_issues(state="closed", since=since)
    else:
        issues = repo.get_issues(state="closed")

    data = []
    for idx, issue in enumerate(issues):
        if issue.pull_request is not None:
            continue
        # Only keep issues closed before 'until'
        if until and issue.closed_at and issue.closed_at > until:
            continue
        data.append({
            "number": issue.number,
            "title": issue.title,
            "user": issue.user.login if issue.user else None,
            "created_at": issue.created_at,
            "closed_at": issue.closed_at,
            "state": issue.state,
            "labels": [label.name for label in issue.labels],
            "comments": issue.comments,
            "body": issue.body,
        })
        if (idx+1) % 100 == 0:
            print(f"Fetched {idx+1} issues ...")

    df = pd.DataFrame(data)
    if save_path:
        df.to_parquet(save_path, index=False)
        print(f"Saved to {save_path}")
    return df

def run_incremental(github_token, repo_name, data_dir, target_date=None):
    os.makedirs(data_dir, exist_ok=True)
    if target_date is None:
        # Default to fetching yesterday's closed issues
        target = (datetime.utcnow() - timedelta(days=1)).date()
    else:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
    since = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc)
    until = since + timedelta(days=1)
    out_file = f"{data_dir}/issues_closed_{target}.parquet"

    if os.path.exists(out_file):
        print(f"{out_file} already exists. Skipping fetch.")
        return
    print(f"Fetching closed issues from {since} to {until} ...")
    df = fetch_closed_issues(github_token, repo_name, since=since, until=until, save_path=out_file)
    print(f"Number of issues fetched: {len(df)}")

def run_full_backfill(github_token, repo_name, data_dir):
    os.makedirs(data_dir, exist_ok=True)
    out_file = f"{data_dir}/issues_closed_full.parquet"
    if os.path.exists(out_file):
        print(f"{out_file} already exists. Skipping full fetch.")
        return
    print("Fetching all closed issues ...")
    df = fetch_closed_issues(github_token, repo_name, save_path=out_file)
    print(f"Number of issues fetched: {len(df)}")

# generate_features.py

import os
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

def extract_features(row):
    # Fill missing text
    body = row["body"] if pd.notna(row["body"]) else ""
    title = row["title"] if pd.notna(row["title"]) else ""
    labels = row["labels"] if isinstance(row["labels"], list) else []
    created_at = row["created_at"]
    closed_at = row["closed_at"]

    has_bug_label = int("bug" in labels)
    closed_within_7_days = (
        int((closed_at - created_at) <= timedelta(days=7))
        if pd.notna(closed_at) and pd.notna(created_at) else 0
    )

    return {
        "title_len": len(title),
        "body_len": len(body),
        "num_labels": len(labels),
        "has_bug_label": has_bug_label,
        "hour_created": created_at.hour if not pd.isna(created_at) else None,
        "comments": row["comments"],
        "closed_within_7_days": closed_within_7_days
    }

def generate_features(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"[SKIP] Input file {input_path} does not exist.")
        return
    if os.path.exists(output_path):
        print(f"[SKIP] Output file {output_path} already exists.")
        return

    df = pd.read_parquet(input_path)

    feature_rows = []
    print(f"[INFO] Extracting features from {len(df)} rows...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating features"):
        feats = extract_features(row)
        feature_rows.append(feats)

    feature_df = pd.DataFrame(feature_rows)
    if "number" in df.columns:
        feature_df["number"] = df["number"].values
    feature_df["closed_within_7_days"] = feature_df["closed_within_7_days"].astype(int)
    feature_df["has_bug_label"] = feature_df["has_bug_label"].astype(int)
    feature_df.to_parquet(output_path, index=False)
    print(f"[DONE] Saved features to {output_path}")
    print(feature_df.head())

def run_full_feature_generation(raw_dir, feature_dir):
    input_path = os.path.join(raw_dir, "issues_closed_full.parquet")
    output_path = os.path.join(feature_dir, "issues_features_full.parquet")
    generate_features(input_path, output_path)

def run_incremental_feature_generation(raw_dir, feature_dir, date_str=None):
    if date_str is None:
        target_date = (datetime.utcnow() - timedelta(days=1)).date()
    else:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    input_path = os.path.join(raw_dir, f"issues_closed_{target_date}.parquet")
    output_path = os.path.join(feature_dir, f"issues_features_{target_date}.parquet")
    generate_features(input_path, output_path)

# merge_features.py

def merge_features(feature_dir, output_name="issues_features_merged.parquet"):
    files = [f for f in os.listdir(feature_dir) if f.endswith(".parquet")]
    if not files:
        print("[ERROR] No parquet files found in features directory.")
        return

    # Prefer using the full dataset if available
    full_file = None
    daily_files = []
    for f in files:
        if f == "issues_features_full.parquet":
            full_file = f
        elif f.startswith("issues_features_"):
            daily_files.append(f)

    dfs = []
    if full_file:
        print(f"[INFO] Found full features: {full_file}")
        full_path = os.path.join(feature_dir, full_file)
        full_df = pd.read_parquet(full_path)
        dfs.append(full_df)
        try:
            full_latest = full_df["number"].max()  
        except:
            full_latest = None
        if full_latest:
            for dfname in daily_files:
                daily_df = pd.read_parquet(os.path.join(feature_dir, dfname))
                if "number" in daily_df.columns:
                    daily_df = daily_df[~daily_df["number"].isin(full_df["number"])]
                if not daily_df.empty:
                    dfs.append(daily_df)
    else:
        print("[WARN] No full features file found, merging all incrementals.")
        for dfname in daily_files:
            daily_df = pd.read_parquet(os.path.join(feature_dir, dfname))
            dfs.append(daily_df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["number"])

    out_path = os.path.join(feature_dir, output_name)
    merged.to_parquet(out_path, index=False)
    print(f"[DONE] Merged features saved to {out_path}. Shape: {merged.shape}")
