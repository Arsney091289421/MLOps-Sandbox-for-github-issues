import os
import pandas as pd
from github import Github
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import argparse

# Load .env file
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO_NAME = "huggingface/transformers"
DATA_DIR = "/Users/daniel/Documents/projects/data"

def fetch_closed_issues(since=None, until=None, save_path=None):
    """
    Fetch closed issues from a GitHub repository.
    Supports full extraction and time window filtering.
    Uses UTC timezone for all timestamps. Shows progress during fetch.
    """
    g = Github(GITHUB_TOKEN)
    repo = g.get_repo(REPO_NAME)

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

def run_incremental(target_date=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    if target_date is None:
        # Default to fetching yesterday's closed issues
        target = (datetime.utcnow() - timedelta(days=1)).date()
    else:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
    since = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc)
    until = since + timedelta(days=1)
    out_file = f"{DATA_DIR}/issues_closed_{target}.parquet"

    if os.path.exists(out_file):
        print(f"{out_file} already exists. Skipping fetch.")
        return
    print(f"Fetching closed issues from {since} to {until} ...")
    df = fetch_closed_issues(since=since, until=until, save_path=out_file)
    print(f"Number of issues fetched: {len(df)}")

def run_full_backfill():
    os.makedirs(DATA_DIR, exist_ok=True)
    out_file = f"{DATA_DIR}/issues_closed_full.parquet"
    if os.path.exists(out_file):
        print(f"{out_file} already exists. Skipping full fetch.")
        return
    print("Fetching all closed issues ...")
    df = fetch_closed_issues(save_path=out_file)
    print(f"Number of issues fetched: {len(df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental", help="Fetch full dataset or incremental update")
    parser.add_argument("--date", type=str, help="Target date in YYYY-MM-DD format (only used in incremental mode)")
    args = parser.parse_args()
    if args.mode == "full":
        run_full_backfill()
    else:
        run_incremental(args.date)
