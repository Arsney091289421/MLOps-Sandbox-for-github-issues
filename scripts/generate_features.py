import os
import pandas as pd
from datetime import datetime, timedelta
import argparse
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE_DIR = os.getenv("DATA_BASE_DIR", "./data")  # fallback to ./data
RAW_DIR = BASE_DIR
FEATURE_DIR = os.path.join(BASE_DIR, "features")
os.makedirs(FEATURE_DIR, exist_ok=True)

def extract_features(row):
    # Fill missing text
    body = row["body"] if pd.notna(row["body"]) else ""
    title = row["title"] if pd.notna(row["title"]) else ""
    labels = row["labels"] if isinstance(row["labels"], list) else []
    created_at = row["created_at"]
    closed_at = row["closed_at"]

    return {
        "title_len": len(title),
        "body_len": len(body),
        "num_labels": len(labels),
        "has_bug_label": "bug" in labels,
        "hour_created": created_at.hour if not pd.isna(created_at) else None,
        "comments": row["comments"],
        "closed_within_7_days": (closed_at - created_at) <= timedelta(days=7) if pd.notna(closed_at) and pd.notna(created_at) else None
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
    # save original issue number
    if "number" in df.columns:
        feature_df["number"] = df["number"].values

    feature_df.to_parquet(output_path, index=False)
    print(f"[DONE] Saved features to {output_path}")
    print(feature_df.head())

def run_full():
    input_path = os.path.join(RAW_DIR, "issues_closed_full.parquet")
    output_path = os.path.join(FEATURE_DIR, "issues_features_full.parquet")
    generate_features(input_path, output_path)

def run_incremental(date_str=None):
    if date_str is None:
        target_date = (datetime.utcnow() - timedelta(days=1)).date()
    else:
        target_date = datetime.strptime(date_str, "%Y-%m-%d").date()

    input_path = os.path.join(RAW_DIR, f"issues_closed_{target_date}.parquet")
    output_path = os.path.join(FEATURE_DIR, f"issues_features_{target_date}.parquet")
    generate_features(input_path, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental", help="Run mode")
    parser.add_argument("--date", type=str, help="Target date in YYYY-MM-DD format (only used in incremental mode)")
    args = parser.parse_args()

    if args.mode == "full":
        run_full()
    else:
        run_incremental(args.date)
