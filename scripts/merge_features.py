import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.getenv("DATA_BASE_DIR", "./data")
FEATURE_DIR = os.path.join(BASE_DIR, "features")

# Merge the full dataset and all incremental files newer than the full one
def merge_features(output_name="issues_features_merged.parquet"):
    files = [f for f in os.listdir(FEATURE_DIR) if f.endswith(".parquet")]
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
        full_path = os.path.join(FEATURE_DIR, full_file)
        full_df = pd.read_parquet(full_path)
        dfs.append(full_df)
        # Only merge incremental files that contain data newer than the full dataset
        # Find the maximum "number" in the full dataset
        try:
            full_latest = full_df["number"].max()  
        except:
            full_latest = None
        if full_latest:
            for dfname in daily_files:
                # For incremental files, exclude entries already in the full dataset
                daily_df = pd.read_parquet(os.path.join(FEATURE_DIR, dfname))
                # Only add rows whose "number" is not already in the full dataset (to avoid duplicates)
                if "number" in daily_df.columns:
                    daily_df = daily_df[~daily_df["number"].isin(full_df["number"])]
                if not daily_df.empty:
                    dfs.append(daily_df)
    else:
        print("[WARN] No full features file found, merging all incrementals.")
        for dfname in daily_files:
            daily_df = pd.read_parquet(os.path.join(FEATURE_DIR, dfname))
            dfs.append(daily_df)

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["number"])  # Avoid duplicate issues

    out_path = os.path.join(FEATURE_DIR, output_name)
    merged.to_parquet(out_path, index=False)
    print(f"[DONE] Merged features saved to {out_path}. Shape: {merged.shape}")

if __name__ == "__main__":
    merge_features("issues_features_full_plus_increment.parquet")
