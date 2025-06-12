#!/bin/bash
set -e

echo "Step 1: Fetching all closed issues (full backfill)..."
python scripts/fetch_closed_issues.py --mode full

echo "Step 2: Generating all features (full mode)..."
python scripts/generate_features.py --mode full

echo "Full initialization complete."
