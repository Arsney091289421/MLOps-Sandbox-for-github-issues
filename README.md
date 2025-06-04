## 1. Project Overview

This repository provides a fully automated pipeline for collecting, processing, and modeling GitHub issues, with the goal of predicting whether an issue will be closed within 7 days. It is designed to run locally or in the cloud, supporting incremental data updates, automated model training, and model export to AWS S3.

> **Note:** This project works together with [mlops-serve](https://github.com/Arsney091289421/mlops-serve),  
> which handles model inference, prediction serving, and result upload on AWS EC2.  
> This repo focuses on data collection, feature engineering, and model lifecycle automation.


## 2. Features

- Automated collection of GitHub issues (supports both full and incremental modes)
- Flexible, modular feature engineering for issue data
- Incremental feature merging for seamless model updates
- Automated hyperparameter tuning and model training (XGBoost)
- Model versioning and export to AWS S3
- Workflow orchestration and monitoring with Prefect
- Integrated CI pipeline by GitHub Actions for automated testing and validation
- Easy local or cloud deployment (no Docker required)
- Comprehensive unit and integration tests for all key modules

## 3. Tech Stack

- Python 3.9
- Prefect 
- XGBoost 
- AWS S3 
- pytest, moto 
- GitHub Actions (CI)

## 4. System Architecture

![System Architecture](docs/architecture.svg)

## 5. Quick Start

### 5.1 Prerequisites

- **AWS Account**
  - An S3 bucket for storing trained models (`models/latest_model.json`)
- **GitHub Personal Access Token**
  - Only `public_repo` scope is required for collecting public issue data
- **Python 3.9 environment**
  - Can be a local machine or cloud VM (e.g., AWS EC2)
- **(Optional) Prefect UI**
  - For workflow visualization, monitoring, and alerting

> **Note:**  
> To avoid Prefect Cloud subscription fees, use `prefect server start` to run a local Prefect server for free workflow scheduling and monitoring.

---

### 5.2 Deployment

1. **Clone the repository**

   ```bash
   git clone https://github.com/Arsney091289421/MLOps-Sandbox-for-github-issues.git
   cd MLOps-Sandbox-for-github-issues
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env to match your S3 bucket, GitHub token, and local directory settings
   ```

4. **Run full data fetch and feature generation**

   ```bash
   chmod +x run_full_init.sh
   bash run_full_init.sh
   # This will download all closed issues and generate the full feature dataset.
   ```

5. **Run the main workflow**

   ```bash
   python main_flow.py
   # This will execute the entire Prefect workflow: incremental data fetch, feature engineering, feature merging, hyperparameter tuning, model training, and model export to S3.
   ```

