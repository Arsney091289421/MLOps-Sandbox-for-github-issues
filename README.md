
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
- Integrated CI pipeline for automated testing and validation
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