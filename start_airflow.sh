#!/bin/bash

source ~/Documents/projects/MLOps-Sandbox-for-github-issues/.venv-airflow/bin/activate

nohup airflow scheduler > scheduler.log 2>&1 &

nohup airflow webserver -p 8080 > webserver.log 2>&1 &

echo "Airflow scheduler and webserver started."
echo "Logs at scheduler.log and webserver.log"