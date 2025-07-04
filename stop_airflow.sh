#!/bin/bash

pkill -f "airflow scheduler"
pkill -f "airflow webserver"

echo "Airflow scheduler and webserver are shut down"
