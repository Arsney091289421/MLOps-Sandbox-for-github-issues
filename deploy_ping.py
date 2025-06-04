from prefect import flow
from prefect.deployments import DeploymentSpec
from prefect.server.schemas.schedules import CronSchedule
from test_flow import write_test

DeploymentSpec(
    flow=write_test,
    name="ping-deploy",
    schedule=CronSchedule(cron="* * * * *", timezone="America/Toronto"),
    tags=["local"],
)
