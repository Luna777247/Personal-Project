# deploy.py
from flows.crawl_vnexpress import crawl_vnexpress_disaster
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

# Deploy with daily schedule at 7 AM
deployment = Deployment.build_from_flow(
    flow=crawl_vnexpress_disaster,
    name="daily-disaster-crawl",
    schedule=CronSchedule(cron="0 7 * * *"),  # 7h sáng mỗi ngày
)

if __name__ == "__main__":
    deployment.apply()