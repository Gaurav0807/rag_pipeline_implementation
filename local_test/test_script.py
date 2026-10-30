import json
import random
import uuid
import boto3
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
TOTAL_JIRA_TICKETS = 50
TOTAL_CONFLUENCE_PAGES = 20

BUCKET_NAME = "gaurav-hudi-data"
S3_PREFIX = "rag_test_data"

s3 = boto3.client("s3")


projects = ["ENG", "DATA", "PLATFORM", "OPS"]

ticket_titles = [
    "Kafka consumer lag",
    "Spark job failure",
    "S3 permission denied",
    "Airflow DAG issue",
    "EMR cluster timeout",
    "Glue crawler failed",
    "High API latency",
    "OpenSearch indexing issue",
    "Lambda deployment failed",
    "Kubernetes pod crash"
]

descriptions = [
    "Issue observed during peak production load.",
    "System intermittently fails under high traffic.",
    "Root cause investigation is ongoing.",
    "Temporary workaround has been applied.",
    "Need optimization for better scalability.",
    "Monitoring alerts triggered multiple times.",
    "Customer impact reported by support team.",
]

comments = [
    "Restarted the service and verified logs.",
    "Scaling improved the performance.",
    "Added retries to reduce failures.",
    "Issue reproduced in staging environment.",
    "Memory utilization was unusually high.",
    "Need follow-up with infrastructure team.",
]

labels = [
    "kafka",
    "spark",
    "airflow",
    "production",
    "critical",
    "aws",
    "bug",
    "performance",
    "monitoring"
]

confluence_titles = [
    "Kafka Architecture Guide",
    "EMR Deployment Steps",
    "Airflow Best Practices",
    "Spark Optimization Guide",
    "Disaster Recovery Process",
    "OpenSearch Troubleshooting",
    "Data Pipeline Standards",
    "Kubernetes Deployment Guide"
]



def random_date():
    days_ago = random.randint(1, 365)
    return (datetime.now() - timedelta(days=days_ago)).isoformat()

def upload_to_s3(key, data):
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=json.dumps(data, indent=2),
        ContentType="application/json"
    )

    print(f"Uploaded: s3://{BUCKET_NAME}/{key}")


def generate_jira_ticket(index):
    project = random.choice(projects)

    return {
        "ticket_id": f"{project}-{1000 + index}",
        "summary": random.choice(ticket_titles),
        "description": random.choice(descriptions),
        "status": random.choice(
            ["OPEN", "IN_PROGRESS", "DONE", "BLOCKED"]
        ),
        "priority": random.choice(
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        ),
        "assignee": f"user{random.randint(1,20)}",
        "comments": random.sample(comments, k=3),
        "labels": random.sample(labels, k=3),
        "created_at": random_date(),
        "updated_at": random_date(),
        "uuid": str(uuid.uuid4())
    }


# ==========================================
# CONFLUENCE DATA GENERATOR
# ==========================================
def generate_confluence_page(index):

    title = random.choice(confluence_titles)

    return {
        "page_id": str(uuid.uuid4()),
        "title": title,
        "space": random.choice(
            ["Engineering", "DataPlatform", "Operations"]
        ),
        "content": f"""
# {title}

This document explains internal platform processes.

## Overview
The system is designed for scalability and reliability.

## Architecture
Services communicate using Kafka and Spark pipelines.

## Monitoring
CloudWatch and Grafana are used for observability.

## Troubleshooting
Check logs and metrics before escalating incidents.
        """,
        "author": f"user{random.randint(1,10)}",
        "created_at": random_date(),
        "labels": random.sample(labels, k=2)
    }


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":

    # -----------------------------
    # Upload Jira Data
    # -----------------------------
    for i in range(TOTAL_JIRA_TICKETS):

        ticket = generate_jira_ticket(i)

        s3_key = (
            f"{S3_PREFIX}/jira/"
            f"{ticket['ticket_id']}.json"
        )

        upload_to_s3(s3_key, ticket)

    # -----------------------------
    # Upload Confluence Data
    # -----------------------------
    for i in range(TOTAL_CONFLUENCE_PAGES):

        page = generate_confluence_page(i)

        s3_key = (
            f"{S3_PREFIX}/confluence/"
            f"page_{i+1}.json"
        )

        upload_to_s3(s3_key, page)

    print("\n All random test data uploaded to S3")