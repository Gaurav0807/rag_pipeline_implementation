import boto3
import time

bedrock_agent = boto3.client("bedrock-agent",region_name="us-east-1")

KB_ID= "WAQARJ0KGQ"
data_source_id = "VM6KCWRH9B"

# data_source_id = "VM6KCWRH9B,WAQARJ0KGQ"
# knowledge_base_id = "WAQARJ0KGQ"


response = bedrock_agent.start_ingestion_job(
      knowledgeBaseId=KB_ID,
      dataSourceId=data_source_id
)

job_id = response["ingestionJob"]["ingestionJobId"]
print(f"Ingestion started: {job_id}")

while True:
    status = bedrock_agent.get_ingestion_job(
        knowledgeBaseId=KB_ID,
        dataSourceId=data_source_id,
        ingestionJobId=job_id
    )
    current = status["ingestionJob"]["status"]
    print(f"Status: {current}")

    if current in ["COMPLETE", "FAILED"]:
        stats = status["ingestionJob"].get("statistics", {})
        print(f"\nResults:")
        print(f"  Scanned: {stats.get('numberOfDocumentsScanned', 0)}")
        print(f"  Indexed: {stats.get('numberOfNewDocumentsIndexed', 0)}")
        print(f"  Failed:  {stats.get('numberOfDocumentsFailed', 0)}")
        break

    time.sleep(10)