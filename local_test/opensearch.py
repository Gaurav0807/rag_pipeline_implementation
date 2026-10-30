import boto3
import json
from requests_aws4auth import AWS4Auth
from requests import Session

REGION = "us-east-1"
INDEX_NAME = "rag-pipeline-sandbox-index"
COLLECTION_ENDPOINT = "https://cpuvvwedvaylmxenz0q6.us-east-1.aoss.amazonaws.com"

credentials = boto3.Session().get_credentials().get_frozen_credentials()
auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    REGION,
    "aoss",
    session_token=credentials.token,
)

session = Session()

# Get 1 document with ALL fields (including vector)
search_body = {
    "size": 1
}

response = session.post(
    f"{COLLECTION_ENDPOINT}/{INDEX_NAME}/_search",
    auth=auth,
    json=search_body,
    headers={"Content-Type": "application/json"},
)

results = response.json()
doc = results["hits"]["hits"][0]["_source"]

print("=== FULL DOCUMENT IN OPENSEARCH ===\n")

print("1. VECTOR (first 20 numbers out of 1024):")
print(doc["AMAZON_BEDROCK_VECTOR"][:])
print(f"   ... total numbers: {len(doc['AMAZON_BEDROCK_VECTOR'])}")

print("\n2. TEXT:")
print(doc["AMAZON_BEDROCK_TEXT"][:])

print("\n3. METADATA:")
print(doc["AMAZON_BEDROCK_METADATA"])