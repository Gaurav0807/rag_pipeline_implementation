import boto3
import json
from requests_aws4auth import AWS4Auth
from requests import Session

REGION = "us-east-1"
INDEX_NAME = "rag-pipeline-sandbox-index"
COLLECTION_ENDPOINT = "https://cpuvvwedvaylmxenz0q6.us-east-1.aoss.amazonaws.com"

# Use default credentials from AWS CLI profile
session = boto3.Session()
credentials = session.get_credentials().get_frozen_credentials()
AWS_ACCESS_KEY_ID = credentials.access_key
AWS_SECRET_ACCESS_KEY = credentials.secret_key
AWS_SESSION_TOKEN = credentials.token




auth = AWS4Auth(
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    REGION,
    "aoss",
    session_token=AWS_SESSION_TOKEN,
)

index_body = {
    "settings": {
        "index": {
            "knn": True
        }
    },
    "mappings": {
        "properties": {
            "AMAZON_BEDROCK_VECTOR": {
                "type": "knn_vector",
                "dimension": 1024,
                "method": {
                    "name": "hnsw",
                    "engine": "faiss",
                    "space_type": "l2",
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            },
            "AMAZON_BEDROCK_TEXT": {
                "type": "text",
                "index": True
            },
            "AMAZON_BEDROCK_METADATA": {
                "type": "text",
                "index": False
            }
        }
    }
}

session = Session()
url = f"{COLLECTION_ENDPOINT}/{INDEX_NAME}"

print(f"Creating index: {INDEX_NAME}")
response = session.put(
    url,
    auth=auth,
    json=index_body,
    headers={"Content-Type": "application/json"},
)

print(f"Status: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")