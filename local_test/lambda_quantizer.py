"""
Lambda function to intercept Bedrock KB → OpenSearch flow and quantize vectors
Deploy this to intercept ingestion WITHOUT changing KB or data source config
"""

import json
import boto3
import numpy as np
from typing import Dict, List, Tuple

# AWS clients
opensearch_client = boto3.client("opensearchserverless")
s3_client = boto3.client("s3")


class VectorQuantizer:
    """Quantize float32 vectors to int8 before storing in OpenSearch"""

    @staticmethod
    def quantize(vector: List[float]) -> Tuple[List[int], Dict]:
        """Convert float32 → int8 with metadata for dequantization"""
        vec = np.array(vector, dtype=np.float32)

        min_val = float(vec.min())
        max_val = float(vec.max())

        # Normalize to [0, 1] then scale to [0, 255]
        if max_val == min_val:
            quantized = np.zeros_like(vec, dtype=np.int8)
        else:
            normalized = (vec - min_val) / (max_val - min_val)
            scaled = (normalized * 255).astype(np.int8)
            quantized = scaled

        metadata = {"min": min_val, "max": max_val, "dtype": "int8"}

        return quantized.tolist(), metadata

    @staticmethod
    def dequantize(quantized: List[int], metadata: Dict) -> List[float]:
        """Convert int8 → float32 for search"""
        quantized_arr = np.array(quantized, dtype=np.int8)
        min_val = metadata["min"]
        max_val = metadata["max"]

        # Reverse: [0, 255] → [0, 1] → [min, max]
        normalized = quantized_arr.astype(np.float32) / 255.0
        recovered = normalized * (max_val - min_val) + min_val

        return recovered.tolist()


def lambda_handler(event, context):
    """
    Triggered when: Bedrock KB finishes ingesting and uploads vectors to OpenSearch

    Flow:
    1. Bedrock KB generates vectors (float32)
    2. EventBridge triggers this Lambda
    3. This Lambda reads from OpenSearch temp index
    4. Quantizes vectors (float32 → int8)
    5. Re-indexes to optimized OpenSearch collection
    """

    print("🔄 Vector Quantization Lambda Started")

    KB_COLLECTION_ENDPOINT = event.get("collection_endpoint")
    INDEX_NAME = event.get("index_name", "rag-pipeline-sandbox-index")
    BATCH_SIZE = 100

    try:
        # Step 1: Read float32 vectors from OpenSearch (Bedrock KB indexed them)
        print(f"📖 Reading vectors from index: {INDEX_NAME}")

        # Query all documents with their vectors
        query = {"query": {"match_all": {}}, "size": BATCH_SIZE}

        # Step 2: Process in batches
        quantized_count = 0
        storage_saved = 0

        all_docs = []  # Collect all docs
        scroll_id = None

        # Scroll through all documents
        from opensearchpy import OpenSearch
        from opensearchpy.connection.http_urllib3 import URLLIB3HttpConnection

        # Create unsigned request
        # (In production, use AWS auth signing)
        client = OpenSearch(hosts=[{"host": KB_COLLECTION_ENDPOINT, "port": 443}])

        # Search all documents
        response = client.search(index=INDEX_NAME, body=query, scroll="5m")

        while len(response["hits"]["hits"]) > 0:
            for doc in response["hits"]["hits"]:
                all_docs.append(doc)

                # Step 3: Quantize the vector
                float_vector = doc["_source"]["AMAZON_BEDROCK_VECTOR"]
                quantized_vector, metadata = VectorQuantizer.quantize(float_vector)

                # Step 4: Calculate savings
                original_size = len(float_vector) * 4  # float32 = 4 bytes
                quantized_size = len(quantized_vector) * 1  # int8 = 1 byte
                storage_saved += original_size - quantized_size

                # Step 5: Update document with quantized vector
                updated_doc = {
                    **doc["_source"],
                    "AMAZON_BEDROCK_VECTOR": quantized_vector,
                    "QUANTIZATION_METADATA": metadata,  # Store for dequantization during search
                    "VECTOR_DTYPE": "int8"  # Mark as quantized
                }

                # Re-index with quantized vector
                client.index(index=INDEX_NAME, id=doc["_id"], body=updated_doc)

                quantized_count += 1

            # Get next batch
            response = client.scroll(scroll_id=response["_scroll_id"], scroll="5m")

        # Step 6: Report results
        print(f"\n✅ Quantization Complete!")
        print(f"   Vectors quantized: {quantized_count}")
        print(f"   Storage saved: {storage_saved / 1024 / 1024:.2f} MB")
        print(f"   Per vector: {(1 - (quantized_count * 1024) / (quantized_count * 1024 * 4)) * 100:.1f}% reduction")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Quantization successful",
                "vectors_quantized": quantized_count,
                "storage_saved_mb": storage_saved / 1024 / 1024
            })
        }

    except Exception as e:
        print(f"❌ Error during quantization: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


# For local testing
if __name__ == "__main__":
    test_event = {
        "collection_endpoint": "https://cpuvvwedvaylmxenz0q6.us-east-1.aoss.amazonaws.com",
        "index_name": "rag-pipeline-sandbox-index"
    }

    result = lambda_handler(test_event, None)
    print(json.dumps(json.loads(result["body"]), indent=2))
