# RAG Pipeline Implementation

AWS Bedrock Knowledge Base with OpenSearch Serverless for document retrieval and AI-powered Q&A.

## Architecture

```
S3 (Documents) → Knowledge Base → OpenSearch Serverless (Vectors) → Bedrock LLM
```

## Setup

1. Deploy infrastructure:
```bash
cd terraform
terraform init
terraform apply
```

2. Upload documents to S3 bucket

3. Trigger ingestion:
```bash
cd local_test
python script_test.py
```

## Querying the Knowledge Base

### Option 1: Retrieve + Generate (Recommended)

Single API call - retrieves relevant chunks and generates answer:

```python
import boto3

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

KB_ID = "YOUR_KB_ID"

response = bedrock_agent_runtime.retrieve_and_generate(
    input={"text": "What are the Kafka issues reported?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": KB_ID,
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
        }
    }
)

print(response["output"]["text"])
```

### Option 2: Retrieve Only (Use Your Own LLM)

Get relevant chunks, then call any LLM:

```python
import boto3
import json

bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

KB_ID = "YOUR_KB_ID"

# Step 1: Retrieve relevant chunks
response = bedrock_agent_runtime.retrieve(
    knowledgeBaseId=KB_ID,
    retrievalQuery={"text": "What are Kafka issues?"},
    retrievalConfiguration={
        "vectorSearchConfiguration": {
            "numberOfResults": 5
        }
    }
)

# Step 2: Extract chunks
chunks = [r["content"]["text"] for r in response["retrievalResults"]]
context = "\n\n".join(chunks)

# Step 3: Call LLM with context
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

llm_response = bedrock_runtime.invoke_model(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    contentType="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": f"Based on this context:\n{context}\n\nAnswer: What are Kafka issues?"
        }]
    })
)

result = json.loads(llm_response["body"].read())
print(result["content"][0]["text"])
```

## API Reference

| Client | Method | Use Case |
|--------|--------|----------|
| `bedrock-agent-runtime` | `retrieve_and_generate()` | One-call RAG (retrieval + LLM answer) |
| `bedrock-agent-runtime` | `retrieve()` | Get chunks only, use your own LLM |
| `bedrock-runtime` | `invoke_model()` | Direct LLM call (no KB) |
| `bedrock-agent` | `start_ingestion_job()` | Sync documents to KB |

## Project Structure

```
├── terraform/           # Infrastructure as code
│   ├── knowledge_base.tf   # KB + embedding config
│   ├── opensearch.tf       # Vector store
│   ├── data_source.tf      # S3 data source
│   └── iam.tf              # Permissions
├── local_test/          # Test scripts
│   ├── test_script.py      # Generate sample data
│   ├── script_test.py      # Trigger ingestion
│   └── boto3_script.py     # Create OpenSearch index
└── docs/                # Documentation
```
