import boto3
import json

KB_ID = "WAQARJ0KGQ"

# Step 1: Retrieve ONLY from knowledge base
bedrock_agent_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name="us-east-1"
)

question = input("Ask a question: ")

# Retrieve relevant chunks from knowledge base
response = bedrock_agent_runtime.retrieve(
    knowledgeBaseId=KB_ID,
    retrievalQuery={"text": question},
    retrievalConfiguration={
        "vectorSearchConfiguration": {
            "numberOfResults": 5  # Top 5 chunks
        }
    }
)

# Step 2: Extract and verify retrieved chunks
chunks = response["retrievalResults"]

if not chunks:
    print("❌ No information found in knowledge base for your question.")
    exit()

print(f"\n📚 Found {len(chunks)} relevant chunks from knowledge base:\n")
for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} (Score: {chunk['score']:.2f}) ---")
    print(chunk["content"]["text"][:300] + "...\n")

# Step 3: Create context from ONLY knowledge base chunks
context = "\n\n".join([c["content"]["text"] for c in chunks])

# Step 4: Call Claude with STRICT instruction to use ONLY KB context
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

llm_response = bedrock_runtime.invoke_model(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    contentType="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{
            "role": "user",
            "content": f"""You are an AI assistant that ONLY answers questions based on the provided knowledge base context.
You MUST NOT use any of your training data or general knowledge.

If the answer is not in the context, say "This information is not available in the knowledge base."

KNOWLEDGE BASE CONTEXT:
{context}

QUESTION: {question}

ANSWER (only using above context):"""
        }]
    })
)

result = json.loads(llm_response["body"].read())
answer = result["content"][0]["text"]

print(f"\n✅ ANSWER (from knowledge base only):\n{answer}")
