import boto3
import json

KB_ID = "WAQARJ0KGQ"
CONFIDENCE_THRESHOLD = 0.5

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

# Define trusted sources (metadata filtering rules)
TRUSTED_SOURCES = {
    "confluence": {"weight": 0.9, "reason": "Official documentation"},
    "jira": {"weight": 0.7, "reason": "Issue tracking (engineering context)"},
}

UNTRUSTED_SOURCES = {
    "slack": {"weight": 0.3, "reason": "Chat messages (unverified)"},
    "comments": {"weight": 0.2, "reason": "User comments (opinion)"},
}

def extract_source_type(uri):
    """Extract source type from URI."""
    # s3://bucket/rag_test_data/confluence/page_9.json -> confluence
    # s3://bucket/rag_test_data/jira/DATA-1035.json -> jira
    parts = uri.split("/")
    if len(parts) >= 5:
        return parts[4]  # Extract source type
    return "unknown"

def is_trusted_source(chunk):
    """Check if chunk is from trusted source."""
    uri = chunk['location']['s3Location']['uri']
    source_type = extract_source_type(uri)

    if source_type in TRUSTED_SOURCES:
        return True, source_type, TRUSTED_SOURCES[source_type]["reason"]
    elif source_type in UNTRUSTED_SOURCES:
        return False, source_type, UNTRUSTED_SOURCES[source_type]["reason"]
    else:
        return False, source_type, "Unknown source type"

def retrieve_and_filter(question):
    """Retrieve chunks and filter by metadata."""
    response = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}}
    )

    chunks = response.get("retrievalResults", [])

    print(f"\n📊 Metadata Filtering:")
    print(f"   Total retrieved: {len(chunks)}")

    trusted_chunks = []
    filtered_chunks = []

    for chunk in chunks:
        is_trusted, source_type, reason = is_trusted_source(chunk)

        if is_trusted:
            chunk["source_type"] = source_type
            chunk["trust_reason"] = reason
            trusted_chunks.append(chunk)
            print(f"   ✅ {source_type}: {reason}")
        else:
            filtered_chunks.append(chunk)
            print(f"   ❌ {source_type}: {reason}")

    return trusted_chunks, filtered_chunks

def call_llm_with_context(question, chunks):
    """Call Claude with trusted chunks only."""

    context = ""
    sources = {}

    for i, chunk in enumerate(chunks, 1):
        source_uri = chunk['location']['s3Location']['uri']
        source_name = source_uri.split("/")[-1]
        source_type = chunk.get("source_type", "unknown")
        trust_reason = chunk.get("trust_reason", "")

        sources[f"[Source {i}]"] = {
            "name": source_name,
            "type": source_type,
            "reason": trust_reason
        }
        context += f"\n[Source {i}] ({source_type}):\n{chunk['content']['text']}\n"

    prompt = f"""Answer based ONLY on these TRUSTED sources. Always cite them.

RULES:
1. Answer only from sources below
2. All sources are verified/official
3. Always say "According to [Source X], ..."
4. If sources don't answer, say "I don't have this information"
5. Never make up information

TRUSTED SOURCES:
{context}

Question: {question}

Answer:"""

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    result = json.loads(response["body"].read())
    answer = result["content"][0]["text"]

    return answer, sources

def print_result(answer, sources, filtered_count):
    """Pretty print answer with metadata info."""
    print("\n" + "="*70)
    print("ANSWER (Metadata Filtered):")
    print("="*70)
    print(answer)
    print("\n" + "="*70)
    print("TRUSTED SOURCES USED:")
    for label, info in sources.items():
        print(f"  {label}: {info['name']} ({info['type']})")
        print(f"          {info['reason']}")
    print(f"\n  Filtered out: {filtered_count} untrusted chunks")
    print("="*70)

# Main loop
while True:
    question = input("\nAsk a question or q(quit): ").strip()

    if question.lower() == "q":
        break

    if not question:
        continue

    print("\n🔍 Retrieving chunks...")

    # Step 1: Retrieve and filter by metadata
    trusted, filtered = retrieve_and_filter(question)

    if not trusted:
        print("\n⚠️  No trusted sources found")
        print(f"   Available: {len(filtered)} untrusted sources")
        print("   Cannot answer safely - would cause hallucination")
        continue

    print(f"\n✅ Using {len(trusted)} trusted sources")

    # Step 2: Call LLM with trusted chunks only
    answer, sources = call_llm_with_context(question, trusted)

    # Step 3: Display result
    print_result(answer, sources, len(filtered))
