import boto3
import json

KB_ID = "WAQARJ0KGQ"
CONFIDENCE_THRESHOLD = 0.5

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

def retrieve_chunks(question):
    """Retrieve chunks from KB."""
    response = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}}
    )
    return response.get("retrievalResults", [])

def rerank_chunks(question, chunks, top_k=5):
    """Rerank chunks using Claude as a judge."""

    if not chunks:
        return []

    # Build reranking prompt
    chunk_text = ""
    for i, chunk in enumerate(chunks, 1):
        text = chunk['content']['text'][:200]
        chunk_text += f"\n[{i}] {text}...\n"

    rerank_prompt = f"""You are a relevance judge. Rate these chunks by how well they answer the question.

    Question: {question}

    Chunks to rank:
    {chunk_text}

    TASK: Return a JSON list with chunk numbers and relevance scores (0-1).
    Only return chunks with score >= 0.5.

    Example format:
    [
    {{"chunk_id": 1, "score": 0.9, "reason": "directly answers question"}},
    {{"chunk_id": 3, "score": 0.7, "reason": "partially related"}},
    ]

    Return ONLY valid JSON, no other text."""

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": rerank_prompt}]
        })
    )

    result = json.loads(response["body"].read())
    response_text = result["content"][0]["text"]

    # Parse JSON response
    try:
        rankings = json.loads(response_text)
    except:
        print("⚠️  Failed to parse reranking response")
        return chunks[:top_k]

    # Reorder chunks by rerank score
    ranked_chunks = []
    for rank in rankings:
        chunk_id = rank["chunk_id"] - 1  # Convert to 0-indexed
        if chunk_id < len(chunks):
            chunk = chunks[chunk_id].copy()
            chunk["rerank_score"] = rank["score"]
            chunk["rerank_reason"] = rank.get("reason", "")
            ranked_chunks.append(chunk)

    # Sort by rerank score descending
    ranked_chunks.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    return ranked_chunks[:top_k]

def call_llm_with_context(question, chunks):
    """Call Claude with reranked chunks."""

    context = ""
    sources = {}

    for i, chunk in enumerate(chunks, 1):
        source_uri = chunk['location']['s3Location']['uri']
        source_name = source_uri.split("/")[-1]
        rerank_score = chunk.get("rerank_score", 0)
        rerank_reason = chunk.get("rerank_reason", "")

        sources[f"[Source {i}]"] = {
            "name": source_name,
            "rerank_score": rerank_score,
            "reason": rerank_reason
        }
        context += f"\n[Source {i}] (Relevance: {rerank_score:.2f}):\n{chunk['content']['text']}\n"

    prompt = f"""Answer based ONLY on these sources. Always cite them.

        RULES:
        1. Answer only from sources
        2. Always say "According to [Source X], ..."
        3. If sources don't answer, say "I don't have this information"
        4. Never make up information

        SOURCES:
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

def print_result(answer, sources, search_type):
    """Pretty print answer with rerank scores."""
    print("\n" + "="*70)
    print(f"ANSWER ({search_type}):")
    print("="*70)
    print(answer)
    print("\n" + "="*70)
    print("SOURCES (Ranked by Relevance):")
    for label, info in sources.items():
        print(f"  {label}: {info['name']}")
        print(f"         Relevance: {info['rerank_score']:.2f} - {info['reason']}")
    print("="*70)

# Main loop
while True:
    question = input("\nAsk a question or q(quit): ").strip()

    if question.lower() == "q":
        break

    if not question:
        continue

    print("\n🔍 Retrieving chunks...")

    # Step 1: Retrieve initial chunks
    chunks = retrieve_chunks(question)

    if not chunks:
        print("⚠️  No results found")
        continue

    print(f"   Retrieved {len(chunks)} chunks")

    # Step 2: Rerank chunks
    print("🎯 Reranking by relevance...")
    reranked = rerank_chunks(question, chunks, top_k=5)

    if not reranked:
        print("⚠️  No relevant chunks after reranking")
        continue

    print(f"   Selected {len(reranked)} most relevant chunks")

    # Step 3: Call LLM
    answer, sources = call_llm_with_context(question, reranked)

    # Step 4: Display result
    print_result(answer, sources, "Reranked Search")
