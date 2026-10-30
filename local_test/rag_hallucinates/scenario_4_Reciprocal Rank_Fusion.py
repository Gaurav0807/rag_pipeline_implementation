import boto3
import json
from collections import defaultdict

KB_ID = "WAQARJ0KGQ"
CONFIDENCE_THRESHOLD = 0.5
RRF_K = 60  # Constant for RRF formula

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

def retrieve_vector_search(question):
    """Vector search - semantic similarity."""
    response = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}}
    )
    return response.get("retrievalResults", [])

def retrieve_keyword_search(question):
    """Keyword search - exact/partial word matches."""
    response = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}}
    )

    keywords = question.lower().split()
    results = response.get("retrievalResults", [])

    keyword_scored = []
    for r in results:
        text = r['content']['text'].lower()
        matches = sum(1 for kw in keywords if kw in text)
        if matches > 0:
            keyword_scored.append((r, matches))

    return [r for r, _ in sorted(keyword_scored, key=lambda x: x[1], reverse=True)]

def rrf_combine(vector_results, keyword_results, top_k=5):
    """Combine results using Reciprocal Rank Fusion (RRF).

    RRF formula: score = 1 / (k + rank)
    Combines multiple ranking signals without extra LLM calls.
    """

    rrf_scores = defaultdict(float)
    doc_map = {}

    print(f"\n📊 RRF Combining Results:")
    print(f"   K constant: {RRF_K}")

    # Score vector search results
    print(f"\n   Vector Search ({len(vector_results)} results):")
    for rank, result in enumerate(vector_results, 1):
        uri = result['location']['s3Location']['uri']
        rrf_score = 1 / (RRF_K + rank)
        rrf_scores[uri] += rrf_score
        doc_map[uri] = result
        print(f"     Rank {rank}: score = 1/({RRF_K}+{rank}) = {rrf_score:.4f}")

    # Score keyword search results
    print(f"\n   Keyword Search ({len(keyword_results)} results):")
    for rank, result in enumerate(keyword_results, 1):
        uri = result['location']['s3Location']['uri']
        rrf_score = 1 / (RRF_K + rank)
        rrf_scores[uri] += rrf_score
        doc_map[uri] = result
        print(f"     Rank {rank}: score = 1/({RRF_K}+{rank}) = {rrf_score:.4f}")

    # Sort by combined RRF score
    ranked_results = []
    for uri, combined_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        result = doc_map[uri].copy()
        result["rrf_score"] = combined_score
        ranked_results.append(result)

    print(f"\n   Combined RRF Scores:")
    for i, result in enumerate(ranked_results[:top_k], 1):
        uri = result['location']['s3Location']['uri'].split("/")[-1]
        print(f"     {i}. {uri}: {result['rrf_score']:.4f}")

    return ranked_results[:top_k]

def call_llm_with_context(question, chunks):
    """Call Claude with RRF-ranked chunks."""

    context = ""
    sources = {}

    for i, chunk in enumerate(chunks, 1):
        source_uri = chunk['location']['s3Location']['uri']
        source_name = source_uri.split("/")[-1]
        rrf_score = chunk.get("rrf_score", 0)

        sources[f"[Source {i}]"] = {
            "name": source_name,
            "rrf_score": rrf_score,
        }
        context += f"\n[Source {i}] (RRF: {rrf_score:.4f}):\n{chunk['content']['text']}\n"

    prompt = f"""Answer based ONLY on these sources. Always cite them.

        RULES:
        1. Answer only from sources
        2. Always say "According to [Source X], ..."
        3. If sources don't answer, say "I don't have this information"
        4. Never make up information

        SOURCES (Ranked by RRF):
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

def print_result(answer, sources):
    """Pretty print answer with RRF scores."""
    print("\n" + "="*70)
    print("ANSWER (RRF Combined):")
    print("="*70)
    print(answer)
    print("\n" + "="*70)
    print("SOURCES (RRF Ranked):")
    for label, info in sources.items():
        print(f"  {label}: {info['name']}")
        print(f"         RRF Score: {info['rrf_score']:.4f}")
    print("="*70)

# Main loop
while True:
    question = input("\nAsk a question or q(quit): ").strip()

    if question.lower() == "q":
        break

    if not question:
        continue

    print("\n🔍 Retrieving chunks...")

    # Step 1: Vector search
    vector_results = retrieve_vector_search(question)
    print(f"   Vector search: {len(vector_results)} results")

    # Step 2: Keyword search
    keyword_results = retrieve_keyword_search(question)
    print(f"   Keyword search: {len(keyword_results)} results")

    # Step 3: RRF combine (no extra LLM call needed!)
    rrf_ranked = rrf_combine(vector_results, keyword_results, top_k=5)

    if not rrf_ranked:
        print("\n⚠️  No results after RRF combining")
        continue

    # Step 4: Call LLM
    answer, sources = call_llm_with_context(question, rrf_ranked)

    # Step 5: Display result
    print_result(answer, sources)
