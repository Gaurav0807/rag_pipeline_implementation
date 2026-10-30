import boto3
import json
from collections import defaultdict

KB_ID = "WAQARJ0KGQ"
CONFIDENCE_THRESHOLD = 0.5

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

def retrieve_vector_search(question):
    """Vector search - semantic similarity."""
    response = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}}
    )
    return response.get("retrievalResults", [])

def retrieve_keyword_search(question):
    """Keyword search - exact/partial word matches."""
    response = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}}
    )
    # Note: Bedrock KB doesn't have native keyword search,
    # so we filter results by keyword matching
    keywords = question.lower().split()
    results = response.get("retrievalResults", [])

    keyword_scored = []
    for r in results:
        text = r['content']['text'].lower()
        matches = sum(1 for kw in keywords if kw in text)
        if matches > 0:
            keyword_scored.append((r, matches))

    return [r for r, _ in sorted(keyword_scored, key=lambda x: x[1], reverse=True)]

def merge_results(vector_results, keyword_results):
    """Merge and deduplicate vector + keyword results."""
    seen = set()
    merged = []

    # Add vector results first (better semantics)
    for r in vector_results:
        uri = r['location']['s3Location']['uri']
        if uri not in seen:
            seen.add(uri)
            merged.append(r)

    # Add keyword results if not already included
    for r in keyword_results:
        uri = r['location']['s3Location']['uri']
        if uri not in seen:
            seen.add(uri)
            merged.append(r)

    return merged

def call_llm_with_context(question, chunks):
    """Call Claude with cited answer."""

    context = ""
    sources = {}

    for i, chunk in enumerate(chunks, 1):
        source_uri = chunk['location']['s3Location']['uri']
        source_name = source_uri.split("/")[-1]
        sources[f"[Source {i}]"] = source_name
        context += f"\n[Source {i}]:\n{chunk['content']['text']}\n"

    prompt = f"""Answer based ONLY on these sources. Always cite them.

            RULES:
            1. Answer only from sources
            2. Always say "According to [Source ], ..."
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
    """Pretty print answer with search type and sources."""
    print("\n" + "="*70)
    print(f"ANSWER ({search_type}):")
    print("="*70)
    print(answer)
    print("\n" + "="*70)
    print("SOURCES:")
    for label, name in sources.items():
        print(f"  {label}: {name}")
    print("="*70)

# Main loop
while True:
    question = input("\nAsk a question or q(quit): ").strip()

    if question.lower() == "q":
        break

    if not question:
        continue

    print(" Searching...")

    # Step 1: Hybrid search (vector + keyword)
    vector_results = retrieve_vector_search(question)
    keyword_results = retrieve_keyword_search(question)

    merged_results = merge_results(vector_results, keyword_results)
    confident = [r for r in merged_results if r['score'] >= CONFIDENCE_THRESHOLD]

    if not confident:
        print("\n⚠️  No high-confidence results from hybrid search")
        if merged_results:
            print(f"Best available score: {merged_results[0]['score']:.3f}")
        print("Cannot answer safely\n")
        continue

    print(f"✅ Found {len(confident)} confident results (vector+keyword)")

    # Step 2: Call LLM with grounding
    answer, sources = call_llm_with_context(question, confident)

    # Step 3: Display result
    print_result(answer, sources, "Hybrid Search")


# Hybrid Search = Combining Vector Search + Keyword Search

#   Comparison:

#   Vector Search (Semantic)

#   Question: "performance issues"
#   Finds: "latency problems", "slow responses", "bottlenecks"
#   Why: Understands meaning/intent, not just exact words
#   Score: 0.85

#   Keyword Search (Exact Match)

#   Question: "performance issues"
#   Finds: Documents with words "performance" OR "issues"
#   Why: Simple word matching
#   Score: Based on word frequency

#   Hybrid = Both Together

#   Question: "spark optimization"

#   1. Vector Search finds:
#      - "Spark performance tuning" (score 0.54) ✅
#      - "Data processing best practices" (score 0.51)

#   2. Keyword Search finds:
#      - "Spark job failure" (has word "Spark") ✅
#      - "Optimization guide" (has word "Optimization") ✅

#   3. Merge & Deduplicate:
#      - Spark performance tuning (from vector)
#      - Spark job failure (from keyword - NEW!)
#      - Optimization guide (from keyword - NEW!)

#   Result: Better coverage! Found docs vector search missed.

#   Why It's Better:

#   ┌──────────────┬─────────────────────┬─────────────────────────┐
#   │ Search Type  │        Pros         │          Cons           │
#   ├──────────────┼─────────────────────┼─────────────────────────┤
#   │ Vector Only  │ Understands context │ Misses exact matches    │
#   ├──────────────┼─────────────────────┼─────────────────────────┤
#   │ Keyword Only │ Finds exact terms   │ Misses semantic meaning │
#   ├──────────────┼─────────────────────┼─────────────────────────┤
#   │ Hybrid       │ Gets both!          │ Slightly slower         │
#   └──────────────┴─────────────────────┴─────────────────────────┘


