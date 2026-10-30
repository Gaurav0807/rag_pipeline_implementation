import boto3
import json

KB_ID = "WAQARJ0KGQ"
CONFIDENCE_THRESHOLD = 0.5

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

def retrieve_chunks(question):
    """Retrieve confident chunks from KB."""
    response = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}}
    )

    results = response.get("retrievalResults", [])
    confident = [r for r in results if r['score'] >= CONFIDENCE_THRESHOLD]

    return confident, results

def call_llm_with_temperature(question, chunks, temperature):
    """Call Claude with specified temperature.

    Temperature controls creativity:
    - 0.0: Deterministic, stick to facts (NO hallucination)
    - 0.5: Balanced, some variation
    - 1.0: Very creative, likely hallucinate
    """

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}]:\n{chunk['content']['text']}\n"

    prompt = f"""Answer based ONLY on these sources. Always cite them.

RULES:
1. Answer only from sources
2. Always say "According to [Source X], ..."
3. If sources don't answer, say "I don't have this information"
4. Never make up or infer information
5. Be factual, not creative

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
            "temperature": temperature,  # ← KEY: Control temperature
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    result = json.loads(response["body"].read())
    answer = result["content"][0]["text"]

    return answer

def print_comparison(answers, sources):
    """Print answers with different temperatures side-by-side."""
    temps = [0.0, 0.5, 1.0]

    print("\n" + "="*70)
    print("TEMPERATURE COMPARISON")
    print("="*70)

    for i, temp in enumerate(temps):
        print(f"\n{'─'*70}")
        if temp == 0.0:
            print(f"Temperature: {temp} (FACTUAL - Best for RAG)")
        elif temp == 0.5:
            print(f"Temperature: {temp} (BALANCED)")
        else:
            print(f"Temperature: {temp} (CREATIVE - Risk of hallucination)")
        print(f"{'─'*70}")
        print(answers[i])

    print("\n" + "="*70)
    print("SOURCES USED:")
    for label, name in sources.items():
        print(f"  {label}: {name}")
    print("="*70)

    print("\n💡 RECOMMENDATION:")
    print("   Use Temperature 0.0 for RAG to prevent hallucination")
    print("   Temperature 0.5-1.0 makes LLM more creative but less factual")

def main():
    while True:
        question = input("\nAsk a question or q(quit): ").strip()

        if question.lower() == "q":
            break

        if not question:
            continue

        print("\n🔍 Retrieving chunks...")

        # Step 1: Retrieve confident chunks
        confident, all_results = retrieve_chunks(question)

        if not confident:
            print(f"⚠️  No high-confidence results")
            print(f"Best score: {all_results[0]['score']:.3f} (need >= {CONFIDENCE_THRESHOLD})")
            continue

        print(f"✅ Found {len(confident)} confident chunks")

        # Step 2: Call LLM with different temperatures
        print("\n⏳ Calling LLM with 3 different temperatures...")

        temperatures = [0.0, 0.5, 1.0]
        answers = []

        for temp in temperatures:
            print(f"   Temperature {temp}...", end=" ", flush=True)
            answer = call_llm_with_temperature(question, confident, temp)
            answers.append(answer)
            print("✓")

        # Step 3: Show sources
        sources = {}
        for i, chunk in enumerate(confident, 1):
            source_uri = chunk['location']['s3Location']['uri']
            source_name = source_uri.split("/")[-1]
            sources[f"[Source {i}]"] = source_name

        # Step 4: Display comparison
        print_comparison(answers, sources)


if __name__ == "__main__":
    main()
