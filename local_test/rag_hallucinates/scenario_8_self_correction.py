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

def generate_answer(question, chunks):
    """Step 1: Generate initial answer from sources."""

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}]:\n{chunk['content']['text']}\n"

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
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}]
        })
    )

    result = json.loads(response["body"].read())
    answer = result["content"][0]["text"]

    return answer

def verify_answer(question, answer, chunks):
    """Step 2: LLM verifies its own answer against sources.

    This is the KEY to preventing hallucination:
    - Generate answer
    - Ask LLM: "Is this answer supported by sources?"
    - Only use if verified
    """

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}]:\n{chunk['content']['text']}\n"

    verification_prompt = f"""You are a fact-checker. Verify if the answer is supported by sources.

SOURCES:
{context}

ANSWER TO VERIFY:
{answer}

TASK: Check if every claim in the answer is supported by the sources above.

Return JSON with:
- verified: true/false (all claims supported?)
- confidence: 0-1 (how confident are you?)
- issues: list of unsupported claims (if any)
- explanation: brief reason

Example:
{{
  "verified": true,
  "confidence": 0.95,
  "issues": [],
  "explanation": "All claims are directly quoted from sources."
}}

OR

{{
  "verified": false,
  "confidence": 0.8,
  "issues": ["Claims Redis is needed (not in sources)", "Mentions performance by 10x (not stated)"],
  "explanation": "Answer makes claims not present in provided sources."
}}

Return ONLY valid JSON, no other text."""

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": verification_prompt}]
        })
    )

    result = json.loads(response["body"].read())
    response_text = result["content"][0]["text"]

    # Parse verification result
    try:
        verification = json.loads(response_text)
    except:
        print("⚠️  Failed to parse verification response")
        return None, {"verified": False}

    return answer, verification

def correct_answer(question, answer, chunks, issues):
    """Step 3: If verification failed, ask LLM to correct it."""

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}]:\n{chunk['content']['text']}\n"

    correction_prompt = f"""The following answer had issues. Please correct it using ONLY the sources.

SOURCES:
{context}

ORIGINAL ANSWER:
{answer}

ISSUES FOUND:
{', '.join(issues)}

TASK: Rewrite the answer to fix these issues. Remove any claims not supported by sources.
If you cannot answer the question with the sources, say "I don't have this information".

Answer:"""

    response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 500,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": correction_prompt}]
        })
    )

    result = json.loads(response["body"].read())
    corrected_answer = result["content"][0]["text"]

    return corrected_answer

def print_result(answer, verification, sources, corrected=None):
    """Pretty print result with verification details."""

    print("\n" + "="*70)
    print("ANSWER (Self-Corrected):")
    print("="*70)
    print(answer)

    print("\n" + "="*70)
    print("VERIFICATION:")
    print("="*70)

    if verification:
        verified = verification.get("verified", False)
        confidence = verification.get("confidence", 0)
        issues = verification.get("issues", [])
        explanation = verification.get("explanation", "")

        status = "✅ VERIFIED" if verified else "❌ NOT VERIFIED"
        print(f"Status: {status}")
        print(f"Confidence: {confidence:.1%}")
        print(f"Explanation: {explanation}")

        if issues:
            print(f"\nIssues Found:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")

        if corrected:
            print(f"\n📝 CORRECTED ANSWER:")
            print(corrected)

    print("\n" + "="*70)
    print("SOURCES:")
    for label, name in sources.items():
        print(f"  {label}: {name}")
    print("="*70)

def main():
    while True:
        question = input("\nAsk a question or q(quit): ").strip()

        if question.lower() == "q":
            break

        if not question:
            continue

        print("\n🔍 Retrieving chunks...")

        # Step 0: Retrieve confident chunks
        confident, all_results = retrieve_chunks(question)

        if not confident:
            print(f"⚠️  No high-confidence results")
            print(f"Best score: {all_results[0]['score']:.3f} (need >= {CONFIDENCE_THRESHOLD})")
            continue

        print(f"✅ Found {len(confident)} confident chunks")

        # Step 1: Generate answer
        print("\n📝 Step 1: Generating answer...")
        answer = generate_answer(question, confident)

        # Step 2: Verify answer
        print("🔍 Step 2: Verifying answer against sources...")
        verified_answer, verification = verify_answer(question, answer, confident)

        # Step 3: Correct if needed
        corrected_answer = None
        if verification and not verification.get("verified", False):
            print("⚠️  Answer has issues - correcting...")
            corrected_answer = correct_answer(
                question,
                answer,
                confident,
                verification.get("issues", [])
            )
            final_answer = corrected_answer
        else:
            final_answer = answer

        # Step 4: Display result
        sources = {}
        for i, chunk in enumerate(confident, 1):
            source_uri = chunk['location']['s3Location']['uri']
            source_name = source_uri.split("/")[-1]
            sources[f"[Source {i}]"] = source_name

        print_result(final_answer, verification, sources, corrected_answer)


if __name__ == "__main__":
    main()
