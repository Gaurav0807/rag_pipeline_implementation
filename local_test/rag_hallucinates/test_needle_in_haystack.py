"""
Needle in a Haystack Test for RAG

Tests if your RAG can find relevant info when buried in noise.

Metrics:
- Retrieval accuracy: Did it find the needle?
- Hallucination rate: Did it make stuff up?
- Noise handling: Did noise confuse it?
"""

import boto3
import json

KB_ID = "WAQARJ0KGQ"

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

# Test cases: (question, expected_keywords_in_answer, test_name)
TEST_CASES = [
    (
        "What are the Spark job failures?",
        ["Spark", "failure", "job"],
        "Spark Needle Test"
    ),
    (
        "What is in the Data Pipeline Standards document?",
        ["pipeline", "standards", "platform"],
        "Pipeline Needle Test"
    ),
    (
        "What monitoring tools are mentioned?",
        ["CloudWatch", "Grafana", "monitoring"],
        "Monitoring Needle Test"
    ),
]

def retrieve_chunks(question):
    """Retrieve chunks from KB."""
    response = bedrock_agent.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 10}}
    )

    chunks = response.get("retrievalResults", [])
    return chunks

def call_llm(question, chunks):
    """Call LLM to answer question."""

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}]:\n{chunk['content']['text'][:200]}...\n"

    prompt = f"""Answer based ONLY on these sources. Always cite them.

RULES:
1. Answer only from sources
2. Always cite sources: "According to [Source X], ..."
3. If sources don't answer, say: "I don't have this information"
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

def evaluate_answer(answer, expected_keywords):
    """Check if answer contains expected keywords (found the needle)."""

    answer_lower = answer.lower()
    found_keywords = []
    missing_keywords = []

    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    # Score: percentage of keywords found
    score = len(found_keywords) / len(expected_keywords) if expected_keywords else 0

    return {
        "score": score,
        "found_keywords": found_keywords,
        "missing_keywords": missing_keywords,
        "success": score >= 0.7,  # 70% threshold
    }

def check_for_hallucination(answer):
    """Check if answer admits uncertainty or makes things up."""

    hallucination_indicators = [
        "I don't have this information",
        "I cannot find",
        "not mentioned",
        "not in the sources",
    ]

    admission_indicators = any(indicator in answer for indicator in hallucination_indicators)

    if "According to [Source" in answer and not admission_indicators:
        has_citations = True
    else:
        has_citations = False

    return {
        "admits_uncertainty": admission_indicators,
        "has_citations": has_citations,
        "likely_hallucination": not (has_citations or admission_indicators),
    }

def print_test_result(test_name, question, answer, chunks, evaluation, hallucination_check):
    """Print detailed test results."""

    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print("="*70)

    print(f"\n❓ Question:")
    print(f"   {question}")

    print(f"\n📊 Retrieval Stats:")
    print(f"   Retrieved {len(chunks)} chunks")
    print(f"   Top score: {chunks[0]['score']:.3f}")

    print(f"\n📝 Answer:")
    print(f"   {answer[:200]}...")

    print(f"\n✅ Evaluation:")
    print(f"   Accuracy Score: {evaluation['score']:.1%}")
    print(f"   Found Keywords: {', '.join(evaluation['found_keywords']) or 'None'}")
    print(f"   Missing Keywords: {', '.join(evaluation['missing_keywords']) or 'None'}")

    print(f"\n🔍 Hallucination Check:")
    print(f"   Has Citations: {'✅' if hallucination_check['has_citations'] else '❌'}")
    print(f"   Admits Uncertainty: {'✅' if hallucination_check['admits_uncertainty'] else '❌'}")
    print(f"   Likely Hallucination: {'❌ YES (BAD!)' if hallucination_check['likely_hallucination'] else '✅ NO (GOOD!)'}")

    print(f"\n🎯 Result:")
    if evaluation['success'] and not hallucination_check['likely_hallucination']:
        print(f"   ✅ PASS - Found needle, no hallucination")
    elif evaluation['success']:
        print(f"   ⚠️  PARTIAL - Found needle but might hallucinate")
    else:
        print(f"   ❌ FAIL - Didn't find needle")

    print("="*70)

def run_all_tests():
    """Run all needle-in-haystack tests."""

    print("\n" + "="*70)
    print("NEEDLE IN HAYSTACK TESTS - RAG Robustness")
    print("="*70)

    results = []

    for question, expected_keywords, test_name in TEST_CASES:
        print(f"\n🔄 Running: {test_name}...")

        # Step 1: Retrieve
        chunks = retrieve_chunks(question)

        # Step 2: Generate answer
        answer = call_llm(question, chunks)

        # Step 3: Evaluate
        evaluation = evaluate_answer(answer, expected_keywords)
        hallucination_check = check_for_hallucination(answer)

        # Step 4: Display
        print_test_result(test_name, question, answer, chunks, evaluation, hallucination_check)

        results.append({
            "test": test_name,
            "accuracy": evaluation['score'],
            "no_hallucination": not hallucination_check['likely_hallucination'],
            "passed": evaluation['success'] and not hallucination_check['likely_hallucination'],
        })

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"\n{status} - {result['test']}")
        print(f"   Accuracy: {result['accuracy']:.1%}")
        print(f"   No Hallucination: {'✅' if result['no_hallucination'] else '❌'}")

    passed = sum(1 for r in results if r['passed'])
    total = len(results)

    print("\n" + "="*70)
    print(f"OVERALL: {passed}/{total} tests passed ({passed/total:.0%})")

    if passed == total:
        print("🎉 EXCELLENT - RAG is robust!")
    elif passed >= total * 0.7:
        print("⚠️  GOOD - Some improvements needed")
    else:
        print("❌ POOR - Major hallucination issues!")

    print("="*70)


if __name__ == "__main__":
    run_all_tests()
