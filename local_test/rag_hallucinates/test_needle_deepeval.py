"""
Needle in Haystack using DeepEval

Uses deepeval framework to automatically evaluate:
- Faithfulness (no hallucination)
- Answer Relevancy
- Contextual Relevancy

Install deepeval:
pip install deepeval
"""

import boto3
import json
from deepeval import evaluate
from deepeval.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextualRelevancy,
)
from deepeval.test_case import LLMTestCase

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

    chunks = response.get("retrievalResults", [])
    return chunks

def call_llm(question, chunks):
    """Call LLM to answer question."""

    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}]:\n{chunk['content']['text']}\n"

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

    return answer, context

# Test cases for Needle in Haystack
TEST_CASES = [
    {
        "question": "What are the Spark job failures?",
        "expected_output": "Should discuss Spark job failures and issues",
        "retrieval_ground_truth": "Spark job failures are documented in tickets with memory utilization issues",
    },
    {
        "question": "What monitoring tools are used?",
        "expected_output": "Should mention CloudWatch and Grafana",
        "retrieval_ground_truth": "CloudWatch and Grafana are used for monitoring",
    },
    {
        "question": "What is the Data Pipeline Standards?",
        "expected_output": "Should explain platform processes and architecture",
        "retrieval_ground_truth": "Data Pipeline Standards document explains internal platform processes",
    },
]

def run_deepeval_tests():
    """Run needle-in-haystack tests with deepeval."""

    print("\n" + "="*70)
    print("NEEDLE IN HAYSTACK - DeepEval Evaluation")
    print("="*70)

    test_cases_for_deepeval = []

    for test_case_dict in TEST_CASES:
        question = test_case_dict["question"]
        expected_output = test_case_dict["expected_output"]
        ground_truth = test_case_dict["retrieval_ground_truth"]

        print(f"\n🔄 Processing: {question[:50]}...")

        # Step 1: Retrieve chunks
        chunks = retrieve_chunks(question)
        context_text = "\n".join([chunk['content']['text'] for chunk in chunks])

        # Step 2: Call LLM
        answer, full_context = call_llm(question, chunks)

        print(f"   ✅ Retrieved {len(chunks)} chunks")
        print(f"   ✅ Generated answer ({len(answer)} chars)")

        # Step 3: Create test case for deepeval
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_output,
            retrieval_context=[chunk['content']['text'] for chunk in chunks],
            context=full_context,
        )

        test_cases_for_deepeval.append(test_case)

    # Step 4: Run deepeval metrics
    print("\n" + "="*70)
    print("Running DeepEval Metrics...")
    print("="*70)

    metrics = [
        Faithfulness(),           # Check for hallucination
        AnswerRelevancy(),        # Check if answer is relevant
        ContextualRelevancy(),    # Check if context is relevant
    ]

    try:
        results = evaluate(test_cases_for_deepeval, metrics)

        # Display results
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)

        for i, result in enumerate(results, 1):
            print(f"\n📊 Test Case {i}: {TEST_CASES[i-1]['question'][:50]}...")
            print(f"   Question: {result.test_case.input}")
            print(f"   Answer: {result.test_case.actual_output[:100]}...")

            print(f"\n   Metrics:")
            for metric_result in result.metrics_data:
                metric_name = metric_result.metric
                score = metric_result.score
                reason = metric_result.reason

                # Color code
                if score >= 0.8:
                    status = "✅"
                elif score >= 0.6:
                    status = "⚠️"
                else:
                    status = "❌"

                print(f"   {status} {metric_name}: {score:.2f}")
                if reason:
                    print(f"      Reason: {reason[:80]}...")

        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        total_tests = len(results)
        passed_tests = sum(1 for r in results if all(m.score >= 0.7 for m in r.metrics_data))

        print(f"\nTests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.0%}")

        if passed_tests == total_tests:
            print("\n🎉 EXCELLENT - All needle-in-haystack tests passed!")
        elif passed_tests >= total_tests * 0.7:
            print("\n⚠️  GOOD - Most tests passed, some improvements needed")
        else:
            print("\n❌ POOR - Major issues detected")

        print("="*70)

    except ImportError:
        print("\n❌ DeepEval not installed!")
        print("Install with: pip install deepeval")
        return
    except Exception as e:
        print(f"\n⚠️  Error running deepeval: {e}")
        print("\nNote: DeepEval requires specific setup")
        return


def manual_evaluation():
    """Fallback: Manual evaluation without deepeval."""

    print("\n" + "="*70)
    print("NEEDLE IN HAYSTACK - Manual Evaluation")
    print("(DeepEval not available)")
    print("="*70)

    for test_case_dict in TEST_CASES:
        question = test_case_dict["question"]

        print(f"\n🔄 {question}")

        # Retrieve
        chunks = retrieve_chunks(question)
        print(f"   Retrieved: {len(chunks)} chunks")

        # Answer
        answer, _ = call_llm(question, chunks)
        print(f"   Answer: {answer[:150]}...")

        # Manual check
        if "According to [Source" in answer:
            print(f"   ✅ Has citations")
        else:
            print(f"   ❌ No citations (hallucination risk)")

        print()


if __name__ == "__main__":
    try:
        run_deepeval_tests()
    except ImportError:
        print("\n⚠️  DeepEval not installed. Running manual evaluation instead...")
        manual_evaluation()
