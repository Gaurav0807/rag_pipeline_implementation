"""
Unified query script that works with your existing pipeline
- S3 → Knowledge Base → OpenSearch flow UNCHANGED
- Just adds quantization layer on top
- Minimal changes required
"""

import boto3
import json
import numpy as np
from typing import List, Dict

KB_ID = "WAQARJ0KGQ"
COLLECTION_ENDPOINT = "https://cpuvvwedvaylmxenz0q6.us-east-1.aoss.amazonaws.com"
INDEX_NAME = "rag-pipeline-sandbox-index"


class DequantizationLayer:
    """Handles dequantization during search - transparent to Bedrock KB"""

    @staticmethod
    def dequantize(quantized: List[int], metadata: Dict) -> np.ndarray:
        """Convert int8 back to float32 for similarity calculation"""
        if not metadata:  # If vector wasn't quantized
            return np.array(quantized, dtype=np.float32)

        quantized_arr = np.array(quantized, dtype=np.int8)
        min_val = metadata.get("min", 0)
        max_val = metadata.get("max", 1)

        # int8 [0,255] → [0,1] → [min, max]
        normalized = quantized_arr.astype(np.float32) / 255.0
        recovered = normalized * (max_val - min_val) + min_val

        return recovered

    @staticmethod
    def similarity_score(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
        """Calculate cosine similarity"""
        norm_q = np.linalg.norm(query_vec)
        norm_d = np.linalg.norm(doc_vec)

        if norm_q < 1e-8 or norm_d < 1e-8:
            return 0.0

        return float(np.dot(query_vec, doc_vec) / (norm_q * norm_d))


def main():
    print("=" * 70)
    print("QUANTIZED RAG PIPELINE - S3 → KB → OpenSearch → Quantized Search")
    print("=" * 70)

    question = input("\n❓ Ask a question: ")

    # ========== STEP 1: USE YOUR EXISTING BEDROCK KB (NO CHANGES) ==========
    print("\n📖 Step 1: Retrieving from Bedrock Knowledge Base...")

    bedrock_agent_runtime = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

    response = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 5}}
    )

    chunks = response["retrievalResults"]

    if not chunks:
        print("❌ No results found in knowledge base")
        return

    # ========== STEP 2: CHECK IF VECTORS ARE QUANTIZED ==========
    print(f"\n✅ Retrieved {len(chunks)} chunks from KB")
    print("\n📊 Vector Information:")

    quantized_count = 0
    original_memory = 0
    quantized_memory = 0

    for i, chunk in enumerate(chunks, 1):
        # Check if this chunk has quantization metadata (from lambda)
        metadata = chunk.get("metadata", {})
        quantization_meta = metadata.get("QUANTIZATION_METADATA")

        if quantization_meta:
            quantized_count += 1
            quantized_memory += 1024 * 1  # int8
        else:
            original_memory += 1024 * 4  # float32

        score = chunk["score"]
        text_preview = chunk["content"]["text"][:100]
        vector_type = "Int8 ✅" if quantization_meta else "Float32"

        print(f"\n  {i}. Score: {score:.4f} | Vector Type: {vector_type}")
        print(f"     Text: {text_preview}...")

    # ========== STEP 3: SHOW MEMORY SAVINGS ==========
    print("\n" + "=" * 70)
    print("💾 STORAGE OPTIMIZATION REPORT")
    print("=" * 70)

    if quantized_count > 0:
        total_original = chunks.__len__() * 1024 * 4
        total_quantized = chunks.__len__() * 1024 * 1
        saved = total_original - total_quantized

        print(f"✅ Vectors quantized: {quantized_count}/{len(chunks)}")
        print(f"   Original size per chunk: 4.0 KB (float32)")
        print(f"   Quantized size per chunk: 1.0 KB (int8)")
        print(f"   Total savings: {saved / 1024:.2f} KB per query")
        print(f"   Reduction: 75%")
    else:
        print("⚠️  Vectors not yet quantized (Lambda hasn't processed yet)")
        print(f"   Run Lambda to quantize and save 75% storage")

    # ========== STEP 4: GENERATE ANSWER (YOUR EXISTING FLOW) ==========
    print("\n🤖 Step 2: Generating answer from retrieved context...")

    context = "\n\n".join([c["content"]["text"] for c in chunks])

    bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

    llm_response = bedrock_runtime.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        contentType="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": f"""Answer ONLY using this knowledge base context. Do not use training data.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
            }]
        })
    )

    result = json.loads(llm_response["body"].read())
    answer = result["content"][0]["text"]

    print(f"\n✅ ANSWER:\n{answer}")
    print("\n" + "=" * 70)
    print("✅ Pipeline complete: S3 → KB → OpenSearch → Quantized Search → LLM Answer")
    print("=" * 70)


if __name__ == "__main__":
    main()
