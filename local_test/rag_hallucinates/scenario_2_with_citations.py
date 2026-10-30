import boto3
import json
 #2. Source Attribution (Enforce Citations)
KB_ID = "WAQARJ0KGQ"
CONFIDENCE_THRESHOLD = 0.5

bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
bedrock_agent = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

while True:
    question = input("\nAsk a question or q(quit):  ")

    if question.lower() == "q":
        break
    else:

    # Step 1: Retrieve chunks
        response = bedrock_agent.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": 5}
            }
        )

        results = response["retrievalResults"]
        confident = [r for r in results if r['score'] >= CONFIDENCE_THRESHOLD]

        if not confident:
            print("\n❌ Not enough information in knowledge base")
            print(f"Best score: {results[0]['score']:.3f} (need >= {CONFIDENCE_THRESHOLD})")
            continue

        print(f"\n📚 Found {len(confident)} relevant chunks\n")

        # Step 2: Build context with source tracking
        context = ""
        sources = {}

        for i, chunk in enumerate(confident):
            source_uri = chunk['location']['s3Location']['uri']
            source_name = source_uri.split("/")[-1]
            sources[f"[Source {i+1}]"] = source_name

            context += f"\n[Source {i+1}]:\n{chunk['content']['text']}\n"

        # Step 3: Call LLM with grounding prompt
        prompt = f"""You are a helpful assistant with access to a knowledge base.

            IMPORTANT RULES:
            1. Answer ONLY based on the provided sources
            2. Always cite sources when answering: "According to [Source X], ..."
            3. If the sources don't answer the question, say: "I don't have this information"
            4. Never make up or infer information
            5. If you're uncertain, admit it

            Knowledge Base Sources:
            {context}

            Question: {question}

            Answer:"""

        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            contentType="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 500,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            })
        )

        result = json.loads(response["body"].read())
        answer = result["content"][0]["text"]

        print("="*70)
        print("ANSWER:")
        print("="*70)
        print(answer)
        print("\n" + "="*70)
        print("SOURCES USED:")
        for label, name in sources.items():
            print(f"  {label}: {name}")
        print("="*70)
