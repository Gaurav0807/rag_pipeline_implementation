import boto3

KB_ID = "WAQARJ0KGQ"

client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")

CONFIDENCE_THRESHOLD = 0.5  # Adjust this: 0.3 (low KB), 0.5 (medium), 0.7 (high)

while True:
    question = input("Ask a question or q(quit): ")

    if question.lower() == "q":
        break
    else:

        response = client.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": question},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5
                }
            }
        )

        results = response["retrievalResults"]

        # Filter by confidence threshold
        confident = [r for r in results if r['score'] >= CONFIDENCE_THRESHOLD]

        if not confident:
            print("\n⚠️  No high-confidence results found")
            print(f"Best score: {results[0]['score']:.3f} (need >= {CONFIDENCE_THRESHOLD})")
            continue

        print(f"\n📚 Retrieved Chunks ({len(confident)} passed):\n")
        for i, result in enumerate(confident, 1):
            print(f"--- Chunk {i} (Score: {result['score']:.3f}) ---")
            print(result["content"]["text"][:300])
            print(f"Source: {result['location']['s3Location']['uri']}\n")
