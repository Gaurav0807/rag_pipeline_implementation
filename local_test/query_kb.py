import boto3

KB_ID = "WAQARJ0KGQ"

client = boto3.client(
    "bedrock-agent-runtime",
    region_name="us-east-1"
)

question = input("Ask a question: ")

response = client.retrieve_and_generate(
    input={"text":question},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": KB_ID,
            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0"
        }
        
    }
)

print(response["output"]["text"])