

# import boto3
# import json
# bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# response = bedrock_runtime.invoke_model(
#     modelId="anthropic.claude-3-haiku-20240307-v1:0",
#     contentType="application/json",
#     body=json.dumps({
#         "anthropic_version": "bedrock-2023-05-31",
#         "max_tokens": 500,
#         "messages": [
#             {"role": "user", "content": "How are you"}
#         ]
#     })
# )

# print(response)


import boto3
from langchain_aws import ChatBedrock

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

llm = ChatBedrock(
    client=client,
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
)

response = llm.invoke("How are you?")

print(response.content)
print(response.usage_metadata)
print(response.response_metadata)