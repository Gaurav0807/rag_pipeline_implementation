"""
Simple LangChain Example 1: Basic LLM Call
Install: pip install langchain langchain-community boto3
"""

from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)


print("Method 1: Direct LLM Call")
response = llm.invoke("What is RAG?")
print(response.content)


print("\n" + "="*60)
print("Method 2: Prompt Template")
print("="*60)

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 2 sentences."
)

chain = prompt | llm
print("chain")
print(chain)
print("chain_end")

result = chain.invoke({"topic": "Transformers"})
print(result.content)

# Method 3: With system message
print("\n" + "="*60)
print("Method 3: System Message")
print("="*60)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant that explains concepts simply."),
    ("user", "What is {topic}?")
])

chain = prompt | llm
print("Chaining start")
print(chain)
print("Chaining end")
result = chain.invoke({"topic": "Vector Database"})
print(result.content)
