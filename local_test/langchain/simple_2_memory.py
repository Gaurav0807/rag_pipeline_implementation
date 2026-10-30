"""
Simple LangChain Example 2: Conversation with Memory
Memory lets LLM remember previous messages
"""

from langchain_community.chat_models.bedrock import BedrockChat
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = BedrockChat(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)

# Create memory
memory = ConversationBufferMemory()

# Create prompt with chat history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("placeholder", "{chat_history}"),
    ("user", "{input}")
])

# Create chain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)

# Conversation 1
print("="*60)
print("User: What is RAG?")
print("="*60)
response1 = chain.invoke({"input": "What is RAG?"})
print(response1["text"])

# Conversation 2 - LLM remembers previous context
print("\n" + "="*60)
print("User: How is it different from fine-tuning?")
print("="*60)
response2 = chain.invoke({"input": "How is it different from fine-tuning?"})
print(response2["text"])

# See memory content
print("\n" + "="*60)
print("Memory Content:")
print("="*60)
print(memory.buffer)
