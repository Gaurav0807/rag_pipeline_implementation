"""
Simple LangChain Example 3: Basic RAG Chain
Retrieve docs + Generate answer
"""

from langchain_community.chat_models.bedrock import BedrockChat
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

llm = BedrockChat(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)

# Step 1: Create retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="WAQARJ0KGQ",
    region_name="us-east-1"
)

# Step 2: Create prompt
prompt = ChatPromptTemplate.from_template("""
Answer based ONLY on this context. Cite sources.

Context:
{context}

Question: {question}

Answer:
""")

# Step 3: Create RAG chain
# Retrieve docs → Format context → Call LLM → Parse output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Step 4: Use it
print("="*60)
print("RAG Chain Example")
print("="*60)

question = "What are the main monitoring tools used?"
result = rag_chain.invoke(question)

print(f"Question: {question}")
print(f"\nAnswer:\n{result}")
