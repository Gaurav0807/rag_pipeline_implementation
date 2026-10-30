"""
Simple LangChain Example 5: Vector Store & Similarity Search
Store documents as vectors, search by similarity
"""

from langchain.schema import Document
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Initialize embeddings
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

# Create sample documents
documents = [
    Document(page_content="RAG retrieves context before generating answers."),
    Document(page_content="Fine-tuning trains the model on task-specific data."),
    Document(page_content="Prompt engineering optimizes input for better outputs."),
    Document(page_content="Vector databases store embeddings for fast retrieval."),
]

print("="*60)
print("Creating Vector Store (FAISS)")
print("="*60)

# Create vector store from documents
vector_store = FAISS.from_documents(documents, embeddings)
print(f"Created vector store with {len(documents)} documents")

# Similarity search
print("\n" + "="*60)
print("Similarity Search")
print("="*60)

query = "How do we store and retrieve documents?"

# Find similar documents
results = vector_store.similarity_search(query, k=2)

print(f"Query: {query}\n")
print(f"Found {len(results)} similar documents:\n")
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}")

# Similarity search with scores
print("\n" + "="*60)
print("Similarity Search with Scores")
print("="*60)

results_with_scores = vector_store.similarity_search_with_score(query, k=2)

for i, (doc, score) in enumerate(results_with_scores, 1):
    print(f"{i}. Score: {score:.4f}")
    print(f"   Content: {doc.page_content}\n")

# Create retriever
print("="*60)
print("Create Retriever")
print("="*60)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# Use retriever
retrieved_docs = retriever.invoke(query)
print(f"Retriever found {len(retrieved_docs)} documents")
for doc in retrieved_docs:
    print(f"  - {doc.page_content}")
