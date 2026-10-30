"""
Simple LangGraph Example 4: RAG Workflow
Full RAG pipeline as a graph
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_community.chat_models.bedrock import BedrockChat
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

llm = BedrockChat(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)

retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="WAQARJ0KGQ",
    region_name="us-east-1"
)

# State schema
class RAGState(TypedDict):
    query: str
    documents: list
    context: str
    answer: str
    feedback: str

# Nodes
def node_retrieve(state: RAGState) -> RAGState:
    """Retrieve documents from KB"""
    print("🔍 Retrieving documents...")
    try:
        docs = retriever.invoke(state["query"])
        state["documents"] = [doc.page_content for doc in docs]
        state["context"] = "\n\n".join(state["documents"])
        print(f"   Retrieved {len(state['documents'])} documents")
    except Exception as e:
        print(f"   Error retrieving: {e}")
        state["documents"] = []
        state["context"] = "No documents found"
    return state

def node_check_context(state: RAGState) -> str:
    """Check if we have relevant context"""
    print("✅ Checking context quality...")
    if state["documents"]:
        return "generate"
    else:
        return "no_docs"

def node_generate(state: RAGState) -> RAGState:
    """Generate answer from context"""
    print("🤖 Generating answer...")
    prompt = f"""Answer the question based ONLY on this context:

Context:
{state['context']}

Question: {state['query']}

Answer:"""
    response = llm.invoke(prompt)
    state["answer"] = response.content
    state["feedback"] = "Answer generated from retrieved documents"
    return state

def node_no_docs(state: RAGState) -> RAGState:
    """Handle when no documents found"""
    print("⚠️  No relevant documents found")
    response = llm.invoke(f"Question: {state['query']}\n\nAnswer:")
    state["answer"] = response.content
    state["feedback"] = "Answer generated without retrieval (no relevant documents)"
    return state

# Create workflow
workflow = StateGraph(RAGState)

# Add nodes
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("generate", node_generate)
workflow.add_node("no_docs", node_no_docs)

# Add edges
workflow.add_edge("retrieve", "check_context")  # Go to conditional logic
workflow.add_conditional_edges(
    "retrieve",
    node_check_context,
    {
        "generate": "generate",
        "no_docs": "no_docs",
    }
)

workflow.add_edge("generate", END)
workflow.add_edge("no_docs", END)

workflow.set_entry_point("retrieve")

app = workflow.compile()

# Test it
print("="*60)
print("RAG Workflow Example")
print("="*60)

test_queries = [
    "What are the main monitoring tools?",
    "Tell me about cloud computing"
]

for query in test_queries:
    print(f"\n🔹 Query: {query}")
    result = app.invoke({
        "query": query,
        "documents": [],
        "context": "",
        "answer": "",
        "feedback": ""
    })

    print(f"\nDocuments found: {len(result['documents'])}")
    print(f"Feedback: {result['feedback']}")
    print(f"Answer: {result['answer'][:200]}...\n")
    print("-"*60)
