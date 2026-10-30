"""
Simple LangGraph Example 2: Conditional Routing
Route to different nodes based on conditions
"""

from langgraph.graph import Graph
from langchain_community.chat_models.bedrock import BedrockChat

llm = BedrockChat(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)

# Define nodes
def node_classify(state):
    """Classify the input"""
    print("Node 1: Classifying input...")
    question = state["question"].lower()

    if "python" in question:
        state["category"] = "programming"
    elif "rag" in question or "retrieval" in question:
        state["category"] = "rag"
    else:
        state["category"] = "general"

    print(f"  Category: {state['category']}")
    return state

def node_programming(state):
    """Handle programming questions"""
    print("Node 2a: Programming expert...")
    response = llm.invoke(f"As a Python expert, answer: {state['question']}")
    state["response"] = response.content
    return state

def node_rag(state):
    """Handle RAG questions"""
    print("Node 2b: RAG expert...")
    response = llm.invoke(f"As a RAG expert, answer: {state['question']}")
    state["response"] = response.content
    return state

def node_general(state):
    """Handle general questions"""
    print("Node 2c: General assistant...")
    response = llm.invoke(f"Answer: {state['question']}")
    state["response"] = response.content
    return state

def node_output(state):
    """Format output"""
    print("Node 3: Formatting...")
    state["final"] = f"Category: {state['category']}\nAnswer: {state['response']}"
    return state

# Routing function
def route_by_category(state):
    """Decide which node to go to"""
    if state["category"] == "programming":
        return "programming"
    elif state["category"] == "rag":
        return "rag"
    else:
        return "general"

# Create graph
graph = Graph()

# Add nodes
graph.add_node("classify", node_classify)
graph.add_node("programming", node_programming)
graph.add_node("rag", node_rag)
graph.add_node("general", node_general)
graph.add_node("output", node_output)

# Add edges
graph.add_edge("classify", "programming", condition=lambda s: s["category"] == "programming")
graph.add_edge("classify", "rag", condition=lambda s: s["category"] == "rag")
graph.add_edge("classify", "general", condition=lambda s: s["category"] == "general")

# Using conditional_edges (cleaner way)
graph.add_conditional_edges(
    "classify",
    route_by_category,
    {
        "programming": "programming",
        "rag": "rag",
        "general": "general",
    }
)

graph.add_edge("programming", "output")
graph.add_edge("rag", "output")
graph.add_edge("general", "output")

graph.set_entry_point("classify")
graph.set_finish_point("output")

app = graph.compile()

# Test it
print("="*60)
print("Conditional Routing Example")
print("="*60)

test_cases = [
    {"question": "How do I write Python decorators?"},
    {"question": "What is RAG and how does it work?"},
    {"question": "What is the capital of France?"}
]

for test in test_cases:
    print(f"\n🔹 Question: {test['question']}")
    result = app.invoke(test)
    print(f"Answer: {result['response'][:150]}...\n")
