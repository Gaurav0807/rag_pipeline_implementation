"""
Simple LangGraph Example 1: Basic Graph
Create a workflow with nodes and edges
Install: pip install langgraph langchain langchain-community boto3
"""

from langgraph.graph import Graph
from langchain_community.chat_models.bedrock import BedrockChat

llm = BedrockChat(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)

# Define nodes (functions that do work)
def node_input(state):
    """First node - takes user input"""
    print("Node 1: Processing input...")
    state["processed_input"] = state["input"].upper()
    return state

def node_llm(state):
    """Second node - calls LLM"""
    print("Node 2: Calling LLM...")
    response = llm.invoke(f"Explain: {state['processed_input']}")
    state["llm_response"] = response.content
    return state

def node_output(state):
    """Third node - formats output"""
    print("Node 3: Formatting output...")
    state["final_output"] = f"Answer: {state['llm_response']}"
    return state

# Create graph
graph = Graph()

# Add nodes
graph.add_node("input", node_input)
graph.add_node("llm", node_llm)
graph.add_node("output", node_output)

# Add edges (connections between nodes)
graph.add_edge("input", "llm")
graph.add_edge("llm", "output")

# Set entry point
graph.set_entry_point("input")

# Set finish point
graph.set_finish_point("output")

# Compile graph
app = graph.compile()

# Run it
print("="*60)
print("Basic Graph Example")
print("="*60)

initial_state = {"input": "What is RAG?"}
result = app.invoke(initial_state)

print("\nFinal Output:")
print(result["final_output"])
