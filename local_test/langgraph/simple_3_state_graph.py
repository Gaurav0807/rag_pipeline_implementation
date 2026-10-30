"""
Simple LangGraph Example 3: StateGraph
Modern LangGraph way with typed state
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock

llm = ChatBedrock(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)

# Define state schema (what data flows through the graph)
class State(TypedDict):
    question: str
    retrieved_docs: list
    answer: str
    messages: list

# Define nodes
def retrieve_docs(state: State) -> State:
    """Retrieve relevant documents"""
    print("\n" + "="*70)
    print("NODE 1: RETRIEVE DOCS")
    print("="*70)
    print(f"INPUT STATE:")
    print(f"  question: {state['question']}")
    print(f"  retrieved_docs: {state['retrieved_docs']}")
    print(f"  messages: {state['messages']}")

    state["retrieved_docs"] = [
        "RAG combines retrieval and generation.",
        "It improves LLM accuracy with external knowledge.",
    ]
    state["messages"] = [f"Retrieved {len(state['retrieved_docs'])} documents"]

    print(f"\nOUTPUT STATE (after retrieve):")
    print(f"  retrieved_docs: {state['retrieved_docs']}")
    print(f"  messages: {state['messages']}")
    return state

def generate_answer(state: State) -> State:
    """Generate answer using retrieved docs"""
    print("\n" + "="*70)
    print("NODE 2: GENERATE ANSWER")
    print("="*70)
    print(f"INPUT STATE:")
    print(f"  question: {state['question']}")
    print(f"  retrieved_docs: {state['retrieved_docs']}")
    print(f"  messages: {state['messages']}")

    context = "\n".join(state["retrieved_docs"])
    prompt = f"""Answer based on this context:
{context}

Question: {state['question']}
Answer:"""

    print(f"\nPROMPT SENT TO LLM:")
    print(f"{prompt}")

    response = llm.invoke(prompt)
    state["answer"] = response.content
    state["messages"].append("Generated answer")

    print(f"\nOUTPUT STATE (after generate):")
    print(f"  answer: {state['answer'][:100]}...")
    print(f"  messages: {state['messages']}")
    return state

def format_output(state: State) -> State:
    """Format final output"""
    print("📝 Formatting output...")
    state["messages"].append("Formatted output")
    return state

# Create state graph
workflow = StateGraph(State)
print("Workflow start")
print(workflow)
print("Workflow end")

# Add nodes
workflow.add_node("retrieve", retrieve_docs)
workflow.add_node("generate", generate_answer)
workflow.add_node("format", format_output)

# Add edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "format")
workflow.add_edge("format", END)

# Set entry point
workflow.set_entry_point("retrieve")

# Compile
app = workflow.compile()

# Run it
print("="*60)
print("StateGraph Example")
print("="*60)

initial_state = {
    "question": "What is RAG?",
    "retrieved_docs": [],
    "answer": "",
    "messages": []
}

result = app.invoke(initial_state)

print("\n" + "="*60)
print("Result:")
print("="*60)
print(f"Question: {result['question']}")
print(f"\nRetrieved Documents: {result['retrieved_docs']}")
print(f"\nAnswer: {result['answer']}")
print(f"\nMessages: {result['messages']}")
