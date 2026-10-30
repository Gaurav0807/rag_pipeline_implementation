"""
Simple LangGraph Example 5: Agent Loop with Memory
Agent that can iterate and remember decisions
"""

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_community.chat_models.bedrock import BedrockChat
from datetime import datetime

llm = BedrockChat(
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)

# State schema with memory
class AgentState(TypedDict):
    task: str
    thoughts: List[str]
    actions_taken: List[str]
    iteration: int
    max_iterations: int
    final_result: str
    done: bool

# Nodes
def node_analyze(state: AgentState) -> AgentState:
    """Analyze the task"""
    print(f"\n🔄 Iteration {state['iteration']}")
    print("📋 Analyzing task...")

    prompt = f"Analyze this task and think about the next step:\n\nTask: {state['task']}\n\nWhat should I do next?"
    response = llm.invoke(prompt)
    thought = response.content

    state["thoughts"].append(f"[Iteration {state['iteration']}] {thought}")
    state["actions_taken"].append(f"Analyzed: {state['task'][:50]}...")

    print(f"   Thought: {thought[:80]}...")
    return state

def node_plan(state: AgentState) -> AgentState:
    """Plan the next action"""
    print("📝 Planning...")

    prompt = f"""Based on this thought, what's the concrete action?

Thought: {state['thoughts'][-1]}

Action:"""
    response = llm.invoke(prompt)
    action = response.content

    state["actions_taken"].append(f"Planned: {action[:50]}...")
    print(f"   Action: {action[:80]}...")
    return state

def node_execute(state: AgentState) -> AgentState:
    """Execute the plan"""
    print("⚡ Executing...")

    last_action = state["actions_taken"][-1]
    prompt = f"Execute this action: {last_action}\n\nResult:"
    response = llm.invoke(prompt)
    result = response.content

    state["actions_taken"].append(f"Executed: {result[:50]}...")
    print(f"   Result: {result[:80]}...")
    return state

def node_should_continue(state: AgentState) -> str:
    """Decide if we should continue or finish"""
    print("🤔 Should we continue?")

    state["iteration"] += 1

    if state["iteration"] >= state["max_iterations"]:
        print("   ✅ Max iterations reached - finishing")
        return "finish"
    else:
        print(f"   ▶️  Continue (iteration {state['iteration']}/{state['max_iterations']})")
        return "continue"

def node_finish(state: AgentState) -> AgentState:
    """Finalize the result"""
    print("\n✅ Finishing...")

    summary = f"""
Task: {state['task']}

Iterations: {state['iteration']}

Thoughts:
{chr(10).join(state['thoughts'])}

Actions:
{chr(10).join(state['actions_taken'])}

Final Result: Agent completed the task analysis
"""
    state["final_result"] = summary
    state["done"] = True
    return state

# Create workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("analyze", node_analyze)
workflow.add_node("plan", node_plan)
workflow.add_node("execute", node_execute)
workflow.add_node("finish", node_finish)

# Add edges
workflow.add_edge("analyze", "plan")
workflow.add_edge("plan", "execute")
workflow.add_conditional_edges(
    "execute",
    node_should_continue,
    {
        "continue": "analyze",
        "finish": "finish"
    }
)
workflow.add_edge("finish", END)

workflow.set_entry_point("analyze")

app = workflow.compile()

# Run it
print("="*60)
print("Agent Loop Example")
print("="*60)

initial_state = {
    "task": "Build a recommendation system for movie streaming",
    "thoughts": [],
    "actions_taken": [],
    "iteration": 1,
    "max_iterations": 3,
    "final_result": "",
    "done": False
}

result = app.invoke(initial_state)

print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)
print(result["final_result"])
print("="*60)
