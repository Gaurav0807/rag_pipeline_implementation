# Simple LangGraph Examples

Learn LangGraph from basics to advanced workflows.

## What is LangGraph?

**LangGraph** builds workflows as graphs - nodes (work) connected by edges (flow).

```
┌─────────┐     ┌─────────┐     ┌─────────┐
│ Node 1  │────▶│ Node 2  │────▶│ Node 3  │
└─────────┘     └─────────┘     └─────────┘
  (work)         (work)          (work)
```

## Installation

```bash
pip install langgraph langchain langchain-community boto3
```

## Examples

### 1. Basic Graph (`simple_1_basic_graph.py`)
- Create nodes (functions)
- Connect with edges
- Run a simple workflow

**Output:**
```
Input → Process → LLM → Format → Output
```

**Run:**
```bash
python simple_1_basic_graph.py
```

---

### 2. Conditional Routing (`simple_2_conditional_routing.py`)
- Route to different nodes based on conditions
- Classify input and handle each type differently

**Flow:**
```
Classify
├─ Programming → Python Expert
├─ RAG → RAG Expert
└─ General → General Assistant
```

**Run:**
```bash
python simple_2_conditional_routing.py
```

---

### 3. StateGraph (`simple_3_state_graph.py`)
- Modern LangGraph way with typed state
- Define what data flows through

**Key Concept:**
```python
class State(TypedDict):
    question: str
    documents: list
    answer: str
```

**Run:**
```bash
python simple_3_state_graph.py
```

---

### 4. RAG Workflow (`simple_4_rag_workflow.py`)
- Full RAG pipeline as a graph
- Retrieve → Check → Generate

**Flow:**
```
Retrieve Docs
      ↓
Check if relevant
  ├─ YES → Generate Answer
  └─ NO → LLM without Docs
      ↓
    Return Answer
```

**Run:**
```bash
python simple_4_rag_workflow.py
```

---

### 5. Agent Loop (`simple_5_agent_loop.py`)
- Agent that iterates and remembers
- Think → Plan → Execute → Repeat

**Flow:**
```
Analyze
  ↓
Plan
  ↓
Execute
  ↓
Should Continue?
  ├─ YES → Back to Analyze
  └─ NO → Finish
```

**Run:**
```bash
python simple_5_agent_loop.py
```

---

## Key Differences: LangChain vs LangGraph

| Aspect | LangChain | LangGraph |
|--------|-----------|-----------|
| **What it is** | Chaining components | Building graphs/workflows |
| **Complexity** | Simple linear flows | Complex workflows, loops |
| **State** | Data flows through chain | Explicit state object |
| **Routing** | Basic branching | Conditional routing, cycles |
| **Use Case** | RAG chains, prompting | Multi-step agents, workflows |
| **Code Style** | Pipe: `A \| B \| C` | Graph nodes and edges |

---

## LangGraph vs LangChain Example

### LangChain (Simple)
```python
chain = retriever | prompt | llm
result = chain.invoke(question)
```

### LangGraph (Complex)
```python
workflow.add_node("retrieve", retrieve_fn)
workflow.add_node("generate", generate_fn)
workflow.add_conditional_edges("retrieve", route_fn, {...})
app = workflow.compile()
result = app.invoke(state)
```

---

## Common Patterns

### Pattern 1: Linear Workflow
```
Node A → Node B → Node C → END
```

### Pattern 2: Conditional Routing
```
Node A → (if X) → Node B
       → (if Y) → Node C
```

### Pattern 3: Loop
```
Node A → Node B → (Continue?) → back to A
                 → (No) → END
```

### Pattern 4: Fan-out
```
      ┌─ Node B
Node A┼─ Node C
      └─ Node D
        ↓ (all run, then merge)
      Node E
```

---

## StateGraph (Recommended)

Use `StateGraph` for modern LangGraph:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class MyState(TypedDict):
    input: str
    output: str

workflow = StateGraph(MyState)
workflow.add_node("my_node", my_function)
workflow.add_edge("my_node", END)
app = workflow.compile()
```

---

## When to Use LangGraph

✅ Use LangGraph when you need:
- Multi-step workflows
- Conditional branching
- Loops and iterations
- Agent-like behavior
- Complex state management

❌ Use LangChain when you need:
- Simple pipelines
- Quick RAG chains
- Linear data flow
- No branching logic

---

## Debugging

Add `verbose=True`:
```python
app = workflow.compile()
result = app.invoke(state, {"recursion_limit": 25})
```

Visualize the graph:
```python
from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))
```

---

## Quick Reference

| Function | Purpose |
|----------|---------|
| `add_node(name, func)` | Add a work node |
| `add_edge(from, to)` | Direct connection |
| `add_conditional_edges(from, func, map)` | Route based on condition |
| `set_entry_point(name)` | Where to start |
| `set_finish_point(name)` | Where to end |
| `compile()` | Build the graph |
| `invoke(state)` | Run the graph |

---

## Next Steps

1. Run examples 1-5 in order
2. Modify example 4 to add your own retriever
3. Create your own graph combining RAG + Agent
4. Add memory to track conversation history
