# Simple LangChain Examples

Learn LangChain step-by-step with minimal code.

## Installation

```bash
pip install langchain langchain-community boto3
```

## Examples

### 1. Basic LLM (`simple_1_basic_llm.py`)
- Direct LLM calls
- Prompt templates
- System messages

**Run:**
```bash
python simple_1_basic_llm.py
```

---

### 2. Conversation Memory (`simple_2_memory.py`)
- Multi-turn conversations
- Chat history
- LLM remembers context

**Run:**
```bash
python simple_2_memory.py
```

---

### 3. RAG Chain (`simple_3_rag_chain.py`)
- Retrieve documents
- Generate answers
- Full RAG pipeline

**Run:**
```bash
python simple_3_rag_chain.py
```

---

### 4. Document Loading (`simple_4_document_loader.py`)
- Load PDFs
- Load text files
- Split into chunks
- Create documents with metadata

**Run:**
```bash
python simple_4_document_loader.py
```

---

### 5. Vector Store (`simple_5_vector_store.py`)
- Create embeddings
- Store in FAISS
- Similarity search
- Create retriever

**Run:**
```bash
python simple_5_vector_store.py
```

---

## Key Concepts

| Concept | What it does |
|---------|------------|
| **LLM** | Calls the language model |
| **Prompt** | Template for input |
| **Memory** | Stores chat history |
| **Retriever** | Finds relevant documents |
| **VectorStore** | Stores embeddings, enables search |
| **Chain** | Connects components (Retriever → Prompt → LLM) |

---

## Quick Comparison

```
Simple LLM:
Question → LLM → Answer

With Memory:
Question → Memory (add history) → LLM → Answer

With RAG:
Question → Retrieve docs → Prompt (add context) → LLM → Answer

With Vector Store:
Documents → Embed → Store → Search by similarity → Retrieve → LLM → Answer
```

---

## Common Chain Pattern (LCEL)

**LangChain Expression Language** - Pipe components together:

```python
# Pattern: retriever | prompt | llm | parser
chain = retriever | prompt | llm | output_parser
result = chain.invoke(input)
```

This is equivalent to:
1. Retrieve documents using `retriever`
2. Format prompt using `prompt`
3. Call LLM using `llm`
4. Parse output using `output_parser`
