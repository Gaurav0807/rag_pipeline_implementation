"""
Simple LangChain Example 4: Load Documents
Load PDFs, text files, URLs
"""

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Method 1: Load PDF
print("="*60)
print("Loading PDF")
print("="*60)

try:
    pdf_loader = PyPDFLoader("document.pdf")
    pages = pdf_loader.load()
    print(f"Loaded {len(pages)} pages")
    print(f"First page:\n{pages[0].page_content[:200]}...")
except Exception as e:
    print(f"PDF not found: {e}")

# Method 2: Load text file
print("\n" + "="*60)
print("Loading Text File")
print("="*60)

try:
    text_loader = TextLoader("sample.txt")
    documents = text_loader.load()
    print(f"Loaded {len(documents)} documents")
except Exception as e:
    print(f"Text file not found: {e}")

# Method 3: Split documents into chunks
print("\n" + "="*60)
print("Splitting Documents")
print("="*60)

# Create sample documents
sample_docs = [
    Document(page_content="RAG stands for Retrieval Augmented Generation. It combines retrieval and generation."),
    Document(page_content="Transformers use self-attention mechanisms for parallel processing.")
]

# Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,        # Characters per chunk
    chunk_overlap=20       # Overlap between chunks
)

chunks = splitter.split_documents(sample_docs)

print(f"Created {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.page_content}")
    print()

# Method 4: Create documents from text
print("="*60)
print("Creating Documents from Text")
print("="*60)

documents = [
    Document(page_content="Spark is a distributed computing framework.", metadata={"source": "wiki"}),
    Document(page_content="OpenSearch is a vector database.", metadata={"source": "docs"}),
]

print(f"Created {len(documents)} documents with metadata")
for doc in documents:
    print(f"  - {doc.page_content}")
    print(f"    Metadata: {doc.metadata}")
