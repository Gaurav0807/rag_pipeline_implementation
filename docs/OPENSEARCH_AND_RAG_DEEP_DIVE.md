# OpenSearch & RAG Pipeline - Deep Dive Q&A

## 1. How Do You Update Documents? Does Re-ingestion Duplicate Vectors?

### How Updates Work

This project uses **AWS Bedrock Knowledge Base** managed ingestion. The flow is:

1. You upload/modify files in S3 (`s3://gaurav-hudi-data/rag_test_data/`)
2. You trigger a **sync job** (via `StartIngestionJob` API or console)
3. Bedrock compares the current S3 state against what was previously ingested

During sync, Bedrock handles three cases:
- **New files** → chunks them, generates embeddings, writes new vectors to OpenSearch
- **Modified files** (changed content/timestamp) → deletes the old vectors for that file, re-chunks, re-embeds, writes new vectors
- **Deleted files** → removes their vectors from OpenSearch

Bedrock tracks document identity using the **S3 object key** stored in `AMAZON_BEDROCK_METADATA`.

### Does Re-ingestion Duplicate Vectors?

**No** — if you re-run an ingestion job without changing the S3 files, Bedrock skips unchanged documents. It won't create duplicate vectors.

However, there are nuances:

| Scenario | Result |
|----------|--------|
| Re-sync with no S3 changes | No duplicates, unchanged docs skipped |
| Modify a file and re-sync | Old vectors deleted, new ones created (no duplicates) |
| Upload same content with a *new* S3 key | **Will create duplicates** — Bedrock sees it as a new document |
| Delete from S3 but don't sync | Orphaned vectors remain in OpenSearch until next sync |

---

## 2. How Does Bedrock "Remember" What It Already Processed?

Bedrock Knowledge Base maintains an **internal crawl state** (managed by AWS, not visible to you).

### Step-by-step:

```
Ingestion Job Starts
        |
        v
+-----------------------------+
| 1. List all S3 objects in   |
|    the configured prefix    |
|    (rag_test_data/*)        |
+-------------+---------------+
              |
              v
+-----------------------------+
| 2. For each file, check     |
|    S3 ETag (content hash)   |
|    + LastModified timestamp  |
|    against internal state    |
+-------------+---------------+
              |
     +--------+---------+
     v        v         v
  Same?    Changed?   New file?
  SKIP     DELETE old  CREATE new
           chunks ->   chunks
           CREATE new
```

### The two pieces that enable this:

**1. AWS internal crawl state (you don't see this)**
- Bedrock stores a mapping: `S3 key -> ETag at last successful ingestion`
- Next job compares current ETag vs stored ETag
- If ETag matches -> file content hasn't changed -> skip

**2. Metadata in OpenSearch (you can see this)**

The `AMAZON_BEDROCK_METADATA` field in each vector stores the source S3 URI:

```json
{"source": "s3://gaurav-hudi-data/rag_test_data/jira/ENG-1001.json"}
```

This is how Bedrock knows **which chunks in OpenSearch belong to which source file** — so when a file changes, it can find and delete all the old chunks for that specific file before writing new ones.

---

## 3. Does OpenSearch Maintain History of Data?

**No — Bedrock + OpenSearch is a latest-state-only system.**

```
Day 1: Upload ENG-1001.json (status: OPEN)
       -> Sync -> Creates chunks in OpenSearch

Day 2: Update ENG-1001.json (status: IN_PROGRESS)  
       -> Sync -> DELETES Day 1 chunks -> Creates NEW chunks
       
Day 3: Update ENG-1001.json (status: DONE)
       -> Sync -> DELETES Day 2 chunks -> Creates NEW chunks
```

The Day 1 and Day 2 data is **gone forever** from OpenSearch. You can only search the latest version.

### If You Want History — Design for It

**Option 1: Version in the S3 key**

```
rag_test_data/jira/ENG-1001_v1.json   (OPEN)
rag_test_data/jira/ENG-1001_v2.json   (IN_PROGRESS)
rag_test_data/jira/ENG-1001_v3.json   (DONE)
```

Bedrock sees 3 different files -> creates chunks for all 3 -> you can search across historical states.

**Option 2: Append-only with timestamps**

```
rag_test_data/jira/ENG-1001_2026-05-01.json
rag_test_data/jira/ENG-1001_2026-05-10.json
rag_test_data/jira/ENG-1001_2026-05-13.json
```

| Approach | History preserved? | Search returns |
|----------|-------------------|----------------|
| Overwrite same S3 key (current setup) | No | Only latest |
| New S3 key per version | Yes | All versions (need app-level filtering) |

---

## 4. Knowledge Base vs Data Source — Terraform

### Knowledge Base (`knowledge_base.tf`)

```hcl
resource "aws_bedrockagent_knowledge_base" "main" {
    name     = "${local.resource_prefix}-kb"
    role_arn = aws_iam_role.kb_role.arn

    # HOW: Which model converts text -> vectors
    knowledge_base_configuration {
        type = "VECTOR"
        vector_knowledge_base_configuration {
            embedding_model_arn = ".../${var.embedding_model}"
        }
    }

    # WHERE: Where vectors get stored
    storage_configuration {
        type = "OPENSEARCH_SERVERLESS"
        opensearch_serverless_configuration {
            collection_arn    = aws_opensearchserverless_collection.vectors.arn
            vector_index_name = local.index_name
            field_mapping { ... }
        }
    }
}
```

Answers two questions:

| Question | Answer |
|----------|--------|
| **HOW** to create embeddings? | Titan embedding model (1024 dimensions) |
| **WHERE** to store vectors? | OpenSearch Serverless collection + index |

### Data Source (`data_source.tf`)

```hcl
resource "aws_bedrockagent_data_source" "main" {
    name              = "${local.resource_prefix}-datasource"
    knowledge_base_id = aws_bedrockagent_knowledge_base.main.id

    # FROM WHERE: Which S3 location has the documents
    data_source_configuration {
        type = "S3"
        s3_configuration {
            bucket_arn         = "arn:aws:s3:::${var.s3_bucket_name}"
            inclusion_prefixes = [var.s3_data_prefix]
        }
    }

    # HOW: How to split documents before embedding
    vector_ingestion_configuration {
        chunking_configuration {
            chunking_strategy = "FIXED_SIZE"
            fixed_size_chunking_configuration {
                max_tokens         = var.chunk_size      # 512 tokens per chunk
                overlap_percentage = var.chunk_overlap   # 10% overlap
            }
        }
    }
}
```

Answers two different questions:

| Question | Answer |
|----------|--------|
| **FROM WHERE** to read documents? | S3 bucket + prefix |
| **HOW** to chunk them? | Fixed-size, 512 tokens, 10% overlap |

### The Relationship

```
+---------------------------------------------+
|           KNOWLEDGE BASE                     |
|                                              |
|  - Embedding model: Titan                    |
|  - Vector store: OpenSearch                  |
|                                              |
|  +---------------+  +---------------+        |
|  | Data Source 1 |  | Data Source 2 |  ...   |
|  | S3://jira/    |  | S3://confluence/|       |
|  | 512 tokens    |  | 256 tokens     |       |
|  +---------------+  +---------------+        |
+---------------------------------------------+
```

Key points:
- **One KB can have multiple data sources**
- **KB is created first** — it has no documents, just the "processing pipeline" config
- **Data Source is attached to a KB** — via `knowledge_base_id`
- **Ingestion job runs per data source** — you sync each data source independently

---

## 5. Why 1024 Dimensions? What Are the Available Embedding Models?

### What Dimensions Mean

Dimensions = how much detail the model captures about meaning.

```
"Kafka consumer lag is causing data delays"

256 dimensions:  [0.12, -0.45, 0.78, ... 256 numbers]   -> rough sketch
1024 dimensions: [0.12, -0.45, 0.78, ... 1024 numbers]  -> detailed portrait
```

### 256 vs 1024 Tradeoff

| Factor | 256 dimensions | 1024 dimensions |
|--------|---------------|-----------------|
| Search accuracy | Lower — similar concepts may get confused | Higher — fine-grained distinction |
| Storage cost | 4x less space in OpenSearch | 4x more space |
| Search speed | Faster (fewer numbers to compare) | Slower |
| Cost per embedding | Cheaper | More expensive |

### AWS Bedrock Embedding Models Available

| Model | Dimensions | Best For |
|-------|-----------|----------|
| **Titan Embed Text v1** | 1536 (fixed) | Legacy, no dimension choice |
| **Titan Embed Text v2** | 256, 512, or 1024 (you pick) | General purpose, flexible |
| **Cohere Embed English v3** | 1024 (fixed) | English-only, very accurate |
| **Cohere Embed Multilingual v3** | 1024 (fixed) | Multi-language support |

### Why Titan and Not Cohere?

| Factor | Titan v2 | Cohere |
|--------|----------|--------|
| Cost | Cheaper (~$0.00002/1K tokens) | More expensive (~$0.0001/1K tokens) |
| AWS integration | Native — no extra setup | Third-party, needs marketplace acceptance |
| Dimension flexibility | Choose 256/512/1024 | Fixed at 1024 |
| Quality | Good | Slightly better for nuanced search |
| Multilingual | Basic | Excellent (100+ languages) |

---

## 6. Embedding Cost Explained ($0.00002 per 1K Tokens)

### What's a token?

```
"Kafka consumer lag is causing delays"

Tokens: ["Kafka", " consumer", " lag", " is", " causing", " delays"]
         = 6 tokens
```

Roughly: **1 token ~ 0.75 words** (or ~4 characters)

### Cost for this project

```
50 Jira tickets    -> ~200 words each -> 10,000 words
20 Confluence pages -> ~100 words each ->  2,000 words
                                 Total: 12,000 words

12,000 words / 0.75 = ~16,000 tokens

Cost = 16,000 / 1,000 x $0.00002 = $0.00000032
```

Entire project costs a fraction of a cent to embed.

### Cost at scale

| Scale | Tokens | Cost |
|-------|--------|------|
| This project (70 docs) | ~16K | $0.0003 (basically free) |
| 10,000 documents | ~5M | $0.10 |
| 1 million documents | ~500M | $10 |
| Re-ingestion (same data) | $0 | Skipped — not re-embedded |

### When you get charged

You're only charged when text is **actually sent to the embedding model** — not for storage, not for searching. The total costs are spread across multiple services:

| Service | What you pay for | Pricing model |
|---------|-----------------|---------------|
| Bedrock (Titan Embed) | Converting text -> vectors | Per token (during ingestion) |
| Bedrock (Claude/Titan Text) | Generating the final answer | Per token (during query) |
| OpenSearch Serverless | Storing vectors + running searches | Per hour (compute) + per GB (storage) |
| S3 | Storing raw documents | Per GB (storage) + per request |
| Lambda | Running query function | Per invocation + duration |

---

## 7. HNSW Algorithm — How It Works

### Where It's Used

In the OpenSearch index creation (`boto3_script.py`):

```python
"AMAZON_BEDROCK_VECTOR": {
    "type": "knn_vector",
    "dimension": 1024,
    "method": {
        "name": "hnsw",
        "engine": "faiss",
        "space_type": "l2",
        "parameters": {
            "ef_construction": 512,
            "m": 16
        }
    }
}
```

### The Problem HNSW Solves

You have 1000 vectors stored. A user query comes in as a vector. You need to find the closest ones.

- **Brute force:** Compare query against ALL 1000 vectors -> exact but slow at scale
- **HNSW:** Build a smart structure so you only compare against ~50 vectors -> approximate but extremely fast

### How It Works — Airport Analogy

Imagine you're in **Delhi** and need to reach a **small village near Munich, Germany**.

**Without HNSW (Brute Force):** Check distance from Delhi to every single village in the world.

**With HNSW (Hierarchical Search):**

```
Layer 2 -- International Flights (few airports, long jumps)

   Delhi ------- London ------- Frankfurt ------- Tokyo
                                    *
                            "Frankfurt is closest to Munich"


Layer 1 -- Domestic Flights (more cities, shorter jumps)

   Frankfurt ---- Munich ---- Stuttgart ---- Berlin
                    *
               "Munich is closest"


Layer 0 -- Local Roads (all villages, tiny jumps)

   Munich -- Starnberg -- Furstenfeldbruck -- Dachau -- Erding
                                *
                        "This is your village!"
```

**You visited maybe 10 places total instead of checking millions.**

### Applied to This Project

```
Query vector: "Kafka consumer lag"  ->  [0.12, -0.45, ... 1024 numbers]

Layer 2 (few vectors, far apart):
   "Kafka issues" cluster --- "Deployment" cluster --- "Monitoring" cluster
         *

Layer 1 (more vectors):
   "Kafka lag" --- "Kafka config" --- "Kafka scaling"
       *

Layer 0 (all 70 docs):
   ENG-1001 (lag) --- ENG-1023 (lag) --- DATA-1005 (lag)
       *                   *
   
   Returns these two as top matches
```

### Parameters Explained

| Parameter | What it does | Value | Analogy |
|-----------|-------------|-------|---------|
| `m = 16` | Connections per node | 16 | Each airport has direct flights to 16 other airports |
| `ef_construction = 512` | Search depth when building | 512 | When building a new airport, evaluate 512 possible cities to connect to, then pick the best 16 |

### Performance

| Documents | Brute Force | HNSW |
|-----------|-------------|------|
| 70 (this project) | <1ms | <1ms (no difference) |
| 100,000 | ~500ms | ~5ms |
| 10,000,000 | ~50 seconds | ~10ms |

### Why "Approximate"

With HNSW you might get the 2nd closest result instead of the absolute closest — because you never checked every single vector. But it's 99% as good and 1000x faster. For semantic search, this is perfectly fine.

---

## 8. L2 Distance vs Cosine Similarity

Both measure "how close are two vectors?" — but differently.

### L2 Distance (Euclidean Distance)

Measures the **straight-line distance** between two points. Cares about both direction AND magnitude (length).

```
         You (3,4)
          *
         /|
        / |
       /  | distance = sqrt((3-1)^2 + (4-1)^2) = 3.6
      /   |
     *----+
   Doc (1,1)
```

### Cosine Similarity

Measures the **angle** between two vectors. Ignores length entirely.

```
                     * A (far but same direction)
                    /
                   /  angle = 0 degrees -> cosine = 1.0 (identical meaning)
                  /
                 /
   Origin ------* B (close but same direction)
```

### Key Difference

```
3 documents about Kafka:

Doc A: "Kafka consumer lag" (long article, 2000 words)  -> long vector
Doc B: "Kafka consumer lag" (short ticket, 50 words)    -> short vector  
Doc C: "Lambda deployment" (long article)               -> different direction
```

| Comparison | L2 Distance | Cosine Similarity |
|-----------|-------------|-------------------|
| Doc A vs Doc B | Far apart (different lengths) | Very similar (same direction) |
| Doc A vs Doc C | Might be close (similar length) | Very different (different direction) |

### Which Is Better for RAG/Semantic Search?

**Cosine is generally better** — because you care about meaning, not document length.

### Why This Project Uses L2 Anyway

```python
"space_type": "l2"   # in boto3_script.py
```

This works because **Bedrock's Titan model normalizes vectors** — all output vectors have the same length (magnitude = 1). When all vectors are normalized:

```
L2^2(A,B) = 2 - 2 x cosine(A,B)

So ranking by L2 = ranking by cosine (just flipped)
Lower L2 = Higher cosine = More similar
```

**They produce identical rankings when vectors are normalized.**

### When to Use Which

| Use Case | Better Choice |
|----------|--------------|
| Find documents with similar meaning (normalized vectors) | Either — same result |
| Find documents with similar meaning (non-normalized) | Cosine |
| Compare signals where magnitude matters (price, frequency) | L2 |

If you ever switch to a model that doesn't normalize, change to `"space_type": "cosinesimil"` in the index mapping.
