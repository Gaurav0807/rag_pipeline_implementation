# RAG Pipeline Interview Questions & Answers

---

## Level 1: Basic Understanding

### Q1: What is RAG? Why do we need it?

**Answer:**
RAG = Retrieval Augmented Generation. LLMs have a knowledge cutoff and don't know your private data. RAG lets you feed relevant documents to the LLM at query time so it can answer using YOUR data, not just its training data.

Without RAG:
- User asks: "What's the status of ticket ENG-1001?"
- LLM says: "I don't have access to your internal systems"

With RAG:
- User asks same question
- System retrieves ENG-1001 from vector store
- LLM reads it and says: "ENG-1001 is a Kafka consumer lag issue, status OPEN, priority HIGH"

---

### Q2: What is a vector embedding?

**Answer:**
A vector embedding is a list of numbers (e.g., 1024 numbers) that represents the MEANING of text. Similar text gets similar numbers.

```
"Kafka consumer lag"     → [0.23, -0.87, 0.45, ...]  
"Kafka topic issues"     → [0.21, -0.85, 0.44, ...]  ← very similar numbers!
"Team meeting notes"     → [-0.05, 0.08, 0.01, ...]  ← completely different
```

The embedding model (Titan Embed V2) learns these representations from billions of text examples during training.

---

### Q3: Why can't we just use SQL for AI search?

**Answer:**
SQL can only do exact or pattern matching:

```sql
SELECT * FROM docs WHERE content LIKE '%kafka%';
```

If user asks "messaging queue problems" — SQL finds NOTHING because the word "kafka" doesn't appear in the query.

Vector search understands that "messaging queue" and "Kafka" have similar MEANING because their vectors are close together in 1024-dimensional space.

| | SQL | Vector Search |
|--|-----|---------------|
| "messaging queue issues" finds Kafka docs? | No | Yes |
| "job crashed" finds "Spark failure" docs? | No | Yes |
| Searches by | Exact words | Meaning/semantics |

---

### Q4: What is OpenSearch Serverless? How is it different from managed?

**Answer:**

| | OpenSearch Managed | OpenSearch Serverless |
|--|---|---|
| You pick instance type? | Yes (t3.small, r6g.large) | No |
| You pick disk size? | Yes | No |
| Scaling | Manual (add nodes) | Automatic |
| Min cost | ~$30/month | ~$700/month |
| Management | You handle everything | AWS handles everything |
| When to use | Cost-sensitive, predictable load | Variable load, no ops team |

Serverless charges in OCUs (OpenSearch Compute Units). Minimum 4 OCUs always running.

---

### Q5: What does a Knowledge Base do in Bedrock?

**Answer:**
Knowledge Base is a configuration that automates the entire RAG pipeline:

During ingestion:
1. Reads files from S3
2. Chunks text (512 tokens, 10% overlap)
3. Calls Titan Embed model for each chunk
4. Stores vectors in OpenSearch

During query:
1. Embeds the question using same model
2. Searches OpenSearch for similar vectors
3. Returns relevant text chunks

Without KB, you write ~200 lines of code to do all this manually.

---

## Level 2: Architecture & Design

### Q6: Walk me through the data flow from document to answer

**Answer:**

**Ingestion flow (one-time/periodic):**
```
S3 file → Bedrock reads it → chunks into 512-token pieces → 
Titan Embed V2 converts each chunk to 1024-dim vector → 
stored in OpenSearch (vector + text + metadata)
```

**Query flow (every user question):**
```
User question → Titan embeds the question → 
OpenSearch finds nearest vectors (top 5) → 
returns matching text chunks → 
Claude reads chunks + question → generates grounded answer
```

---

### Q7: Why 512 token chunks? What if you use 100? What if 2000?

**Answer:**

512 is a balance:

| Chunk size | Pros | Cons |
|-----------|------|------|
| 100 tokens (small) | Very precise retrieval | Each chunk has little context, fragments don't make sense alone |
| 512 tokens (medium) | Good balance of precision and context | Standard choice |
| 2000 tokens (large) | Lots of context per chunk | Less precise, might return whole page when you need one paragraph |

Also: larger chunks = more expensive to embed and store, slower search.

---

### Q8: Why 10% overlap? What problem does it solve?

**Answer:**

Without overlap, you lose context at chunk boundaries:

```
WITHOUT overlap:
Chunk 1: "Kafka consumers experienced lag. The service was"
Chunk 2: "restarted and lag was resolved by scaling."

Problem: Neither chunk alone answers "how was Kafka lag fixed?"
```

```
WITH 10% overlap:
Chunk 1: "Kafka consumers experienced lag. The service was"
Chunk 2: "The service was restarted and lag was resolved by scaling."

Now Chunk 2 has enough context from Chunk 1.
```

10% is standard. Higher overlap = more redundancy but better context preservation.

---

### Q9: Why Titan Embed V2? Could you use a different model?

**Answer:**

Titan Embed V2 advantages:
- AWS-native (no external API calls, lower latency)
- 1024 dimensions (good balance of accuracy vs storage)
- Supports 8000 tokens input (handles large chunks)
- Integrated with Bedrock KB (no custom code)

Alternatives:
- Cohere Embed (also on Bedrock, 1024 dims)
- OpenAI ada-002 (1536 dims, external API)
- Sentence-transformers (open source, self-hosted)

You MUST use the same model for ingestion and query. If you embed docs with Titan, you must query with Titan.

---

### Q10: How does HNSW algorithm work at a high level?

**Answer:**

HNSW = Hierarchical Navigable Small World

Think of it as a multi-level graph:

```
Level 3 (few nodes):     A -------- B
Level 2 (more nodes):    A --- C --- B --- D
Level 1 (most nodes):    A-E-C-F-B-G-D-H-I
```

When searching:
1. Start at top level (few nodes, big jumps)
2. Find approximate area quickly
3. Move down to lower levels (more nodes, fine-grained)
4. Find exact nearest neighbors

Without HNSW: check ALL vectors (slow for millions of docs)
With HNSW: navigate the graph structure (fast, O(log n))

Parameters:
- `ef_construction = 512`: how many nodes to check when building graph (higher = better quality, slower build)
- `m = 16`: max connections per node (higher = better search, more memory)

---

## Level 3: Operational / Production

### Q11: How do you monitor if ingestion failed?

**Answer:**

Multiple layers of monitoring:

1. **Ingestion job status**: Check via API `GetIngestionJob` — returns COMPLETE/FAILED + statistics
2. **Lambda metrics**: CloudWatch logs from sync Lambda show errors
3. **CloudWatch alarms**: Alert on SyncFailed metric > 0
4. **Document counts**: Compare S3 file count vs OpenSearch document count
5. **EMF metrics**: Custom metrics (DocumentsScanned, DocumentsFailed, SyncDuration)

```python
stats = get_ingestion_job(...)["statistics"]
# numberOfDocumentsScanned: 70
# numberOfDocumentsFailed: 2  ← investigate these!
# numberOfNewDocumentsIndexed: 68
```

---

### Q12: How would you handle 1 million documents?

**Answer:**

Challenges at scale:
1. **Ingestion time**: 1M docs × 512 tokens/chunk = billions of embeddings
2. **Cost**: Titan embedding calls + OpenSearch storage
3. **Search latency**: More vectors = slower search

Solutions:
- **Multiple data sources**: Split by category (jira/, wiki/, etc.) — ingest in parallel
- **Incremental sync**: Only re-process new/modified files (Bedrock does this automatically)
- **Metadata filtering**: Pre-filter by category before vector search (reduces search space)
- **Larger OCUs**: Auto-scales but costs more
- **Chunking optimization**: Larger chunks = fewer vectors (but less precise)
- **Multiple KBs**: Separate KBs for different domains

---

### Q13: What happens if OpenSearch goes down?

**Answer:**

- **User queries fail**: Agent can't retrieve context, returns error or hallucinated answer
- **Ingestion fails**: New documents can't be indexed
- **Data is safe**: OpenSearch Serverless replicates data across AZs internally

Mitigation:
- CloudWatch alarm on VectorStoreAvailability metric
- Agent system prompt: "If you cannot retrieve information, say you're temporarily unable to answer"
- Fallback: Cache frequently asked questions
- OpenSearch Serverless has 99.9% SLA (rare downtime)

---

### Q14: How do you update documents? Does re-ingestion duplicate vectors?

**Answer:**

No duplication. When you re-run ingestion:
- **New files in S3**: Chunked and indexed (added)
- **Modified files**: Old vectors REPLACED with new ones
- **Deleted files from S3**: Vectors removed (if deletion policy allows)

Bedrock tracks files by S3 path. Same path = same document = update, not duplicate.

```
First ingestion:  s3://bucket/ENG-1001.json → indexed
Modify file:      s3://bucket/ENG-1001.json (changed content)
Re-ingest:        Old vectors for ENG-1001 replaced with new ones
Delete file:      s3://bucket/ENG-1001.json removed
Re-ingest:        Vectors for ENG-1001 removed from OpenSearch
```

---

### Q15: How do you handle sensitive/PII documents?

**Answer:**

Multiple layers:
1. **Encryption at rest**: OpenSearch encrypts stored data (AES-256)
2. **Encryption in transit**: HTTPS/TLS for all API calls
3. **IAM access control**: Only specific roles can query the KB
4. **Network policy**: VPC-only access (no public endpoint) in production
5. **Data access policy**: Only authorized principals can read OpenSearch
6. **S3 bucket policy**: Restrict who can upload documents
7. **Metadata filtering**: Tag sensitive docs, restrict at query time
8. **Guardrails**: Bedrock Guardrails can block PII from appearing in responses

---

## Level 4: Deep Technical

### Q16: What is L2 distance vs cosine similarity? Which is better?

**Answer:**

**L2 (Euclidean distance):**
- Straight-line distance between two points
- Considers both direction AND magnitude
- Formula: sqrt(sum((a[i] - b[i])^2))
- Lower = more similar

**Cosine similarity:**
- Angle between two vectors
- Only considers direction, ignores magnitude
- Formula: dot(a,b) / (||a|| * ||b||)
- Higher = more similar (range: -1 to 1)

**When to use which:**
- Cosine: When document length varies (long doc and short doc about same topic should match)
- L2: When embeddings are normalized (Titan V2 normalizes, so L2 and cosine give same results)

For Titan Embed V2 (normalized output): both work equally well. We use L2 because it's simpler.

---

### Q17: What are ef_construction and m parameters in HNSW?

**Answer:**

**ef_construction (512 in our case):**
- Used during INDEX BUILDING time
- "How many neighbors to check when adding a new node to the graph"
- Higher = better graph quality = better search accuracy
- Higher = slower ingestion time
- Typical range: 100-512

**m (16 in our case):**
- Maximum number of connections per node in the graph
- Higher = better search accuracy, but more memory usage
- Lower = faster search, less memory, slightly less accurate
- Typical range: 8-32

```
Low m (4):   A---B---C---D  (few connections, fast but might miss best match)
High m (32): A connects to B,C,D,E,F,G,... (many connections, accurate but heavy)
```

There's also **ef_search** (query time): how many nodes to visit during search. Higher = better results, slower query.

---

### Q18: Why 1024 dimensions? What if you use 256?

**Answer:**

| Dimensions | Accuracy | Storage | Speed |
|-----------|----------|---------|-------|
| 256 | Lower | 4x less | Faster |
| 512 | Medium | 2x less | Medium |
| 1024 | Higher | Baseline | Baseline |
| 1536 (OpenAI) | Highest | 1.5x more | Slower |

**1024 is the balance:**
- Enough dimensions to capture nuanced meaning
- Not so many that storage/search becomes expensive
- Titan V2 default output is 1024

**Lower dimensions (256):**
- Faster search, less storage
- May confuse similar-but-different concepts
- "Kafka lag" and "Kafka deployment" might look too similar

**Higher dimensions (1536):**
- Better at distinguishing subtle differences
- More expensive to store and search
- Diminishing returns past 1024 for most use cases

---

### Q19: How does the Agent decide when to query KB vs answer directly?

**Answer:**

The Agent uses an **orchestration prompt** (hidden system prompt) that tells it:

1. Read user question
2. Think: "Do I need external information to answer this?"
3. If yes → call KB retrieval (action group)
4. If no → answer directly from training knowledge

The decision is based on:
- System prompt instructions ("always use KB for company-specific questions")
- Question type: "What is Python?" → answer directly. "What's in ticket ENG-1001?" → query KB
- Knowledge base description: helps agent know WHAT is in the KB

You can influence this via system prompt:
```
"ALWAYS query the knowledge base before answering. 
Never answer from your general knowledge about internal topics."
```

---

### Q20: How would you evaluate retrieval quality?

**Answer:**

Metrics:
1. **Recall@K**: Of all relevant docs, how many appear in top-K results?
2. **Precision@K**: Of top-K results, how many are actually relevant?
3. **MRR (Mean Reciprocal Rank)**: How high is the first relevant result?
4. **NDCG**: Are relevant docs ranked higher than irrelevant ones?

Practical evaluation:
```
1. Create test set: 50 questions with known correct source documents
2. Run retrieval for each question
3. Check: did the correct document appear in top-5 results?
4. Calculate: precision, recall, average score

Example:
  Question: "Kafka lag issues"
  Expected: ENG-1001.json
  Retrieved top-5: [ENG-1001, DATA-1003, ENG-1045, ...]
  Score: ENG-1001 is #1 → MRR = 1.0 (perfect)
```

If quality is poor:
- Try smaller chunks (more precise)
- Try larger overlap (better context)
- Add metadata filtering
- Try different embedding model

---

## Level 5: Comparison / Decision Making

### Q21: RAG vs fine-tuning — when would you choose each?

**Answer:**

| | RAG | Fine-tuning |
|--|-----|-------------|
| Use when | Data changes frequently | Changing model behavior/style |
| Example | JIRA tickets (new ones daily) | "Always respond in our company's tone" |
| Data freshness | Always current (reads live data) | Frozen at training time |
| Cost | Per-query (retrieval + generation) | One-time training cost + inference |
| Accuracy | Cites specific sources | May hallucinate more |
| Setup | KB + vector store | Training data + fine-tune job |

**Choose RAG when:**
- Data changes often (tickets, docs, wikis)
- You need source citations
- You want accuracy over creativity

**Choose Fine-tuning when:**
- You want different model behavior (tone, format, domain language)
- Your knowledge is stable and won't change often
- You need faster inference (no retrieval step)

**Best: combine both** — fine-tune for style + RAG for fresh data.

---

### Q22: OpenSearch vs Pinecone vs pgvector — how do you choose?

**Answer:**

| | OpenSearch Serverless | Pinecone | pgvector |
|--|---|---|---|
| Managed by | AWS | Pinecone (SaaS) | You (PostgreSQL) |
| Bedrock integration | Native (built-in) | Manual (custom code) | Manual |
| Cost | ~$700/month min | Free tier available, then pay per usage | ~$15/month (RDS micro) |
| Scale | Auto | Auto | Manual |
| Vendor lock-in | AWS | Pinecone | None (open source) |
| Best for | AWS-native RAG with Bedrock | Multi-cloud, startup | Already using PostgreSQL, cost-sensitive |

**Choose OpenSearch** when: Using AWS Bedrock, want managed, enterprise budget
**Choose Pinecone** when: Want simplest setup, have free tier needs, multi-cloud
**Choose pgvector** when: Already have PostgreSQL, cost-sensitive, small scale

---

### Q23: Why Bedrock KB instead of building your own pipeline?

**Answer:**

**Build your own:**
```python
# You write and maintain ALL of this:
documents = read_from_s3()
chunks = chunk_documents(documents, size=512, overlap=10)
vectors = [call_embedding_api(chunk) for chunk in chunks]
store_in_vector_db(vectors)
# ... plus retry logic, error handling, monitoring, scaling
```

**Use Bedrock KB:**
```hcl
# You just configure:
resource "aws_bedrockagent_knowledge_base" "main" {
  # point to S3, point to OpenSearch, done
}
# Bedrock handles ALL the above automatically
```

Tradeoffs:
- Bedrock KB: Less control, but 10x faster to build, no code to maintain
- Custom: Full control over chunking, embedding, retrieval logic — needed for advanced use cases

---

### Q24: Single KB with multiple data sources vs multiple KBs?

**Answer:**

**Single KB, multiple data sources:**
```
KB: "company-kb"
  ├── Data Source: jira/
  ├── Data Source: confluence/
  └── Data Source: templates/
All vectors in same index → searched together
```
- Pros: Simpler, one query searches everything
- Cons: Can't give different KB configs (different embedding models, chunk sizes)

**Multiple KBs:**
```
KB: "jira-kb"     → OpenSearch index 1
KB: "wiki-kb"     → OpenSearch index 2
KB: "prd-kb"      → OpenSearch index 3
```
- Pros: Different configs per KB, agent can choose which KB to query
- Cons: More complex, agent must decide which KB to ask

**Rule of thumb:**
- Same type of content → single KB, multiple data sources
- Very different content types → multiple KBs (agent routes to the right one)

---

### Q25: How would you add real-time data (not just S3 files)?

**Answer:**

Bedrock KB currently only supports S3 as data source. For real-time:

**Option 1: Write to S3 + frequent sync**
```
Event happens → Lambda writes to S3 → EventBridge triggers sync every 15min
Delay: up to 15 minutes
```

**Option 2: Direct OpenSearch write (bypass KB)**
```
Event happens → Lambda embeds text → writes vector directly to OpenSearch
Delay: seconds (real-time)
Downside: you manage embedding logic yourself
```

**Option 3: Streaming ingestion (future)**
```
Bedrock may support streaming data sources in the future (Kafka, Kinesis)
Not available today.
```

For most use cases, Option 1 (S3 + 15-min sync) is sufficient.

---

## Level 6: Terraform / IaC Specific

### Q26: Why can't Terraform create the OpenSearch index directly?

**Answer:**

Terraform manages AWS resources via AWS APIs. But OpenSearch index creation is done via the **OpenSearch HTTP API** (REST endpoint), not the AWS API.

AWS API: "Create collection, create policy" → Terraform can do this
OpenSearch API: "PUT /index-name" → Terraform has no native resource for this

Workaround: `null_resource` with `local-exec` that runs a Python/curl script to call the OpenSearch HTTP API after the collection is created.

```hcl
resource "null_resource" "create_index" {
  provisioner "local-exec" {
    command = "python3 create_index.py"
  }
  depends_on = [aws_opensearchserverless_collection.vectors]
}
```

---

### Q27: Why use `depends_on` for Knowledge Base?

**Answer:**

Terraform creates resources in parallel by default. Without `depends_on`:

```
Terraform might try:
  Create KB  ← simultaneously → Create Collection
  KB says: "where's the OpenSearch collection?" → ERROR
```

With `depends_on`:
```
  Step 1: Create Collection (wait until done)
  Step 2: Create Index (wait until done)  
  Step 3: Create KB (now collection + index exist)
```

Terraform sometimes infers dependencies from resource references, but explicit `depends_on` guarantees order for resources that reference each other indirectly.

---

### Q28: How do you handle multi-environment (dev/staging/prod)?

**Answer:**

Using **tfvars files** per environment:

```
environments/
  sandbox.tfvars  → personal dev, mock data
  dev.tfvars      → real data, hourly sync
  staging.tfvars  → real data, 30-min sync
  prod.tfvars     → real data, 15-min sync, full monitoring
```

Deploy with:
```bash
terraform apply -var-file=environments/dev.tfvars
```

Resource naming includes environment:
```
rag-pipeline-sandbox-kb
rag-pipeline-dev-kb
rag-pipeline-prod-kb
```

Separate state files per environment (different S3 keys or backends).

---

### Q29: Why separate IAM roles for KB, Agent, and Lambda?

**Answer:**

**Least privilege principle**: each service gets ONLY the permissions it needs.

| Role | Can do | Cannot do |
|------|--------|-----------|
| KB Role | Read S3, call Titan, write OpenSearch | Cannot invoke Claude, cannot start Lambda |
| Agent Role | Invoke Claude, query KB | Cannot read S3 directly, cannot write OpenSearch |
| Lambda Role | Start ingestion jobs, write logs | Cannot invoke Claude, cannot read S3 |

Why this matters:
- If KB role is compromised → attacker can read S3 but can't invoke models or trigger jobs
- If Lambda role is compromised → attacker can start ingestion but can't read actual data
- Blast radius is limited per role

Also: different trust policies. KB and Agent roles are assumed by `bedrock.amazonaws.com`, Lambda role is assumed by `lambda.amazonaws.com`.

---

### Q30: How would you set up CI/CD for this pipeline?

**Answer:**

```
GitHub Actions / CodePipeline workflow:

1. PR opened:
   - terraform fmt -check
   - terraform validate
   - terraform plan (show what will change)
   - pytest (Lambda unit tests)

2. PR merged to main:
   - terraform apply (dev environment)
   - Run integration test (trigger ingestion, query agent)
   - If pass → terraform apply (staging)
   - Manual approval → terraform apply (prod)

3. System prompt changes:
   - Update .txt file → PR → merge
   - Terraform detects change → re-prepares agent
   - Creates new agent version automatically
```

Key considerations:
- Terraform state in S3 with DynamoDB locking
- Assume role for each environment (cross-account)
- Agent version management (rollback if new prompt fails)
- Ingestion testing after deployment

---

## Scenario Questions

### "Your RAG is returning irrelevant results. How do you debug?"

**Answer:**

Step-by-step debugging:

```
1. Check ingestion:
   - Did all documents get indexed? (compare S3 count vs OpenSearch _count)
   - Any failed documents in ingestion stats?

2. Test retrieval directly (without Agent):
   - Call Retrieve API with test query
   - Check relevance scores — are they > 0.5?

3. Inspect the stored chunks:
   - Query OpenSearch directly
   - Are chunks meaningful or broken fragments?

4. Check chunking:
   - Are chunks too large (mixing unrelated content)?
   - Are chunks too small (no context)?

5. Test embedding quality:
   - Embed your query manually
   - Do a raw knn search in OpenSearch
   - Do the nearest vectors make sense?

6. Check data format:
   - Is the JSON being read as raw text or parsed?
   - Does the text field contain useful content?

7. Fix options:
   - Adjust chunk size/overlap
   - Add metadata filtering
   - Improve document format (more text, less noise)
   - Try different embedding model
```

---

### "A user asks a question and gets a hallucinated answer. Why?"

**Answer:**

Possible causes:

```
1. Retrieval returned irrelevant chunks
   → Agent used wrong context → generated wrong answer
   Fix: Improve retrieval quality (chunk size, overlap, metadata)

2. Retrieval returned NOTHING
   → Agent has no context but still tries to answer
   Fix: System prompt: "If no relevant info found, say 'I don't have this information'"

3. System prompt is too permissive
   → Agent uses its training knowledge instead of KB
   Fix: "ONLY answer using retrieved documents. Never use general knowledge."

4. Top-K too low (returning 1-2 chunks)
   → Not enough context for accurate answer
   Fix: Increase numberOfResults to 5-10

5. Chunks are partial
   → Agent infers incorrectly from incomplete information
   Fix: Increase chunk size or overlap

6. Conflicting information in chunks
   → Multiple chunks say different things
   Fix: Add timestamps, prioritize recent documents
```

---

### "How would you migrate from OpenSearch to a different vector store?"

**Answer:**

```
1. Set up new vector store (e.g., Pinecone, pgvector)
2. Create new KB pointing to new vector store
3. Keep same data source (same S3 bucket)
4. Run ingestion on new KB
5. Test retrieval quality (compare results)
6. Update Agent to point to new KB
7. Monitor for quality regression
8. Decommission old OpenSearch collection

Key: Keep both running in parallel during transition.
The data source (S3) doesn't change — only the storage backend changes.
```

---

### "How would you add authentication so only specific users can query the agent?"

**Answer:**

```
Multiple approaches:

1. IAM-based (AWS native):
   - Create IAM role per user/team
   - Attach policy: allow bedrock:InvokeAgent only for specific agent ARN
   - Users assume role before calling agent

2. API Gateway + Cognito:
   - Put API Gateway in front of agent
   - Cognito user pool for authentication
   - Lambda authorizer for fine-grained access

3. Application-level:
   - Your app authenticates users
   - App calls agent with appropriate session attributes
   - Agent can see user context in prompt

4. Bedrock Guardrails:
   - Block certain topics regardless of user
   - PII filtering for all responses
```

---

## Quick Reference: Key Numbers

```
Embedding dimension:     1024 (Titan V2)
Chunk size:             512 tokens (~380 words)
Chunk overlap:          10%
HNSW ef_construction:   512
HNSW m:                 16
Distance metric:        L2 (Euclidean)
Min OCUs (Serverless):  4 (2 indexing + 2 search)
OCU cost:               $0.24/hour per OCU
Min monthly cost:       ~$700 (OpenSearch Serverless)
Ingestion check:        GetIngestionJob API
Agent session TTL:      600 seconds (10 minutes)
```
