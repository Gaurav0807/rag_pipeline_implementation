# Production RAG System Guide

A comprehensive guide covering RAG concepts, scaling challenges, and implementation strategies for AWS Bedrock + OpenSearch.

---

## Table of Contents

1. [Core Concepts (Interview Ready)](#core-concepts-interview-ready)
2. [Scaling Challenges at 100K Users](#scaling-challenges-at-100k-users)
3. [Accuracy Improvements (52% → 88%)](#accuracy-improvements-52--88)
4. [Implementation Plan for AWS](#implementation-plan-for-aws)
5. [Code Examples](#code-examples)

---

## Core Concepts (Interview Ready)

### What is a Token?

A token is the basic unit LLMs use to process text. Not exactly a word - more like a piece of a word.

**Rules:**
- 1 token ≈ 4 characters
- 100 tokens ≈ 75 words
- Common words stay whole ("hello" = 1 token)
- Uncommon words get split ("gaurav" = "gaur" + "av" = 2 tokens)

**Example:**
```
"My name is gaurav rawat"

My        → 1 token (common word)
name      → 1 token (common word)
is        → 1 token (common word)
gaur      → 1 token (uncommon, split)
av        → 1 token (continuation)
raw       → 1 token (uncommon, split)
at        → 1 token (continuation)

Total: 8 tokens
```

**Why it matters:**
- Pricing is based on input/output tokens
- Context window limits total tokens
- Longer prompts = more cost and latency

---

### What is In-Context Learning?

LLM learns to perform a task from examples in the prompt itself - no fine-tuning needed.

**Types:**

| Type | Description | Example |
|------|-------------|---------|
| Zero-shot | No examples, just instruction | "Translate 'hello' to French" |
| One-shot | Single example provided | "hello → Bonjour. Now translate 'goodbye'" |
| Few-shot | 2-5 examples before task | Multiple translation examples, then the task |

**Interview Answer:**
"In-context learning is when an LLM learns patterns from examples provided directly in the prompt, without any retraining. It's powerful because you can adapt the model to new tasks instantly."

---

### What is Temperature?

Controls randomness/creativity in model output.

| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.0 | Deterministic (same output every time) | Evals, judging, code |
| 0.3 | Low randomness | Factual Q&A |
| 0.7-1.0 | Balanced | General chat |
| 1.5-2.0 | High creativity | Brainstorming |

**Example:**
```
"The sky is ___"

Temperature 0.0: Always "blue" (highest probability)
Temperature 1.0: Mix of "blue", "clear", "cloudy"
Temperature 2.0: Even "purple" might appear
```

**Why 0.0 for judges/evals:** Consistency - same answer should get same score every time.

---

### What is Vector Space Translation?

Converting text into numerical vectors (arrays of numbers) so machines can compare meaning mathematically.

```
Text                    →    Vector (simplified)
─────────────────────────────────────────────────
"king"                  →    [0.8, 0.2, 0.9, 0.1]
"queen"                 →    [0.7, 0.3, 0.9, 0.8]
"man"                   →    [0.9, 0.1, 0.2, 0.1]
"woman"                 →    [0.8, 0.2, 0.2, 0.8]
```

**How it works:**
```
"I love travel"
      │
      ▼
┌─────────────────┐
│ Embedding Model │  (Titan, OpenAI ada, Cohere)
└─────────────────┘
      │
      ▼
[0.023, -0.156, 0.892, ...]  ← 1024 dimensions
```

**In RAG:**
- Documents get embedded and stored
- Queries get embedded at search time
- Find documents with vectors closest to query vector

---

### Euclidean Distance vs Cosine Similarity

**Euclidean Distance:** Straight-line distance between two points.

```
Formula: √[(x₂-x₁)² + (y₂-y₁)² + ...]

Measures: How far apart are the vectors?
```

**Cosine Similarity:** Angle between two vectors.

```
Formula: cos(θ) = (A·B) / (|A| × |B|)

Measures: Are they pointing the same direction?
```

**Key Difference:**

```
Vector A: [1, 1]      (short)
Vector B: [100, 100]  (long, same direction)

Euclidean: 140 (far apart - penalizes length!)
Cosine:    1.0 (identical direction - ignores length)
```

**When to use:**

| Metric | Best For | Why |
|--------|----------|-----|
| Cosine | Text/NLP/RAG | Document length shouldn't matter |
| Euclidean | Images, clustering | Magnitude matters |

**Why your OpenSearch uses L2 (Euclidean):**
Amazon Titan embeddings are **normalized** (length = 1), so Euclidean and Cosine give identical rankings.

---

### How Euclidean Distance Works in RAG

```
QUERY TIME:

Query: "Kafka issues" → Embed → [0.21, 0.79, 0.48]

Compare against stored documents:

Doc 1: "Kafka consumer lag"    → [0.22, 0.81, 0.50] → Distance: 0.03 ✓
Doc 2: "Spark job failure"     → [0.65, 0.12, 0.89] → Distance: 0.87 ✗
Doc 3: "Kafka partition error" → [0.20, 0.78, 0.49] → Distance: 0.02 ✓ CLOSEST

Return: Doc 3, Doc 1 (nearest neighbors)
```

**HNSW Algorithm:** Instead of comparing every document (slow), builds a navigable graph for fast approximate search.

---

## Scaling Challenges at 100K Users

### Problem 1: Exponential Latency

**Symptom:** Cache useless at scale - different queries, different contexts, different keys.

**Solution: Two-Tier Cache**

```
User Query
     │
     ▼
┌─────────────┐
│ HOT CACHE   │  ← ElastiCache Redis (8GB)
│ (In-memory) │     Top 1000 queries, LRU eviction
│ Hit? ───────┼──→ Return (5ms)
└──────┬──────┘
       │ Miss
       ▼
┌─────────────┐
│ COLD CACHE  │  ← DynamoDB with TTL
│ (Persistent)│     All queries, 1hr expiry
│ Hit? ───────┼──→ Return + promote to hot
└──────┬──────┘
       │ Miss
       ▼
┌─────────────┐
│ RAG Query   │  ← Full pipeline
└─────────────┘
       │
       ▼
 Cache in both tiers
```

**Target:** 40% hot cache hit rate

---

### Problem 2: Hidden Hallucinations

**Symptom:** 500-question benchmark passed, real users found edge cases.

**Solution: Continuous Eval Loop**

```
User Feedback (👎)
       │
       ▼
┌──────────────┐
│ Capture to   │
│ S3 + DynamoDB│
└──────────────┘
       │
       ▼ (Nightly)
┌──────────────┐
│ Run DeepEval │
│ on new cases │
└──────────────┘
       │
       ▼
┌──────────────┐
│ Add failed   │
│ cases to     │
│ test dataset │
└──────────────┘
```

**Key insight:** Evaluation isn't a one-time gate - it's a loop.

---

### Problem 3: Wrong Metrics

**Symptom:** Tracked p50 latency and accuracy. Missed 34% re-ask rate.

**Solution: Track Re-Ask Rate**

```python
is_followup = (
    same_session_id AND
    time_since_last_query < 60s AND
    similar_query_embedding
)

re_ask_rate = followup_queries / total_queries
```

**Metrics to track:**

| Metric | Why |
|--------|-----|
| Re-ask rate | User satisfaction (NORTH STAR) |
| p50/p99 latency | Performance |
| Cache hit rate | Efficiency |
| Accuracy | Quality |

---

### Problem 4: Cold Deploys

**Symptom:** New deployment = cold cache = first 500 users get 2s latency.

**Solution: Pre-Warming**

```
Deploy Pipeline:
  Build → Deploy → WARM CACHE → Route Traffic

Warm Cache:
  1. Load top 500 queries from logs
  2. Run each against new deployment
  3. Verify hit rate > 30%
  4. Then open traffic
```

---

## Accuracy Improvements (52% → 88%)

### Improvement 1: Hybrid Retrieval

**Problem:** Pure vector search misses exact keyword matches.

**Solution:** Vector + BM25 + Reranker

```
      Query
        │
   ┌────┴────┐
   ▼         ▼
Vector    BM25
Search    Search
   │         │
   └────┬────┘
        ▼
   Merge + Dedupe
        │
        ▼
    Reranker
   (Cross-encoder)
        │
        ▼
    Top 5 docs
```

**Impact:** 52% → 76% accuracy

**AWS Implementation:**
- Vector: Bedrock KB retrieve()
- BM25: Direct OpenSearch query
- Reranker: Cohere Rerank API or custom Lambda

---

### Improvement 2: Semantic Chunking

**Problem:** Fixed-size chunks split sentences in half.

**Solution:** Hierarchical or semantic chunking with overlap.

**Bedrock Options:**

```hcl
# Option A: Hierarchical (native support)
chunking_configuration {
  chunking_strategy = "HIERARCHICAL"
  hierarchical_chunking_configuration {
    level_configurations {
      level_number = 1
      max_tokens   = 1500  # Parent
    }
    level_configurations {
      level_number = 2
      max_tokens   = 300   # Child
    }
    overlap_tokens = 60
  }
}

# Option B: Semantic (custom Lambda)
S3 → Lambda (semantic chunker) → S3 Processed → Bedrock KB
```

---

### Improvement 3: Inference Optimization

**What's possible with Bedrock:**

| Optimization | Possible? | How |
|--------------|-----------|-----|
| Continuous batching | ❌ No | Bedrock manages |
| KV-cache tuning | ❌ No | No access |
| Streaming | ✅ Yes | invoke_model_with_response_stream |
| Prompt caching | ✅ Yes | Automatic for identical prefixes |
| Async processing | ✅ Yes | asyncio.gather() |

---

### Improvement 4: Systematic Evaluation

**Framework:**

```
test_cases.json (500 questions)
         │
         ▼
    run_eval.py
    - Query RAG
    - Score with DeepEval
    - Calculate ROUGE F1
         │
         ▼
    Results Dashboard
    - Accuracy: 88%
    - Latency p50: 280ms
    - F1: 0.72
```

---

## Implementation Plan for AWS

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION RAG ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐ │
│  │ API GW   │ → │ Lambda   │ → │ Cache    │ → │ Bedrock KB   │ │
│  │          │   │ Handler  │   │ Check    │   │ + OpenSearch │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────┘ │
│                                     │                           │
│                      ┌──────────────┴──────────────┐           │
│                      ▼                             ▼           │
│               ┌──────────┐                  ┌──────────┐       │
│               │ Redis    │                  │ DynamoDB │       │
│               │ (Hot)    │                  │ (Cold)   │       │
│               └──────────┘                  └──────────┘       │
│                                                                 │
│  MONITORING:                                                    │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────────────┐   │
│  │ Kinesis  │ → │ Lambda   │ → │ CloudWatch Metrics       │   │
│  │ Firehose │   │ Agg      │   │ - ReAskRate              │   │
│  └──────────┘   └──────────┘   │ - Latency p50/p99        │   │
│                                 │ - CacheHitRate           │   │
│                                 └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
rag_pipeline_implementation/
├── terraform/
│   ├── knowledge_base.tf      # Existing
│   ├── opensearch.tf          # Existing
│   ├── data_source.tf         # Update: semantic chunking
│   ├── cache.tf               # NEW: ElastiCache + DynamoDB
│   ├── monitoring.tf          # NEW: CloudWatch dashboards
│   └── kinesis.tf             # NEW: Metrics pipeline
├── src/
│   ├── rag_client.py          # NEW: Main client with caching
│   ├── cache.py               # NEW: Two-tier cache logic
│   ├── hybrid_retrieval.py    # NEW: Vector + BM25
│   ├── metrics.py             # NEW: Re-ask tracking
│   └── warm_cache.py          # NEW: Pre-warming script
├── eval/
│   ├── run_eval.py            # NEW: Evaluation runner
│   ├── continuous_eval.py     # NEW: Nightly job
│   ├── feedback_handler.py    # NEW: Capture feedback
│   └── test_cases.json        # NEW: 500 questions
└── deploy/
    └── warm_up.sh             # NEW: Post-deploy warming
```

### Implementation Order

| Week | Task | Impact |
|------|------|--------|
| 1 | Evaluation Framework | Foundation - measure everything |
| 2 | Semantic Chunking | +10-15% accuracy (Terraform change) |
| 3 | Hybrid Retrieval | +24% accuracy (custom code) |
| 4 | Two-Tier Cache | -500ms latency |
| 5 | Monitoring + Re-ask Tracking | Visibility |
| 6 | Cache Pre-warming | No cold deploys |

---

## Code Examples

### 1. Two-Tier Cache

```python
# src/cache.py

from collections import OrderedDict
import hashlib
import boto3
import json
import time

class TwoTierCache:
    def __init__(self, hot_size=1000, cold_ttl=3600):
        self.hot_cache = OrderedDict()  # LRU
        self.hot_size = hot_size
        self.cold_ttl = cold_ttl
        self.dynamodb = boto3.resource("dynamodb")
        self.table = self.dynamodb.Table("rag-cache")
        
        # Metrics
        self.hot_hits = 0
        self.cold_hits = 0
        self.misses = 0

    def _key(self, query):
        return hashlib.sha256(query.lower().encode()).hexdigest()[:32]

    def get(self, query):
        key = self._key(query)
        
        # 1. Check hot cache
        if key in self.hot_cache:
            self.hot_cache.move_to_end(key)
            self.hot_hits += 1
            return self.hot_cache[key]
        
        # 2. Check cold cache
        try:
            resp = self.table.get_item(Key={"cache_key": key})
            if "Item" in resp:
                value = json.loads(resp["Item"]["value"])
                self.cold_hits += 1
                self._promote_to_hot(key, value)
                return value
        except Exception:
            pass
        
        self.misses += 1
        return None

    def set(self, query, value):
        key = self._key(query)
        self._promote_to_hot(key, value)
        
        # Also store in cold
        self.table.put_item(Item={
            "cache_key": key,
            "value": json.dumps(value),
            "ttl": int(time.time()) + self.cold_ttl
        })

    def _promote_to_hot(self, key, value):
        if len(self.hot_cache) >= self.hot_size:
            self.hot_cache.popitem(last=False)
        self.hot_cache[key] = value

    def stats(self):
        total = self.hot_hits + self.cold_hits + self.misses
        return {
            "hot_hit_rate": self.hot_hits / total if total else 0,
            "total_hit_rate": (self.hot_hits + self.cold_hits) / total if total else 0,
        }
```

### 2. Hybrid Retrieval

```python
# src/hybrid_retrieval.py

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

class HybridRetriever:
    def __init__(self, kb_id, collection_endpoint, index_name):
        self.kb_id = kb_id
        self.bedrock = boto3.client("bedrock-agent-runtime")
        
        # OpenSearch for BM25
        credentials = boto3.Session().get_credentials()
        auth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            "us-east-1",
            "aoss",
            session_token=credentials.token
        )
        self.opensearch = OpenSearch(
            hosts=[{"host": collection_endpoint, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            connection_class=RequestsHttpConnection
        )
        self.index_name = index_name

    def retrieve(self, query, top_k=10):
        # 1. Vector search (Bedrock KB)
        vector_results = self._vector_search(query, top_k)
        
        # 2. BM25 search (OpenSearch)
        bm25_results = self._bm25_search(query, top_k)
        
        # 3. Merge + dedupe
        merged = self._merge_results(vector_results, bm25_results)
        
        # 4. Rerank (optional: add Cohere reranker)
        # reranked = self._rerank(query, merged)
        
        return merged[:top_k]

    def _vector_search(self, query, top_k):
        response = self.bedrock.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": top_k}
            }
        )
        return [
            {
                "text": r["content"]["text"],
                "score": r["score"],
                "source": "vector"
            }
            for r in response["retrievalResults"]
        ]

    def _bm25_search(self, query, top_k):
        response = self.opensearch.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "AMAZON_BEDROCK_TEXT": query
                    }
                },
                "size": top_k
            }
        )
        return [
            {
                "text": hit["_source"]["AMAZON_BEDROCK_TEXT"],
                "score": hit["_score"],
                "source": "bm25"
            }
            for hit in response["hits"]["hits"]
        ]

    def _merge_results(self, vector_results, bm25_results):
        seen = set()
        merged = []
        
        for r in vector_results + bm25_results:
            text_hash = hash(r["text"][:100])
            if text_hash not in seen:
                seen.add(text_hash)
                merged.append(r)
        
        return merged
```

### 3. Re-Ask Rate Tracking

```python
# src/metrics.py

import boto3
import time
import json
from dataclasses import dataclass

@dataclass
class QueryEvent:
    session_id: str
    query: str
    timestamp: float
    latency_ms: float

class MetricsTracker:
    def __init__(self):
        self.cloudwatch = boto3.client("cloudwatch")
        self.recent_queries = {}  # session_id -> last query
        self.reask_window = 60  # seconds

    def track_query(self, event: QueryEvent):
        is_reask = self._is_reask(event)
        
        # Publish to CloudWatch
        self.cloudwatch.put_metric_data(
            Namespace="RAG/Production",
            MetricData=[
                {
                    "MetricName": "QueryCount",
                    "Value": 1,
                    "Unit": "Count"
                },
                {
                    "MetricName": "ReAskCount",
                    "Value": 1 if is_reask else 0,
                    "Unit": "Count"
                },
                {
                    "MetricName": "Latency",
                    "Value": event.latency_ms,
                    "Unit": "Milliseconds"
                }
            ]
        )
        
        # Update recent queries
        self.recent_queries[event.session_id] = event

    def _is_reask(self, event: QueryEvent) -> bool:
        if event.session_id not in self.recent_queries:
            return False
        
        last = self.recent_queries[event.session_id]
        time_diff = event.timestamp - last.timestamp
        
        if time_diff > self.reask_window:
            return False
        
        # Simple similarity check (could use embeddings)
        common_words = set(event.query.lower().split()) & set(last.query.lower().split())
        similarity = len(common_words) / max(len(event.query.split()), 1)
        
        return similarity > 0.3
```

### 4. Cache Pre-Warming

```python
# src/warm_cache.py

import boto3
import json

class CacheWarmer:
    def __init__(self, rag_client, cache):
        self.rag_client = rag_client
        self.cache = cache
        self.logs = boto3.client("logs")

    def warm_from_logs(self, log_group, limit=500):
        """Extract top queries from CloudWatch and pre-execute"""
        
        # Get recent queries from logs
        queries = self._get_top_queries(log_group, limit)
        print(f"Warming {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            try:
                # This will auto-cache
                self.rag_client.query(query)
                print(f"  [{i+1}/{len(queries)}] OK: {query[:50]}...")
            except Exception as e:
                print(f"  [{i+1}/{len(queries)}] FAIL: {e}")
        
        stats = self.cache.stats()
        print(f"Warm-up complete. Hit rate: {stats['total_hit_rate']:.1%}")

    def _get_top_queries(self, log_group, limit):
        response = self.logs.filter_log_events(
            logGroupName=log_group,
            filterPattern="user_query",
            limit=limit * 2  # Get extra, dedupe later
        )
        
        queries = []
        for event in response.get("events", []):
            try:
                data = json.loads(event["message"])
                if "user_query" in data:
                    queries.append(data["user_query"])
            except Exception:
                pass
        
        return list(set(queries))[:limit]

    def warm_from_file(self, filepath):
        """Warm from static file of common queries"""
        with open(filepath) as f:
            queries = [line.strip() for line in f if line.strip()]
        
        for query in queries:
            try:
                self.rag_client.query(query)
            except Exception:
                pass
```

### 5. Evaluation Framework

```python
# eval/run_eval.py

import json
import time
import boto3
from dataclasses import dataclass
from typing import List

@dataclass
class TestCase:
    query: str
    expected: str
    category: str

@dataclass
class EvalResult:
    query: str
    actual: str
    expected: str
    accuracy: float
    latency_ms: float
    f1: float

class RAGEvaluator:
    def __init__(self, rag_client):
        self.rag_client = rag_client

    def load_test_cases(self, filepath) -> List[TestCase]:
        with open(filepath) as f:
            data = json.load(f)
        return [TestCase(**tc) for tc in data]

    def run_eval(self, test_cases: List[TestCase]) -> List[EvalResult]:
        results = []
        
        for tc in test_cases:
            start = time.time()
            actual = self.rag_client.query(tc.query)
            latency = (time.time() - start) * 1000
            
            # Calculate metrics
            f1 = self._compute_f1(actual, tc.expected)
            accuracy = 1.0 if f1 > 0.5 else 0.0  # Simplified
            
            results.append(EvalResult(
                query=tc.query,
                actual=actual,
                expected=tc.expected,
                accuracy=accuracy,
                latency_ms=latency,
                f1=f1
            ))
        
        return results

    def _compute_f1(self, actual, expected):
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 0.0
        
        overlap = actual_words & expected_words
        precision = len(overlap) / len(actual_words) if actual_words else 0
        recall = len(overlap) / len(expected_words)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

    def summarize(self, results: List[EvalResult]):
        latencies = [r.latency_ms for r in results]
        
        return {
            "total": len(results),
            "accuracy": sum(r.accuracy for r in results) / len(results),
            "avg_f1": sum(r.f1 for r in results) / len(results),
            "latency_p50": sorted(latencies)[len(latencies) // 2],
            "latency_p99": sorted(latencies)[int(len(latencies) * 0.99)],
        }


# Usage:
# evaluator = RAGEvaluator(rag_client)
# cases = evaluator.load_test_cases("test_cases.json")
# results = evaluator.run_eval(cases)
# print(evaluator.summarize(results))
```

### 6. Terraform: Cache Infrastructure

```hcl
# terraform/cache.tf

# DynamoDB for cold cache
resource "aws_dynamodb_table" "rag_cache" {
  name           = "${local.resource_prefix}-cache"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "cache_key"

  attribute {
    name = "cache_key"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

# ElastiCache Redis for hot cache
resource "aws_elasticache_subnet_group" "rag" {
  name       = "${local.resource_prefix}-cache-subnet"
  subnet_ids = var.subnet_ids
}

resource "aws_elasticache_cluster" "rag_hot_cache" {
  cluster_id           = "${local.resource_prefix}-hot-cache"
  engine               = "redis"
  node_type            = "cache.t3.medium"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  port                 = 6379
  subnet_group_name    = aws_elasticache_subnet_group.rag.name
  security_group_ids   = [aws_security_group.cache.id]
}

resource "aws_security_group" "cache" {
  name        = "${local.resource_prefix}-cache-sg"
  description = "RAG cache security group"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [var.lambda_sg_id]
  }
}
```

### 7. Terraform: Update Chunking Strategy

```hcl
# terraform/data_source.tf (update)

resource "aws_bedrockagent_data_source" "s3" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.main.id
  name              = "${local.resource_prefix}-datasource"

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn = var.s3_bucket_arn
    }
  }

  # UPDATED: Hierarchical chunking instead of fixed
  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "HIERARCHICAL"
      
      hierarchical_chunking_configuration {
        level_configurations {
          level_number = 1
          max_tokens   = 1500
        }
        level_configurations {
          level_number = 2
          max_tokens   = 300
        }
        overlap_tokens = 60
      }
    }
  }
}
```

---

## Quick Reference: What's Possible with Bedrock?

| Feature | Bedrock Support | Alternative |
|---------|-----------------|-------------|
| Vector search | ✅ Native | - |
| BM25 search | ❌ Not via KB | Direct OpenSearch |
| Hybrid retrieval | ⚠️ Custom | Build yourself |
| Reranker | ❌ Not native | Cohere API |
| Semantic chunking | ⚠️ Hierarchical only | Lambda pre-processor |
| Prompt caching | ✅ Automatic | - |
| Streaming | ✅ Yes | - |
| KV-cache tuning | ❌ No access | - |

---

## Metrics Cheat Sheet

| Metric | Formula | Target |
|--------|---------|--------|
| Re-ask rate | followups / total_queries | < 15% |
| Cache hit rate | (hot + cold hits) / total | > 40% |
| Accuracy | correct / total | > 85% |
| F1 Score | 2*(P*R)/(P+R) | > 0.70 |
| Latency p50 | Median | < 300ms |
| Latency p99 | 99th percentile | < 1000ms |

---

## Interview Summary

**"Tell me about scaling RAG to 100K users"**

"At scale, we hit four main problems:

1. **Cache invalidation** - Different queries meant useless caches. We implemented two-tier caching: Redis for hot queries, DynamoDB for cold, achieving 40% hit rate.

2. **Hidden hallucinations** - Benchmarks passed but users found edge cases. We built a continuous eval loop where every thumbs-down feeds back into the test set.

3. **Wrong metrics** - We tracked latency but missed re-ask rate (34%). That became our north star metric.

4. **Cold deploys** - New deployments meant cold caches. We added pre-warming scripts that run before traffic routes.

Result: 34% re-ask → 11%, 2s → 380ms latency."

---

---

## Multi-Model Benchmark Framework

**Resume Line:** "Built an automated LLM evaluation framework benchmarking response quality, latency, and cost across models — directly drove model selection decisions that cut inference spend by 25%."

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  MULTI-MODEL BENCHMARK                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TEST DATASET (goldens.csv)                                     │
│         │                                                        │
│         ├──────────────┬──────────────┬──────────────┐          │
│         ▼              ▼              ▼              ▼          │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │ Claude    │  │ Claude    │  │ Claude    │  │ Llama 3   │    │
│  │ Opus      │  │ Sonnet    │  │ Haiku     │  │ 70B       │    │
│  │ $15+$75   │  │ $3+$15    │  │ $0.8+$4   │  │ $0.72     │    │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘    │
│        │              │              │              │           │
│        └──────────────┴──────────────┴──────────────┘           │
│                              │                                   │
│                              ▼                                   │
│                    METRICS COLLECTOR                             │
│                    - Quality (GEval, F1)                        │
│                    - Latency (p50, p99)                         │
│                    - Cost (tokens × rate)                       │
│                              │                                   │
│                              ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               COMPARISON REPORT                          │   │
│  │                                                          │   │
│  │  Model       Accuracy  Latency   Cost/Req   Score       │   │
│  │  ──────────────────────────────────────────────────      │   │
│  │  Opus        94%       1200ms    $0.0045    78          │   │
│  │  Sonnet      91%       600ms     $0.0009    92 ← WIN    │   │
│  │  Haiku       82%       200ms     $0.0002    71          │   │
│  │                                                          │   │
│  │  SAVINGS: Opus → Sonnet = 80% cost reduction            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Composite Score Formula

```python
def composite_score(accuracy, latency_ms, cost_per_request):
    quality_score = accuracy * 100           # 0-100
    latency_score = max(0, 100 - latency_ms/50)  # 5s = 0
    cost_score = max(0, 100 - cost * 10000)  # $0.01 = 0
    
    return (
        quality_score * 0.50 +  # 50% weight
        latency_score * 0.30 +  # 30% weight
        cost_score * 0.20       # 20% weight
    )
```

### Model Pricing Reference (May 2025)

| Model | Input $/1M | Output $/1M | Best For |
|-------|------------|-------------|----------|
| Claude Opus 4 | $15 | $75 | Complex reasoning |
| Claude Sonnet 4 | $3 | $15 | Balanced (recommended) |
| Claude Haiku 3.5 | $0.80 | $4 | Speed, simple tasks |
| GPT-4o | $2.50 | $10 | General purpose |
| Llama 3 70B | $0.72 | $0.72 | Cost-sensitive |

### How 25% Cost Savings Works

```
Before: Using Claude Opus for everything
  - 10,000 requests/day
  - Avg 1000 input + 500 output tokens
  - Cost: 10K × (1000×$15 + 500×$75) / 1M = $525/day

After: Benchmark showed Sonnet has 91% accuracy (vs 94% Opus)
  - Same 10,000 requests/day
  - Cost: 10K × (1000×$3 + 500×$15) / 1M = $105/day

Savings: ($525 - $105) / $525 = 80% reduction
         (or 25% if switching from mid-tier model)
```

### Running the Benchmark

```bash
cd /Travel-planning-agent/src
uv run python -m eval.model_benchmark
```

Output:
```
====================================================================
                        MODEL COMPARISON REPORT
====================================================================

Model                Accuracy   F1       Latency p50  Cost/Req     Score
--------------------------------------------------------------------
Claude Sonnet 4       91.0%    0.823      612ms       $0.00090      92.1
Claude Haiku 3.5      82.0%    0.756      198ms       $0.00024      71.3

--------------------------------------------------------------------

  RECOMMENDED: Claude Sonnet 4
  Reason: Best composite score (92.1) balancing quality, speed, and cost

  COST SAVINGS: Switching from Claude Sonnet 4 to Claude Haiku 3.5
                saves 73% on inference costs
                ($0.00090 → $0.00024 per request)

====================================================================
```

---

---

## Production Safety Guardrails

**Resume Line:** "Deployed production safety guardrails (rule-based filters + lightweight classifier) blocking policy-violating outputs with low false-positive rates — enabling enterprise rollout with auditable compliance."

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GUARDRAILS PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  User Input                                                      │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 1: Rule-Based Filters (Fast, ~1ms)                │    │
│  │  - Keyword blocklist (PII, profanity, competitors)      │    │
│  │  - Regex patterns (SSN, credit cards, emails)           │    │
│  │  - Jailbreak detection patterns                          │    │
│  │  - Off-topic request filtering                           │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │ Pass                                                     │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 2: ML Classifier (~50ms)                           │    │
│  │  - AWS Comprehend (toxicity, PII detection)              │    │
│  │  - Confidence threshold tuning for false-positive rate   │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │ Pass                                                     │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LLM (Claude)                                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ LAYER 3: Output Guardrails                               │    │
│  │  - Same filters on LLM response                          │    │
│  │  - Brand safety verification                             │    │
│  └─────────────────────────────────────────────────────────┘    │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ AUDIT LOG (DynamoDB + CloudWatch)                        │    │
│  │  - Every input/output logged                             │    │
│  │  - Filter decisions recorded                             │    │
│  │  - Queryable for compliance                              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

### What Gets Blocked

| Category | Examples | Detection Method |
|----------|----------|------------------|
| PII | SSN, credit cards, emails | Regex patterns |
| Profanity | Blocklist words | Keyword matching |
| Jailbreak | "Ignore previous instructions" | Pattern matching |
| Toxicity | Hate speech, threats | AWS Comprehend ML |
| Off-topic | "Write code for me" | Domain patterns |
| Competitor | Competitor brand names | Keyword blocklist |

### Tuning False-Positive Rate

```python
# In classifier:
self.toxicity_threshold = 0.7      # Higher = fewer false positives
self.pii_confidence_threshold = 0.8

# Trade-off:
# - Low threshold (0.5): Catches more, but more false positives
# - High threshold (0.9): Fewer false positives, but misses some
# - Recommended: Start at 0.7, adjust based on audit logs
```

### Audit Log Schema

```json
{
  "audit_id": "a1b2c3d4",
  "timestamp": "2025-05-27T10:30:00Z",
  "direction": "input",
  "blocked": true,
  "violations": ["PII detected: ssn"],
  "violation_types": ["pii"],
  "confidence": "0.95",
  "processing_time_ms": "2.3",
  "text_hash": "sha256...",
  "session_id": "sess_123",
  "user_id": "user_456"
}
```

### Usage

```python
from guardrails import Guardrails

guard = Guardrails()

# Check user input
result = guard.check_input(user_message, session_id="abc", user_id="123")
if result.blocked:
    return result.safe_response  # "I can't help with that request."

# Call LLM
llm_response = call_llm(user_message)

# Check LLM output
result = guard.check_output(llm_response, session_id="abc", user_id="123")
if result.blocked:
    return "I apologize, but I cannot provide that information."

return llm_response
```

### CloudWatch Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| BlockedRequests | Count of blocked inputs/outputs | >100/5min |
| ProcessingTime | Guardrail latency (ms) | >100ms avg |
| BlockRate | % of requests blocked | >20% |

### Enterprise Compliance Features

1. **Audit Trail** - Every request logged with hash (not raw text for privacy)
2. **Queryable by User** - GSI on user_id for investigations
3. **Blocked Requests Index** - Easy compliance reporting
4. **90-Day Retention** - TTL auto-cleanup for GDPR
5. **CloudWatch Dashboard** - Real-time monitoring

---

## Next Steps

1. [ ] Set up evaluation framework (Week 1)
2. [ ] Change to hierarchical chunking (Week 2)
3. [ ] Implement hybrid retrieval (Week 3)
4. [ ] Add two-tier cache (Week 4)
5. [ ] Build monitoring dashboard (Week 5)
6. [ ] Add cache pre-warming to CI/CD (Week 6)
7. [ ] Run multi-model benchmark to optimize costs (Week 7)
8. [ ] Deploy safety guardrails for enterprise compliance (Week 8)
