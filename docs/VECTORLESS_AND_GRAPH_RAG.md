# Vector-Less RAG & Graph-Based RAG Implementation Guide

This document shows how to extend your existing RAG pipeline to support **vector-less retrieval** and **graph-based RAG** approaches alongside your current vector-based system.

---

## Part 1: Vector-Less RAG Implementation

### Overview

Instead of relying on embeddings, use **BM25 full-text search** in OpenSearch to retrieve relevant documents based on keyword matching and TF-IDF scoring.

### Architecture Comparison

```
Current (Vector-Based):
Document → Chunk → Embedding → Vector Search → Top-K Vectors → LLM

Vector-Less (Keyword-Based):
Document → Parse → Index Terms → BM25 Search → Top-K Matches → LLM
```

### Services Needed for Vector-Less RAG

| Layer | Service | Purpose |
|-------|---------|---------|
| **Document Storage** | S3 (existing) | Store source documents |
| **Parsing** | Lambda | Extract text from PDFs, HTML, Markdown |
| **Indexing** | OpenSearch (existing) | BM25 index for full-text search |
| **LLM** | Bedrock Claude (existing) | Generate responses |
| **Orchestration** | Lambda (existing) | Coordinate retrieval + generation |

### Step 1: Create a BM25 Index in OpenSearch

Add a new index alongside your vector index:

```python
# lambdas/opensearch_index_creator/create_bm25_index.py
import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

def create_bm25_index(opensearch_endpoint):
    """Create BM25 full-text search index in OpenSearch."""
    
    region = 'us-east-1'
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        'aoss',
        session_token=credentials.token
    )
    
    client = OpenSearch(
        hosts=[{'host': opensearch_endpoint, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60
    )
    
    # BM25 index mapping
    index_body = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard",
                            "stopwords": "_english_"
                        }
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "document_id": {"type": "keyword"},
                "source_uri": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "standard",
                    "fields": {"keyword": {"type": "keyword"}}
                },
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "chunk_index": {"type": "integer"},
                "metadata": {
                    "type": "object",
                    "enabled": True
                },
                "created_at": {"type": "date"}
            }
        }
    }
    
    # Create index
    index_name = "bm25-documents"
    try:
        client.indices.create(index=index_name, body=index_body)
        print(f"✓ Created BM25 index: {index_name}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"ℹ Index {index_name} already exists")
        else:
            raise
    
    return client, index_name


def bulk_index_documents(client, index_name, documents):
    """Index documents using BM25."""
    
    operations = []
    for doc in documents:
        operations.append({
            "_index": index_name,
            "_id": doc["document_id"],
            "_source": {
                "document_id": doc["document_id"],
                "source_uri": doc["source_uri"],
                "title": doc.get("title", ""),
                "content": doc["content"],
                "chunk_index": doc.get("chunk_index", 0),
                "metadata": doc.get("metadata", {}),
                "created_at": doc.get("created_at", "")
            }
        })
    
    # Bulk insert
    success, failed = 0, 0
    try:
        from opensearchpy import helpers
        for success_count, error in helpers.bulk(
            client,
            operations,
            chunk_size=500,
            raise_on_error=False
        ):
            if isinstance(error, dict):
                failed += 1
            else:
                success += 1
    except Exception as e:
        print(f"Bulk indexing error: {e}")
    
    return success, failed
```

### Step 2: Create Vector-Less Retrieval Lambda

```python
# lambdas/vectorless_retrieval/retrieve_bm25.py
import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

def retrieve_bm25(query, top_k=5, opensearch_endpoint=None):
    """Retrieve documents using BM25 search (no vectors)."""
    
    if not opensearch_endpoint:
        opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
    
    region = 'us-east-1'
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        'aoss',
        session_token=credentials.token
    )
    
    client = OpenSearch(
        hosts=[{'host': opensearch_endpoint, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    
    # BM25 query
    search_body = {
        "size": top_k,
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title^2", "content"],  # Title matches weighted 2x
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO"  # Handle typos
            }
        },
        "_source": ["document_id", "source_uri", "title", "content", "metadata"]
    }
    
    response = client.search(
        index="bm25-documents",
        body=search_body
    )
    
    # Format results
    results = []
    for hit in response['hits']['hits']:
        results.append({
            "score": hit['_score'],
            "document_id": hit['_source']['document_id'],
            "source_uri": hit['_source']['source_uri'],
            "title": hit['_source']['title'],
            "content": hit['_source']['content'],
            "metadata": hit['_source'].get('metadata', {})
        })
    
    return results


def lambda_handler(event, context):
    """Lambda entry point for BM25 retrieval."""
    
    query = event.get('query')
    top_k = event.get('top_k', 5)
    
    if not query:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'query parameter required'})
        }
    
    try:
        results = retrieve_bm25(query, top_k)
        
        # Format for Bedrock agent
        context_text = "\n\n".join([
            f"[{i+1}] {r['title']}\n{r['content']}\nSource: {r['source_uri']}"
            for i, r in enumerate(results)
        ])
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'context': context_text,
                'sources': [r['source_uri'] for r in results],
                'hit_count': len(results)
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Step 3: Create Hybrid Retrieval (Vector + Keyword)

For best results, combine both approaches:

```python
# lambdas/hybrid_retrieval/hybrid_search.py
import json
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

def hybrid_search(query, top_k=5, vector_weight=0.6, bm25_weight=0.4):
    """
    Retrieve using BOTH vector search and BM25.
    Results are scored as: (vector_score * weight1) + (bm25_score * weight2)
    """
    
    opensearch_endpoint = os.environ.get('OPENSEARCH_ENDPOINT')
    region = 'us-east-1'
    credentials = boto3.Session().get_credentials()
    auth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'aoss', session_token=credentials.token)
    
    client = OpenSearch(
        hosts=[{'host': opensearch_endpoint, 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )
    
    # Get embedding for vector search
    bedrock = boto3.client('bedrock-runtime')
    embedding_response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v2:0',
        body=json.dumps({'inputText': query})
    )
    embedding = json.loads(embedding_response['body'].read())['embedding']
    
    # Hybrid query: combine vector + BM25
    hybrid_query = {
        "size": top_k * 2,  # Get more candidates
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "AMAZON_BEDROCK_VECTOR": {
                                "vector": embedding,
                                "k": top_k
                            }
                        }
                    },
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["AMAZON_BEDROCK_TEXT^2"],
                            "type": "best_fields",
                            "boost": bm25_weight
                        }
                    }
                ]
            }
        },
        "_source": ["AMAZON_BEDROCK_VECTOR", "AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]
    }
    
    response = client.search(index='rag-kb-index', body=hybrid_query)
    
    # Deduplicate and re-rank
    seen = set()
    results = []
    for hit in response['hits']['hits']:
        source = hit['_source']
        text_id = source['AMAZON_BEDROCK_TEXT'][:50]  # Use first 50 chars as ID
        
        if text_id not in seen:
            seen.add(text_id)
            results.append({
                'score': hit['_score'],
                'text': source['AMAZON_BEDROCK_TEXT'],
                'metadata': source.get('AMAZON_BEDROCK_METADATA', {})
            })
            if len(results) >= top_k:
                break
    
    return results
```

### Step 4: Update Bedrock Agent to Use Vector-Less Retrieval

Modify your Bedrock agent prompt to use the BM25 Lambda instead:

```hcl
# terraform/modules/agent/agent.tf
resource "aws_bedrock_agent" "vectorless_agent" {
  agent_name             = "${local.resource_prefix}-vectorless-agent"
  agent_role_arn         = aws_iam_role.agent_role.arn
  foundation_model_arn   = "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.agent_model}"
  description            = "AI Agent with keyword-based (BM25) retrieval instead of vectors"
  
  instruction = file("${path.module}/prompts/vectorless_agent_prompt.txt")
  
  # Tool: Invoke BM25 retrieval Lambda
  tool_use {
    tool_use_name = "keyword_search"
    description   = "Search documents using keyword matching (BM25)"
    
    input_schema {
      json_schema = {
        type = "object"
        properties = {
          query = {
            type        = "string"
            description = "Search query in natural language"
          }
          top_k = {
            type        = "integer"
            description = "Number of results to return (default: 5)"
          }
        }
        required = ["query"]
      }
    }
  }
  
  idle_session_timeout_in_seconds = 900
}
```

---

## Part 2: Graph-Based RAG Implementation

### Overview

Graph-based RAG constructs a **knowledge graph** from documents where:
- **Nodes** = entities, concepts, topics
- **Edges** = relationships between entities

This enables more sophisticated reasoning and context retrieval.

### Architecture

```
Document → Extract Entities → Build Knowledge Graph → Graph Query → LLM
                   ↓                     ↓
               (People, Places,    (relationships,
                Concepts)           hierarchies)
```

### Services Needed for Graph RAG

| Layer | Service | Purpose |
|-------|---------|---------|
| **Document Storage** | S3 (existing) | Store source documents |
| **Parsing** | Lambda | Extract text and structure |
| **Entity Extraction** | Bedrock Claude | Use LLM to extract entities |
| **Graph Database** | Amazon Neptune OR Neo4j | Store knowledge graph |
| **Graph Query** | Graph Query Language (Gremlin/Cypher) | Query relationships |
| **LLM** | Bedrock Claude (existing) | Generate responses |

### Step 1: Set Up Neptune Graph Database

```hcl
# terraform/neptune.tf
resource "aws_neptune_cluster" "knowledge_graph" {
  cluster_identifier          = "${local.resource_prefix}-kg"
  engine                      = "neptune"
  neptune_cluster_parameter_group_name = aws_neptune_cluster_parameter_group.default.name
  backup_retention_period     = 7
  preferred_backup_window     = "03:00-04:00"
  skip_final_snapshot         = var.environment == "sandbox" ? true : false
  iam_database_authentication_enabled = true
  
  tags = local.common_tags
}

resource "aws_neptune_cluster_instance" "knowledge_graph" {
  cluster_identifier = aws_neptune_cluster.knowledge_graph.id
  instance_class     = var.environment == "prod" ? "db.r5.large" : "db.t4g.medium"
  engine              = "neptune"
  
  tags = local.common_tags
}

resource "aws_neptune_cluster_parameter_group" "default" {
  family      = "neptune1.2"
  description = "Neptune parameter group"

  # Enable query logging
  parameter {
    name  = "neptune_lab_mode"
    value = "true"
  }

  tags = local.common_tags
}

# Neptune access policy
resource "aws_neptune_resource_based_policy" "allow_bedrock" {
  resource_arn = aws_neptune_cluster.knowledge_graph.arn

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = aws_iam_role.lambda_role.arn
        }
        Action   = "neptune-db:*"
        Resource = "${aws_neptune_cluster.knowledge_graph.arn}/*"
      }
    ]
  })
}

output "neptune_endpoint" {
  value = aws_neptune_cluster.knowledge_graph.endpoint
}
```

### Step 2: Create Entity Extraction Lambda

```python
# lambdas/entity_extraction/extract_entities.py
import json
import boto3
import re

bedrock = boto3.client('bedrock-runtime')

ENTITY_EXTRACTION_PROMPT = """
Extract entities and relationships from the following document.

Return a JSON object with:
{
  "entities": [
    {"id": "entity_1", "name": "Entity Name", "type": "PERSON|ORGANIZATION|CONCEPT|PLACE|PRODUCT", "description": "brief description"}
  ],
  "relationships": [
    {"source": "entity_1", "target": "entity_2", "relationship_type": "MANAGES|DEVELOPS|LOCATED_IN|USES|MENTIONS", "description": "brief description"}
  ]
}

Document:
{document_text}

Extract entities and relationships:
"""

def extract_entities_from_document(document_text, document_id):
    """Use Bedrock Claude to extract entities from a document."""
    
    prompt = ENTITY_EXTRACTION_PROMPT.format(document_text=document_text)
    
    response = bedrock.invoke_model(
        modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-06-01",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
    )
    
    response_text = json.loads(response['body'].read())['content'][0]['text']
    
    # Extract JSON from response
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            extracted = json.loads(json_match.group())
        else:
            extracted = json.loads(response_text)
    except json.JSONDecodeError:
        print(f"Failed to parse entity extraction response: {response_text}")
        extracted = {"entities": [], "relationships": []}
    
    # Add document source to metadata
    for entity in extracted.get('entities', []):
        entity['source_document'] = document_id
    
    for relationship in extracted.get('relationships', []):
        relationship['source_document'] = document_id
    
    return extracted


def lambda_handler(event, context):
    """Lambda handler for entity extraction."""
    
    document_id = event.get('document_id')
    document_text = event.get('document_text')
    
    if not document_text:
        return {'statusCode': 400, 'error': 'document_text required'}
    
    try:
        extracted = extract_entities_from_document(document_text, document_id)
        return {
            'statusCode': 200,
            'body': json.dumps(extracted)
        }
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'error': str(e)
        }
```

### Step 3: Create Neptune Graph Builder Lambda

```python
# lambdas/graph_builder/build_knowledge_graph.py
import json
import boto3
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

def build_knowledge_graph(entities, relationships, neptune_endpoint):
    """Build knowledge graph in Neptune from extracted entities."""
    
    # Connect to Neptune
    connection = DriverRemoteConnection(
        f'wss://{neptune_endpoint}:8182/gremlin',
        'g'
    )
    g = traversal().withRemote(connection)
    
    try:
        # Add entities as vertices
        for entity in entities:
            g.addV(entity['type']) \
                .property('id', entity['id']) \
                .property('name', entity['name']) \
                .property('description', entity.get('description', '')) \
                .property('source_document', entity.get('source_document', '')) \
                .next()
        
        # Add relationships as edges
        for rel in relationships:
            source_entity = rel['source']
            target_entity = rel['target']
            rel_type = rel['relationship_type']
            
            g.V() \
                .has('id', source_entity) \
                .addE(rel_type) \
                .to(g.V().has('id', target_entity)) \
                .property('description', rel.get('description', '')) \
                .property('source_document', rel.get('source_document', '')) \
                .next()
        
        connection.close()
        return {
            'entities_added': len(entities),
            'relationships_added': len(relationships)
        }
    
    except Exception as e:
        connection.close()
        raise e


def lambda_handler(event, context):
    """Lambda handler for graph building."""
    
    entities = event.get('entities', [])
    relationships = event.get('relationships', [])
    
    if not entities:
        return {'statusCode': 400, 'error': 'entities required'}
    
    neptune_endpoint = os.environ.get('NEPTUNE_ENDPOINT')
    
    try:
        result = build_knowledge_graph(entities, relationships, neptune_endpoint)
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'error': str(e)
        }
```

### Step 4: Create Graph Query Lambda (Graph RAG Retrieval)

```python
# lambdas/graph_query/query_knowledge_graph.py
import json
import boto3
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection

def query_graph_for_context(query_entity, depth=2, neptune_endpoint=None):
    """
    Query Neptune for entity and its related entities (context).
    Useful for finding connected information.
    """
    
    if not neptune_endpoint:
        neptune_endpoint = os.environ.get('NEPTUNE_ENDPOINT')
    
    connection = DriverRemoteConnection(
        f'wss://{neptune_endpoint}:8182/gremlin',
        'g'
    )
    g = traversal().withRemote(connection)
    
    try:
        # Find entity and traverse relationships up to depth
        results = g.V() \
            .has('name', query_entity) \
            .repeat(both().simplePath()) \
            .times(depth) \
            .dedup() \
            .valueMap() \
            .toList()
        
        connection.close()
        
        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                'id': result.get('id', [None])[0],
                'name': result.get('name', [None])[0],
                'type': result.get('label', ['UNKNOWN'])[0],
                'description': result.get('description', [''])[0],
                'source_document': result.get('source_document', [''])[0]
            })
        
        return formatted_results
    
    except Exception as e:
        connection.close()
        raise e


def graph_rag_retrieval(user_query, neptune_endpoint=None):
    """
    Graph RAG: Extract intent from user query, find matching entity in graph,
    and return entity + its relationships as context.
    """
    
    bedrock = boto3.client('bedrock-runtime')
    
    # Step 1: Extract entities from user query
    prompt = f"""
    Extract the main entity or concept the user is asking about.
    Return ONLY the entity name, nothing else.
    
    User query: {user_query}
    Entity:
    """
    
    response = bedrock.invoke_model(
        modelId='us.anthropic.claude-sonnet-4-20250514-v1:0',
        body=json.dumps({
            "anthropic_version": "bedrock-2023-06-01",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": prompt}]
        })
    )
    
    query_entity = json.loads(response['body'].read())['content'][0]['text'].strip()
    
    # Step 2: Query graph for related entities
    graph_context = query_graph_for_context(query_entity, depth=2, neptune_endpoint=neptune_endpoint)
    
    # Step 3: Format as context for LLM
    context_text = f"Found entity: {query_entity}\n\nRelated entities:\n"
    for item in graph_context:
        context_text += f"- {item['name']} ({item['type']}): {item['description']}\n"
    
    return {
        'query_entity': query_entity,
        'related_entities': graph_context,
        'context_text': context_text
    }


def lambda_handler(event, context):
    """Lambda handler for graph-based retrieval."""
    
    user_query = event.get('query')
    
    if not user_query:
        return {'statusCode': 400, 'error': 'query required'}
    
    try:
        result = graph_rag_retrieval(user_query)
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'error': str(e)
        }
```

### Step 5: Create Complete Data Pipeline (Ingest → Extract → Build Graph)

```python
# lambdas/document_processor/process_document.py
"""
Complete pipeline: Document → Entity Extraction → Graph Building
Triggered when new documents are added to S3.
"""

import json
import boto3
import os

s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

def extract_text_from_document(bucket, key):
    """Extract text from S3 document (simple version for text/HTML)."""
    
    obj = s3.get_object(Bucket=bucket, Key=key)
    content = obj['Body'].read()
    
    # For markdown/text files
    if key.endswith(('.txt', '.md')):
        return content.decode('utf-8')
    
    # For HTML files (simple strip)
    elif key.endswith('.html'):
        import re
        html = content.decode('utf-8')
        # Remove script/style tags
        clean = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL)
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', clean)
        return clean
    
    else:
        return ""


def lambda_handler(event, context):
    """
    S3 trigger: When document uploaded → Extract text → Extract entities → Build graph
    """
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    document_id = key.split('/')[-1].replace('.txt', '').replace('.md', '')
    
    # Step 1: Extract text
    text = extract_text_from_document(bucket, key)
    if not text:
        return {'statusCode': 400, 'error': 'Could not extract text'}
    
    # Step 2: Entity extraction
    entity_extraction_lambda = os.environ.get('ENTITY_EXTRACTION_LAMBDA')
    
    entity_response = lambda_client.invoke(
        FunctionName=entity_extraction_lambda,
        InvocationType='RequestResponse',
        Payload=json.dumps({
            'document_id': document_id,
            'document_text': text
        })
    )
    
    extracted = json.loads(entity_response['Payload'].read())
    if extracted.get('statusCode') != 200:
        return {'statusCode': 500, 'error': 'Entity extraction failed'}
    
    extracted_data = json.loads(extracted['body'])
    
    # Step 3: Build knowledge graph
    graph_builder_lambda = os.environ.get('GRAPH_BUILDER_LAMBDA')
    
    graph_response = lambda_client.invoke(
        FunctionName=graph_builder_lambda,
        InvocationType='RequestResponse',
        Payload=json.dumps(extracted_data)
    )
    
    graph_result = json.loads(graph_response['Payload'].read())
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'document_id': document_id,
            'entities_extracted': len(extracted_data.get('entities', [])),
            'relationships_extracted': len(extracted_data.get('relationships', [])),
            'graph_build_result': json.loads(graph_result.get('body', '{}'))
        })
    }
```

---

## Part 3: Comparison of Three RAG Approaches

| Aspect | Vector-Based (Current) | Vector-Less (BM25) | Graph-Based |
|--------|----------------------|-------------------|------------|
| **Retrieval Method** | Semantic similarity | Keyword/TF-IDF | Graph traversal |
| **Cost** | $$ (embeddings + vector DB) | $ (OpenSearch only) | $$$ (Neptune + processing) |
| **Latency** | Medium (embedding lookup) | Fast (keyword search) | Medium (graph queries) |
| **Semantic Understanding** | Excellent | Good (handles synonyms) | Excellent (context-aware) |
| **Scalability** | Good (vector indexing) | Excellent (BM25) | Good (graph queries) |
| **Use Cases** | General Q&A, semantic search | Precise fact retrieval, legal docs | Complex reasoning, relationship discovery |
| **Setup Complexity** | High | Low | High |

---

## Part 4: Unified Agent Architecture

Create an agent that chooses the best retrieval method:

```python
# lambdas/intelligent_retrieval/router.py
"""
Smart routing: Analyze query and choose optimal retrieval method.
"""

def route_query(user_query, context=None):
    """
    Analyze query characteristics and route to best retrieval method.
    """
    
    query_lower = user_query.lower()
    
    # Route logic
    if any(word in query_lower for word in ['what', 'who', 'where', 'when', 'how many']):
        # Factual questions → BM25 (precise)
        return 'BM25'
    
    elif any(word in query_lower for word in ['relate', 'connect', 'impact', 'affect', 'depend']):
        # Relationship questions → Graph
        return 'GRAPH'
    
    elif any(word in query_lower for word in ['similar', 'like', 'comparable', 'example']):
        # Semantic questions → Vector
        return 'VECTOR'
    
    else:
        # Default: Hybrid (combine all)
        return 'HYBRID'


def intelligent_retrieval(user_query, context=None):
    """Execute retrieval using best strategy."""
    
    retrieval_method = route_query(user_query, context)
    
    if retrieval_method == 'BM25':
        return retrieve_bm25(user_query)
    elif retrieval_method == 'GRAPH':
        return graph_rag_retrieval(user_query)
    elif retrieval_method == 'VECTOR':
        return retrieve_vector(user_query)
    else:  # HYBRID
        results = {
            'vector': retrieve_vector(user_query),
            'bm25': retrieve_bm25(user_query),
            'graph': graph_rag_retrieval(user_query)
        }
        return merge_results(results)
```

---

## Part 5: Deployment Steps

### Step 1: Add to Terraform

```bash
# Create new modules
mkdir -p terraform/modules/{bm25,neptune,entity-extraction}
mkdir -p lambdas/{vectorless_retrieval,entity_extraction,graph_builder,graph_query}
```

### Step 2: Update Lambda Roles

```hcl
# terraform/iam.tf - Add permissions
resource "aws_iam_role_policy" "lambda_neptune" {
  name   = "${local.resource_prefix}-lambda-neptune"
  role   = aws_iam_role.lambda_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["neptune-db:*"]
      Resource = "${aws_neptune_cluster.knowledge_graph.arn}/*"
    }]
  })
}
```

### Step 3: Deploy

```bash
cd /Users/grawat/Downloads/aws_project/rag_pipeline_implementation

# Install Python dependencies
poetry add opensearch-py gremlin-python

# Update Terraform
terraform init
terraform plan -var-file=environments/dev.tfvars
terraform apply -var-file=environments/dev.tfvars
```

---

## Cost Comparison (Monthly Estimate - us-east-1)

| Approach | S3 | Compute | Storage | Total |
|----------|----|---------|-----------| ----|
| Vector (Bedrock KB) | $10 | $50 (Lambda) | $350 (OpenSearch min) | **$410** |
| Vector-Less (BM25) | $10 | $30 (Lambda) | $350 (OpenSearch min) | **$390** |
| Graph (Neptune) | $10 | $80 (Lambda + Bedrock) | $600 (Neptune min) | **$690** |
| Hybrid (All 3) | $10 | $150 (all Lambdas) | $800-900 | **$960-1010** |

---

## Recommendations

1. **Start with Vector-Less (BM25)**: Lowest cost, fastest to implement, handles 80% of use cases
2. **Add Graph RAG**: When you need relationship discovery or complex reasoning
3. **Keep Vector Search**: For semantic similarity and general Q&A
4. **Use Hybrid Routing**: Intelligent query router that chooses best method

Would you like me to implement any specific part of this guide in your project?
