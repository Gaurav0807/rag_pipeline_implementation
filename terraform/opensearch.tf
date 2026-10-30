
# POLICY 1: Encryption
# Defines HOW data is encrypted at rest
# Must exist BEFORE creating the collection


resource "aws_opensearchserverless_security_policy" "encryption" {
    name = "${local.resource_prefix}-enc"
    type = "encryption"

    policy = jsonencode({
        Rules = [
        {
            ResourceType = "collection"
            Resource     = ["collection/${local.collection_name}"]
        }
        ]
        AWSOwnedKey = true
    })
}


# POLICY 2: Network
# Defines WHO can reach the collection endpoint
# Public = accessible from internet (fine for learning)

resource "aws_opensearchserverless_security_policy" "network" {
    name = "${local.resource_prefix}-net"
    type = "network"

    policy = jsonencode([
        {
        Rules = [
            {
            ResourceType = "collection"
            Resource     = ["collection/${local.collection_name}"]
            },
            {
            ResourceType = "dashboard"
            Resource     = ["collection/${local.collection_name}"]
            }
        ]
        AllowFromPublic = true
        }
    ])
}


# POLICY 3: Data Access
# Defines WHICH IAM roles can create indexes, read/write documents
# We give access to: KB role, Lambda role, and your account

resource "aws_opensearchserverless_access_policy" "data" {
    name = "${local.resource_prefix}-data"
    type = "data"

    policy = jsonencode([
        {
        Rules = [
            {
            ResourceType = "collection"
            Resource     = ["collection/${local.collection_name}"]
            Permission = [
                "aoss:CreateCollectionItems",
                "aoss:DeleteCollectionItems",
                "aoss:UpdateCollectionItems",
                "aoss:DescribeCollectionItems"
            ]
            },
            {
            ResourceType = "index"
            Resource     = ["index/${local.collection_name}/*"]
            Permission = [
                "aoss:CreateIndex",
                "aoss:DeleteIndex",
                "aoss:UpdateIndex",
                "aoss:DescribeIndex",
                "aoss:ReadDocument",
                "aoss:WriteDocument"
            ]
            }
        ]
        Principal = [
            aws_iam_role.kb_role.arn,
            aws_iam_role.lambda_role.arn,
            "arn:aws:iam::${var.aws_account_id}:root"
        ]
        }
    ])
}


# THE COLLECTION (Vector Database)
# This is the actual OpenSearch instance
# Type = VECTORSEARCH means it supports knn_vector fields
# A collection is like a database server that holds your indexes (tables)
#for serverless we don;t need to provide memory, disk, compute 
# """
#   There are 3 types of collections in OpenSearch Serverless:

#   ---
#   All 3 types:
  
#   ┌──────────────┬─────────────────────────┬───────────────────────────────────────────────┐
#   │     Type     │         Purpose         │               What it's good at               │
#   ├──────────────┼─────────────────────────┼───────────────────────────────────────────────┤
#   │ VECTORSEARCH │ AI/ML similarity search │ Find similar meanings using vectors           │
#   ├──────────────┼─────────────────────────┼───────────────────────────────────────────────┤
#   │ SEARCH       │ Full-text search        │ Find exact/partial word matches (like Google) │
#   ├──────────────┼─────────────────────────┼───────────────────────────────────────────────┤
#   │ TIMESERIES   │ Time-based data         │ Logs, metrics, events ordered by timestamp    │
#   └──────────────┴─────────────────────────┴───────────────────────────────────────────────┘

# """
resource "aws_opensearchserverless_collection" "vectors" {
    name = local.collection_name
    type = "VECTORSEARCH"

    depends_on = [
        aws_opensearchserverless_security_policy.encryption,
        aws_opensearchserverless_security_policy.network,
        aws_opensearchserverless_access_policy.data
    ]
}



