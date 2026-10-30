#   Knowledge Base = The "brain" (HOW to process + WHERE to store)
#   Data Source    = The "input" (FROM WHERE to read + HOW to chunk)

resource "aws_bedrockagent_knowledge_base" "main" {
    name     = "${local.resource_prefix}-kb"
    role_arn = aws_iam_role.kb_role.arn

    # WHAT: How to create embeddings
    # HOW: Which model converts text → vectors
    knowledge_base_configuration {
        type = "VECTOR"

        vector_knowledge_base_configuration {
        embedding_model_arn = "arn:aws:bedrock:${var.aws_region}::foundation-model/${var.embedding_model}"
        }
    }

    # WHERE: Where to store the vectors
    # WHERE: Where vectors get stored
    storage_configuration {
        type = "OPENSEARCH_SERVERLESS"

        opensearch_serverless_configuration {
        collection_arn    = aws_opensearchserverless_collection.vectors.arn
        vector_index_name = local.index_name

        field_mapping {
            vector_field   = "AMAZON_BEDROCK_VECTOR"
            text_field     = "AMAZON_BEDROCK_TEXT"
            metadata_field = "AMAZON_BEDROCK_METADATA"
        }
        }
    }
}
