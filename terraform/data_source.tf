# ==========================================
# DATA SOURCE
# Tells the KB: "Read documents from THIS S3 location"
# Also configures HOW to chunk the documents
# ==========================================

resource "aws_bedrockagent_data_source" "main" {
    name                 = "${local.resource_prefix}-datasource"
    knowledge_base_id    = aws_bedrockagent_knowledge_base.main.id
    data_deletion_policy = "RETAIN" #If you delete this data source, keep vectors in OpenSearch (don't delete them)

    # FROM WHERE: Which S3 bucket and prefix
    # FROM WHERE: Which S3 location has the documents
    data_source_configuration {
        type = "S3"

        s3_configuration {
        bucket_arn         = "arn:aws:s3:::${var.s3_bucket_name}"
        inclusion_prefixes = [var.s3_data_prefix]
        }
    }

    # HOW: How to split documents into chunks
    # HOW: How to split documents before embedding
    vector_ingestion_configuration {
        chunking_configuration {
        chunking_strategy = "FIXED_SIZE" #  Split docs into equal-sized chunks  

        fixed_size_chunking_configuration {
            max_tokens         = var.chunk_size # Each chunk = max 512 tokens (~380 words)  
            overlap_percentage = var.chunk_overlap # Each chunk shares 10% text with the previous chunk      
        }
        }
    }
}

