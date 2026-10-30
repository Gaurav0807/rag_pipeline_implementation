output "knowledge_base_id" {
    value = aws_bedrockagent_knowledge_base.main.id
}

output "data_source_id" {
    value = aws_bedrockagent_data_source.main.id
}

output "opensearch_endpoint" {
    value = aws_opensearchserverless_collection.vectors.collection_endpoint
}

output "opensearch_arn" {
    value = aws_opensearchserverless_collection.vectors.arn
}