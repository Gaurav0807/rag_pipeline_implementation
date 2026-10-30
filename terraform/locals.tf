locals {
    resource_prefix = "${var.project_name}-${var.environment}"
    collection_name = "${local.resource_prefix}-vectors"
    index_name      = "${local.resource_prefix}-index"
}