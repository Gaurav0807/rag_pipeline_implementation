
variable "aws_region" {
  type = string
  default = "us-east-1"
}

variable "environment" {
  type = string
  default = "sandbox"
}

variable "project_name" {
    type    = string
    default = "rag-pipeline"
}

variable "aws_account_id" {
    type        = string
    default = "302263043826"
}

variable "s3_bucket_name" {
    type    = string
    default = "gaurav-hudi-data"
}

variable "s3_data_prefix" {
    type    = string
    default = "rag_test_data/"
}

variable "embedding_model" {
    type = string
    default = "amazon.titan-embed-text-v2:0"
}

variable "embedding_dimension" {
    type    = number
    default = 1024
}

variable "chunk_size" {
    type    = number
    default = 512
}

variable "chunk_overlap" {
    type    = number
    default = 10
}
