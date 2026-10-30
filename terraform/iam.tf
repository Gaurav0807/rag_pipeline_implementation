
# ROLE 1: Knowledge Base Role
# Used by: Bedrock Knowledge Base service
# Purpose: Read docs from S3, generate embeddings, store in OpenSearch

resource "aws_iam_role" "kb_role" {
  name = "${local.resource_prefix}-kb-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
        {
          Effect = "Allow"
          Principal = {
            Service = "bedrock.amazonaws.com"
          }
          Action = "sts:AssumeRole"
          Condition = {
            StringEquals = {
              "aws:SourceAccount" = var.aws_account_id
            }
          }
        }
      ]
  })
}

resource "aws_iam_role_policy" "kb_s3" {
    name = "s3-read"
    role = aws_iam_role.kb_role.id

    policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
        {
            Effect = "Allow"
            Action = [
            "s3:GetObject",
            "s3:ListBucket"
            ]
            Resource = [
            "arn:aws:s3:::gaurav-hudi-data",
            "arn:aws:s3:::gaurav-hudi-data/rag_test_data/*"
            ]
        }
        ]
    })
}


resource "aws_iam_role_policy" "kb_bedrock" {
    name = "bedrock-embed"
    role = aws_iam_role.kb_role.id

    policy = jsonencode({
      Version = "2012-10-17"
      Statement = [
        {
          Effect = "Allow"
          Action = ["bedrock:InvokeModel"]
          Resource = [
            "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0"
          ]
        }
      ]
    })
}

resource "aws_iam_role_policy" "kb_opensearch" {
    name = "opensearch-write"
    role = aws_iam_role.kb_role.id

    policy = jsonencode({
      Version = "2012-10-17"
      Statement = [
        {
          Effect   = "Allow"
          Action   = ["aoss:APIAccessAll"]
          Resource = ["arn:aws:aoss:us-east-1:${var.aws_account_id}:collection/*"]
        }
      ]
    })
}

resource "aws_iam_role" "agent_role" {
    name = "${local.resource_prefix}-agent-role"

    assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
        {
            Effect = "Allow"
            Principal = {
            Service = "bedrock.amazonaws.com"
            }
            Action = "sts:AssumeRole"
            Condition = {
            StringEquals = {
                "aws:SourceAccount" = var.aws_account_id
            }
            }
        }
        ]
    })
}

resource "aws_iam_role_policy" "agent_bedrock" {
    name = "bedrock-invoke"
    role = aws_iam_role.agent_role.id

    policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
        {
            Effect = "Allow"
            Action = [
            "bedrock:InvokeModel",
            "bedrock:Retrieve",
            "bedrock:RetrieveAndGenerate"
            ]
            Resource = ["*"]
        }
        ]
    })
}

# ==========================================
# ROLE 3: Lambda Role
# Used by: Lambda functions (data sync, metrics)
# Purpose: Trigger ingestion jobs, write CloudWatch logs
# ==========================================

resource "aws_iam_role" "lambda_role" {
    name = "${local.resource_prefix}-lambda-role"

    assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
        {
            Effect = "Allow"
            Principal = {
            Service = "lambda.amazonaws.com"
            }
            Action = "sts:AssumeRole"
        }
        ]
    })
}

resource "aws_iam_role_policy" "lambda_bedrock" {
    name = "bedrock-ingestion"
    role = aws_iam_role.lambda_role.id

    policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
        {
            Effect = "Allow"
            Action = [
            "bedrock:StartIngestionJob",
            "bedrock:GetIngestionJob",
            "bedrock:ListIngestionJobs"
            ]
            Resource = ["*"]
        }
        ]
    })
}

# Attach AWS-managed policy for CloudWatch logs
resource "aws_iam_role_policy_attachment" "lambda_logs" {
    role       = aws_iam_role.lambda_role.name
    policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}


