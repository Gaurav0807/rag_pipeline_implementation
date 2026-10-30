# ==========================================
# QUANTIZATION LAMBDA
# Sits between Bedrock KB and OpenSearch
# Converts float32 → int8 vectors (75% storage reduction)
# NO CHANGES to existing KB or data source needed
# ==========================================

# Lambda execution role
resource "aws_iam_role" "quantizer_lambda_role" {
  name = "${local.resource_prefix}-quantizer-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

# Policies
resource "aws_iam_role_policy" "quantizer_lambda_policy" {
  name = "${local.resource_prefix}-quantizer-lambda-policy"
  role = aws_iam_role.quantizer_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "opensearchserverless:APIAccessAll",
          "aoss:APIAccessAll"
        ]
        Resource = "*"
      }
    ]
  })
}

# Lambda function for quantization
resource "aws_lambda_function" "vector_quantizer" {
  filename      = "lambda_quantizer.zip"
  function_name = "${local.resource_prefix}-vector-quantizer"
  role          = aws_iam_role.quantizer_lambda_role.arn
  handler       = "lambda_quantizer.lambda_handler"
  runtime       = "python3.11"
  timeout       = 300

  environment {
    variables = {
      COLLECTION_ENDPOINT = aws_opensearchserverless_collection.vectors.collection_endpoint
      INDEX_NAME          = local.index_name
    }
  }

  vpc_config {
    subnet_ids         = var.subnet_ids        # Add to variables.tf
    security_group_ids = var.security_group_ids # Add to variables.tf
  }

  # Dependencies
  depends_on = [aws_iam_role_policy.quantizer_lambda_policy]
}

# EventBridge rule - triggers Lambda after KB ingestion completes
resource "aws_cloudwatch_event_rule" "kb_ingestion_complete" {
  name        = "${local.resource_prefix}-kb-ingestion-complete"
  description = "Trigger quantization after KB ingestion"

  event_pattern = jsonencode({
    source      = ["aws.bedrock"]
    detail-type = ["Bedrock Knowledge Base Ingestion Status"]
    detail = {
      status = ["COMPLETE"]
    }
  })
}

# EventBridge target
resource "aws_cloudwatch_event_target" "invoke_quantizer" {
  rule      = aws_cloudwatch_event_rule.kb_ingestion_complete.name
  target_id = "QuantizerLambda"
  arn       = aws_lambda_function.vector_quantizer.arn

  input = jsonencode({
    collection_endpoint = aws_opensearchserverless_collection.vectors.collection_endpoint
    index_name          = local.index_name
  })
}

# Permission for EventBridge to invoke Lambda
resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.vector_quantizer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.kb_ingestion_complete.arn
}

# Output
output "quantizer_lambda_arn" {
  value = aws_lambda_function.vector_quantizer.arn
  description = "ARN of the vector quantization Lambda function"
}

output "quantizer_lambda_name" {
  value = aws_lambda_function.vector_quantizer.function_name
  description = "Name of the vector quantization Lambda function"
}
