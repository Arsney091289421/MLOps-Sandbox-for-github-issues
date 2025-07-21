provider "aws" {
  region = var.aws_region
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = var.model_bucket

  tags = {
    Name        = "MLOps Model Storage"
    Environment = "Dev"
  }
}

resource "aws_s3_bucket_versioning" "enable_versioning" {
  bucket = aws_s3_bucket.model_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}
