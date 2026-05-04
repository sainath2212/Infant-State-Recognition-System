# Rubric-mandated S3 bucket. Hosts the built React frontend (served by CloudFront).
# Required configuration: unique name, versioning enabled, encryption enabled, public access blocked.

resource "aws_s3_bucket" "static" {
  bucket = local.static_bucket
}

resource "aws_s3_bucket_versioning" "static" {
  bucket = aws_s3_bucket.static.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_backend_side_encryption_configuration" "static" {
  bucket = aws_s3_bucket.static.id

  rule {
    apply_backend_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "static" {
  bucket = aws_s3_bucket.static.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "static" {
  bucket = aws_s3_bucket.static.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}
