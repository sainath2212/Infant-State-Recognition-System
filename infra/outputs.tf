output "static_bucket_name" {
  description = "Rubric-mandated S3 bucket. Used for build artifacts."
  value       = aws_s3_bucket.static.id
}

output "alb_dns_name" {
  description = "Public ALB DNS. App is at http://<this>/, API at http://<this>/api/..."
  value       = aws_lb.main.dns_name
}

output "ecr_backend_repo" {
  description = "ECR repository URL for the backend image."
  value       = aws_ecr_repository.backend.repository_url
}

output "ecr_frontend_repo" {
  description = "ECR repository URL for the frontend (nginx) image."
  value       = aws_ecr_repository.frontend.repository_url
}

output "ecs_cluster" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "ecs_backend_service" {
  description = "ECS backend service name."
  value       = aws_ecs_service.backend.name
}

output "ecs_frontend_service" {
  description = "ECS frontend service name."
  value       = aws_ecs_service.frontend.name
}

output "rds_endpoint" {
  description = "RDS MySQL endpoint."
  value       = aws_db_instance.mysql.endpoint
  sensitive   = true
}

output "database_url_secret_arn" {
  description = "Secrets Manager ARN containing DATABASE_URL."
  value       = aws_secretsmanager_secret.database_url.arn
}

output "jwt_secret_arn" {
  description = "Secrets Manager ARN containing JWT_SECRET."
  value       = aws_secretsmanager_secret.jwt_secret.arn
}
