resource "aws_db_subnet_group" "mysql" {
  name       = local.db_subnet_grp
  subnet_ids = data.aws_subnets.default.ids
}

resource "random_password" "db_password" {
  length           = 16
  special          = true
  override_special = "!#$%&*()-_=+[]{}<>:?" # Removed '/', '@', '"', ' ' which RDS doesn't like for MasterUserPassword
}

resource "aws_db_instance" "mysql" {
  identifier              = local.rds_identifier
  engine                  = "mysql"
  engine_version          = "8.0"
  instance_class          = "db.t3.micro"
  allocated_storage       = 20
  storage_type            = "gp3"
  storage_encrypted       = true
  db_name                 = var.db_name
  username                = var.db_username
  password                = random_password.db_password.result
  port                    = 3306
  publicly_accessible     = false
  multi_az                = false
  skip_final_snapshot     = true
  deletion_protection     = false
  apply_immediately       = true
  db_subnet_group_name    = aws_db_subnet_group.mysql.name
  vpc_security_group_ids  = [aws_security_group.rds.id]
  backup_retention_period = 0
}

# Store the assembled DATABASE_URL in Secrets Manager so ECS tasks can read it.
resource "aws_secretsmanager_secret" "database_url" {
  name                    = "${local.name_prefix}/database-url/${local.unique_suffix}"
  description             = "DATABASE_URL for the ECS backend (mysql://...)"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "database_url" {
  secret_id     = aws_secretsmanager_secret.database_url.id
  secret_string = "mysql://${var.db_username}:${random_password.db_password.result}@${aws_db_instance.mysql.endpoint}/${var.db_name}"
}

resource "aws_secretsmanager_secret" "jwt_secret" {
  name                    = "${local.name_prefix}/jwt-secret/${local.unique_suffix}"
  description             = "JWT signing secret for the backend"
  recovery_window_in_days = 0
}

resource "aws_secretsmanager_secret_version" "jwt_secret" {
  secret_id     = aws_secretsmanager_secret.jwt_secret.id
  secret_string = "${random_id.suffix.hex}-jwt-secret-change-me"
}
