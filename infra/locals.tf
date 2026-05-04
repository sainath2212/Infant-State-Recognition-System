resource "random_id" "suffix" {
  byte_length = 4
}

locals {
  name_prefix    = var.project_name
  unique_suffix  = random_id.suffix.hex
  account_id     = data.aws_caller_identity.current.account_id
  region         = data.aws_region.current.name
  static_bucket  = "${local.name_prefix}-static-${local.unique_suffix}"
  ecr_backend    = "${local.name_prefix}-backend-${local.unique_suffix}"
  ecr_frontend     = "${local.name_prefix}-frontend-${local.unique_suffix}"
  cluster_name   = "${local.name_prefix}-cluster-${local.unique_suffix}"
  service_prefix = "${local.name_prefix}-svc-${local.unique_suffix}"
  alb_name       = "${local.name_prefix}-alb-${local.unique_suffix}"
  rds_identifier = "${local.name_prefix}-mysql-${local.unique_suffix}"
  log_group      = "/ecs/${local.name_prefix}-${local.unique_suffix}"
  tg_backend     = "${local.name_prefix}-tg-be-${local.unique_suffix}"
  tg_frontend      = "${local.name_prefix}-tg-fe-${local.unique_suffix}"
  db_subnet_grp  = "${local.name_prefix}-db-sub-${local.unique_suffix}"
}
