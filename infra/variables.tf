variable "region" {
  description = "AWS region. AWS Academy is locked to us-east-1."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short prefix used for all resource names and tags."
  type        = string
  default     = "crynet"
}

variable "db_name" {
  description = "MySQL database name created on the RDS instance."
  type        = string
  default     = "crynet"
}

variable "db_username" {
  description = "Master username for RDS MySQL."
  type        = string
  default     = "crynet_admin"
}

variable "db_password" {
  description = "Master password for RDS MySQL. Pass via TF_VAR_db_password."
  type        = string
  sensitive   = true
}

variable "container_image_tag" {
  description = "Image tag for the ECS task to pull from ECR. Use 'latest' for the first apply, then specific SHAs."
  type        = string
  default     = "latest"
}

variable "lab_role_arn" {
  description = "AWS Academy Learner Lab pre-provisioned LabRole ARN."
  type        = string
  default     = "arn:aws:iam::480009894503:role/LabRole"
}
