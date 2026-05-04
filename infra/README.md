# ShopSmart — Terraform Infrastructure

Provisions:

- S3 bucket (rubric-mandated: versioning, encryption, public access blocked)
- CloudFront distribution serving the frontend from S3
- ECR repository for the backend image
- RDS MySQL `db.t3.micro`
- AWS Secrets Manager entries for `DATABASE_URL` and `JWT_SECRET`
- ECS Fargate cluster + ALB + task definition + service (2 tasks)
- Security groups: ALB → ECS → RDS, layered

## Prerequisites

- Terraform ≥ 1.6
- AWS Academy Learner Lab credentials sourced into the shell:
  `source ../.env.aws.local`
- Bootstrap state bucket and DynamoDB lock table already exist
  (created once via AWS CLI; see project root README)

## Common workflow

```bash
cd infra
source ../.env.aws.local

# Pass DB password without writing it to disk
export TF_VAR_db_password="$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)"

terraform init
terraform validate
terraform plan -out=tfplan
terraform apply tfplan
terraform output
```

When you're done for the day:

```bash
terraform destroy
```

## Why LabRole everywhere

AWS Academy doesn't allow creating new IAM roles. Both the ECS task execution
role and the ECS task role are set to the pre-provisioned `LabRole`. That role
already has permissions for ECR pulls, CloudWatch logs, and Secrets Manager reads.

## What lives where

| File            | Purpose                                   |
| --------------- | ----------------------------------------- |
| `backend.tf`    | Provider versions and remote state config |
| `providers.tf`  | AWS provider, default tags, identity data |
| `variables.tf`  | Inputs (region, project name, db creds)   |
| `locals.tf`     | Computed names + random suffix            |
| `vpc.tf`        | Default VPC + subnet data sources         |
| `security.tf`   | ALB / ECS / RDS security groups           |
| `s3.tf`         | Rubric-mandated frontend bucket           |
| `ecr.tf`        | Backend container registry                |
| `rds.tf`        | MySQL + Secrets Manager                   |
| `ecs.tf`        | Cluster, ALB, task def, service           |
| `cloudfront.tf` | CDN in front of S3                        |
| `outputs.tf`    | URLs and ARNs you'll need next            |
