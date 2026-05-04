# Use the default VPC + subnets that AWS Academy provisions in every region.
# Building a VPC from scratch would be cleaner but requires NAT-gateway IAM
# permissions that the Learner Lab doesn't grant.

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
