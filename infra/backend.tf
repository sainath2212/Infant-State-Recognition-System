terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.70"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

/*
  backend "s3" {
    bucket         = "crynet-tf-state-480009894503"
    key            = "crynet/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "crynet-tf-locks"
    encrypt        = true
  }
*/
}
