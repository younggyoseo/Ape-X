###################################################################################
# HOW TO USE                                                                      #
###################################################################################
# This `variables.tf` is used to define variables and enter their default values. # 
# Read them and enter your specific value in `terraform.tfvars`.                  #
# Terraform will automatically read `terraform.tfvars` and use them to apply/plan.#
###################################################################################

# Your access key.
variable "access_key" {}

# Your secret key.
variable "secret_key" {}

# Your region.
variable "region" {
    default = "ap-northeast-2"
}

# Number of nodes(instances) to run.
variable "num_actor_node" {
    default = 12
}

# Number of actors which runs on each node
variable "actor_per_node" {
    default = 4
}

# You are able to block access from non-specified IP address by setting specific cidr.
variable "ssh_cidr" {}

# Absolute / Relative path to your ssh key file(*.pem)
variable "ssh_key_file" {}

# Key Pair name
variable "aws_key_name" {}

# Your availablity zone in your region.
variable "availability_zone" {
    default = "ap-northeast-2a"
}

# Your name!
variable "proj_owner" {
    default = "belepi93"
}

# AMI ID to use in each instance
variable "learner_ami" {
    default = "ami-00f4bec93372230e3"
}
variable "replay_ami" {
    default = "ami-0a762ffacbdb6d82a"
}
variable "actor_ami" {
    default = "ami-077b11fae3680e1dc"
}

variable "evaluator_ami" {
    default = "ami-077b11fae3680e1dc"
}
