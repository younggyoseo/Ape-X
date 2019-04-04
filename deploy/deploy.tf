# Terraform Code to deploy Ape-X

output "subnet_cidr" {
    value = "${aws_default_subnet.default.cidr_block}"
}

output "learner_ip" {
    value = "${aws_instance.learner.public_ip}"
}

output "replay_ip" {
    value = "${aws_instance.replay.public_ip}"
}

output "actor_ips" {
    value = "${aws_instance.actor.*.public_ip}"
}

provider "aws" {
    access_key = "${var.access_key}"
    secret_key = "${var.secret_key}"
    region = "${var.region}"
}


resource "aws_default_subnet" "default" {
    availability_zone = "${var.availability_zone}"

    tags {
        Name = "Default subnet"
    }
}


resource "aws_security_group" "learner" {
    name = "learner-sg"
    description = "Security group for Ape-X learner"

    # For learner bindings
    ingress {
        from_port = 52001
        to_port = 52002
        protocol = "tcp"
        cidr_blocks = ["${aws_default_subnet.default.cidr_block}"]
    }

    # For ssh
    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    # For tensorboard
    ingress {
        from_port = 6006
        to_port = 6006
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    egress {
        from_port = 0
        to_port = 0
        protocol = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags {
        Name = "learner-sg"
        Owner = "${var.proj_owner}"
    }
}


resource "aws_security_group" "replay" {
    name = "replay-sg"
    description = "Security group for Ape-X replay"

    # For replay bindings
    ingress {
        from_port = 51001
        to_port = 51003
        protocol = "tcp"
        cidr_blocks = ["${aws_default_subnet.default.cidr_block}"]
    }

    # For ssh
    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }
    # For Tensorboard
    ingress {
        from_port = 6006
        to_port = 6006
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    egress {
        from_port = 0
        to_port = 0
        protocol = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags {
        Name = "replay-sg"
        Owner = "${var.proj_owner}"
    }
}


resource "aws_security_group" "actor" {
    name = "actor-sg"
    description = "Security group for Ape-X actor"

    # For ssh
    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    # For tensorboard
    ingress {
        from_port = 6006
        to_port = 6006
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    egress {
        from_port = 0
        to_port = 0
        protocol = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags {
        Name = "actor-sg"
        Owner = "${var.proj_owner}"
    }
}

resource "aws_security_group" "evaluator" {
    name = "evaluator-sg"
    description = "Security group for Ape-X evaluator"

    # For ssh
    ingress {
        from_port = 22
        to_port = 22
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    # For tensorboard
    ingress {
        from_port = 6006
        to_port = 6006
        protocol = "tcp"
        cidr_blocks = ["${var.ssh_cidr}"]
    }

    egress {
        from_port = 0
        to_port = 0
        protocol = "-1"
        cidr_blocks = ["0.0.0.0/0"]
    }

    tags {
        Name = "actor-sg"
        Owner = "${var.proj_owner}"
    }
}


resource "aws_instance" "replay" {
    ami = "${var.replay_ami}"
    instance_type = "r5.4xlarge"
    key_name = "${var.aws_key_name}"
    vpc_security_group_ids = ["${aws_security_group.replay.id}"]
    subnet_id = "${aws_default_subnet.default.id}"

    provisioner "file" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
            agent = false
        }
        source = "replay.sh"
        destination = "/tmp/replay.sh"
    }


    provisioner "remote-exec" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
            agent = false
        }
        inline = [
            "chmod +x /tmp/replay.sh",
            "bash /tmp/replay.sh",
        ]
    }


    tags {
        Name = "ape-x-replay"
        Owner = "${var.proj_owner}"
    }
}


# learner node
resource "aws_instance" "learner" {
    ami = "${var.learner_ami}"
    instance_type = "p3.2xlarge"
    key_name = "${var.aws_key_name}"
    vpc_security_group_ids = ["${aws_security_group.learner.id}"]
    subnet_id = "${aws_default_subnet.default.id}"
    depends_on = ["aws_instance.replay"]

    provisioner "file" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
            agent = false
        }
        source = "learner.sh"
        destination = "/tmp/learner.sh"
    }

    provisioner "remote-exec" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
            agent = false
        }
        inline = [
            "export REPLAY_IP=${aws_instance.replay.private_ip}",
            "export N_NODE=${var.num_actor_node}",
            "export ACTOR_PER_NODE=${var.actor_per_node}",
            "chmod +x /tmp/learner.sh",
            "bash /tmp/learner.sh",
        ]
    }

    tags {
        Name = "ape-x-learner"
        Owner = "${var.proj_owner}"
    }
}


# actor node
resource "aws_instance" "actor" {
    ami = "${var.actor_ami}"
    instance_type = "m5.xlarge"
    key_name = "${var.aws_key_name}"
    vpc_security_group_ids = ["${aws_security_group.actor.id}"]
    subnet_id = "${aws_default_subnet.default.id}"
    count = "${var.num_actor_node}"
    depends_on = ["aws_instance.replay", "aws_instance.learner"]

    provisioner "file" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
            agent = false
        }
        source = "actor.sh"
        destination = "/tmp/actor.sh"
    }

    provisioner "remote-exec" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
            agent = false
        }
        inline = [
            "export REPLAY_IP=${aws_instance.replay.private_ip}",
            "export LEARNER_IP=${aws_instance.learner.private_ip}",
            "export NODE_ID=${count.index}",
            "export N_NODE=${var.num_actor_node}",
            "export ACTOR_PER_NODE=${var.actor_per_node}",
            "chmod +x /tmp/actor.sh",
            "bash /tmp/actor.sh",
        ]
    }

    tags {
        Name = "ape-x-actor${count.index}"
        Owner = "${var.proj_owner}"
    }
}

# eval node
resource "aws_instance" "evaluator" {
    ami = "${var.evaluator_ami}"
    instance_type = "m5.xlarge"
    key_name = "${var.aws_key_name}"
    vpc_security_group_ids = ["${aws_security_group.evaluator.id}"]
    subnet_id = "${aws_default_subnet.default.id}"
    depends_on = ["aws_instance.learner"]

    provisioner "file" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
            agent = false
        }
        source = "evaluator.sh"
        destination = "/tmp/evaluator.sh"
    }

    provisioner "remote-exec" {
        connection {
            type = "ssh"
            user = "ubuntu"
            private_key = "${file("${var.ssh_key_file}")}"
            agent = false
        }
        inline = [
            "export LEARNER_IP=${aws_instance.learner.private_ip}",
            "chmod +x /tmp/evaluator.sh",
            "bash /tmp/evaluator.sh",
        ]
    }

    tags {
        Name = "ape-x-evaluator"
        Owner = "${var.proj_owner}"
    }
}