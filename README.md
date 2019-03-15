# Ape-X

An Implementation of [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933) (Horgan et al. 2018) in PyTorch.

<img src="https://cl.ly/40b459838c5e/Image%2525202019-03-10%252520at%2525206.53.24%252520PM.png" width="500">

The paper proposes a distributed architecture for deep reinforcement learning with distributed prioritized experience replay. This enables a fast and broad exploration with many actors, which prevents model from learning suboptimal policy.

There are a few implementations which are optimized for powerful single machine with a lot of cores but I tried to implement Ape-X in a multi-node situation with AWS EC2 instances. [ZeroMQ](http://zeromq.org/), [AsyncIO](https://docs.python.org/3/library/asyncio.html), [Multiprocessing](https://docs.python.org/3/library/multiprocessing.html) are really helpful tools for this. 

There are still performance issues with replay server which are caused by the shared memory lock and hyperparameter tuning but this works anyway. Also, there are still some parts  I hard-coded for convenience. I'm trying to improve many parts and really appreciate your help.

# Requirements

```
python 3.7
numpy==1.16.2
torch==1.0.0.dev20190303
pyzmq==18.0.0
opencv-python==4.0.0.21
tensorflow==1.13.0
tensorboardX==1.6
gym==0.12.0
gym[atari]
```

1. `numpy=1.16.0` version has memory leak issue with pickle so try not to use numpy 1.16.0. 
2. CPU performance of `pytorch-nightly-cpu` from conda is much better than normal `torch`
3. `tensorflow` is necessary to use tensorboardX.

# Overall Structure

![image](https://user-images.githubusercontent.com/20944657/54428494-069f1700-4761-11e9-96bc-51ba0b8c39e5.png)


# Result

![image](https://user-images.githubusercontent.com/20944657/54402762-97013b80-4710-11e9-95ba-aca306f5ab3f.png)
![training_speed](https://user-images.githubusercontent.com/20944657/54407775-7b9f2c00-4722-11e9-9409-ec62969aa89d.png)
![eval](figs/eval.gif)

Seaquest result trained with 192 actors. Due to the slow training speed(10~12 batches/s instead of 19 batches/s in paper), It was not possible to reproduce the same result as the paper. But it shows dramatic increase over my baseline implementations(rainbow, acer)

Added gif to show how agent really acts and scores in SeaquestNoFrameskip-v4. I recently noticed that the performance(score) of actor is much better in evaluation setting(epsilon=0.) than the plot 1. 

# How To Use

## Single Machine

My focus was to run Ape-X in a multi-node environment but you can run this model with powerful single machine. I have not run any experiment with single machine so I'm not sure you can achieve satisfactory performance/result. For details, you can see `run.sh` included in this repo.

## Multi-Node with AWS EC2

**Be careful not to include your private AWS secret/access key in public repository while following instructions!**

### Packer

<img src="https://user-images.githubusercontent.com/20944657/54369115-b40a2000-46b8-11e9-8a8d-393e17322052.png" width="200">

[Packer](https://www.packer.io/) is a useful tool to build automated machine images. You'll be able to make AWS AMI with a few line of json formatted file and shell script. There are a lot of more available features in packer's website. I've made all necessary files in `deploy/packer` directory. If you're not interested in using packer, I already included pre-built AMI in `variables.tf` so you can skip this part.

1. Enter your secrey/access key with appropriate IAM policy in `variables.json` file.
2. run below commands in parallel.
    ```
    packer build -var-file=variables.json ape_x_actor.json
    packer build -var-file=variables.json ape_x_replay.json
    packer build -var-file=variables.json ape_x_learner.json
    ```
3. You can see AMIs are created in your AWS account.

### Terraform

<img src="https://user-images.githubusercontent.com/20944657/54369102-b10f2f80-46b8-11e9-8404-96c2a7583bc4.png" width="200">

[Terraform](https://www.terraform.io/) is a useful IaC(Infrastructure as Code) tool. You can start multiple instances with only one command `terraform apply` and destroy all instances with `terraform destroy`. For more information, See Terraform's website tutorial and documentation. I have already included all necessary commands in `deploy` directory. Important files are `deploy.tf`, `variables.tf`, `terraform.tfvars`.

1. Read `variables.tf` and enter necessary values to `terraform.tfvars`. Values included in `terraform.tfvars` will override any default values in `variables.tf`.
2. Change EC2 instance type in `deploy.tf` to meet your budget.
3. Run `terraform init` in `deploy` directory.
4. Run `terraform apply` and terraform will magically create all necessary instances and training will start.
5. To see how trained model works, See tensorboard which includes **actor with larget actor id which has smallest epsilon value**. You could easily access tensorboard by entering **http://public_ip:6006**. Or you could add a evaluator node with a new instance but this costs you more money :(


### Thanks to

1. https://github.com/haje01/distper
2. https://github.com/openai/baselines/