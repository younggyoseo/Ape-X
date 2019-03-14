#!/bin/bash
conda create -y -n apex_torch python=3.7
source anaconda3/bin/activate apex_torch
conda install -y -c pytorch pytorch-nightly-cpu
conda install -y opencv pyzmq numpy==1.16.2
pip install gym[atari] tensorboardX tensorflow
sudo apt install htop