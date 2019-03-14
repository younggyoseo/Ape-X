conda create -y -n RL python=3.7
source activate RL
conda install -y -c pytorch pytorch-nightly-cpu
conda install -y opencv pyzmq numpy==1.16.2
pip install gym[atari] tensorboardX tensorflow
