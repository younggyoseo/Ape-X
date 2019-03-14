#!/bin/bash
git clone https://github.com/belepi93/Ape-X.git
tmux new -s replay -d "source anaconda3/bin/activate apex_torch; cd Ape-X; python replay.py; read"
tmux new -s tensorboard -d "source anaconda3/bin/activate apex_torch; cd Ape-X; tensorboard --logdir=runs; read"
sleep 3
