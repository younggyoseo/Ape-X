#!/bin/bash
git clone https://github.com/belepi93/Ape-X.git
tmux new -s evaluator -d "export LEARNER_IP=$LEARNER_IP; source anaconda3/bin/activate apex_torch; cd Ape-X; python eval.py; read"
tmux new -s tensorboard -d "source anaconda3/bin/activate apex_torch; cd Ape-X; tensorboard --logdir=runs; read"
sleep 3