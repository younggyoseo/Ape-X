#!/bin/bash
git clone https://github.com/belepi93/Ape-X.git
export N_ACTORS=$(($N_NODE * $ACTOR_PER_NODE))
tmux new -s learner -d "export REPLAY_IP=$REPLAY_IP; export N_ACTORS=$N_ACTORS; source anaconda3/bin/activate apex_torch; cd Ape-X; python learner.py --cuda; read"
tmux new -s tensorboard -d "source anaconda3/bin/activate apex_torch; cd Ape-X; tensorboard --logdir=runs; read"
sleep 3
