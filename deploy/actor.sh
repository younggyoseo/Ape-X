#!/bin/bash
git clone https://github.com/belepi93/Ape-X.git
idx=0
export N_ACTORS=$(($N_NODE * $ACTOR_PER_NODE))
while [ $idx -lt $ACTOR_PER_NODE ]
do
    ACTOR_ID=$(($NODE_ID * $ACTOR_PER_NODE + $idx))
    tmux new -s "actor-$ACTOR_ID" -d "export LEARNER_IP=$LEARNER_IP; export REPLAY_IP=$REPLAY_IP; export ACTOR_ID=$ACTOR_ID; export N_ACTORS=$N_ACTORS; source anaconda3/bin/activate apex_torch; cd Ape-X; python actor.py; read"
    idx=`expr $idx + 1`
done
tmux new -s tensorboard -d "source anaconda3/bin/activate apex_torch; cd Ape-X; tensorboard --logdir=runs; read"
sleep 3
