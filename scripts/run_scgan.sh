#!/bin/bash

SESSION_NAME=SCGAN

# Check if the session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

# If the session does not exist, create it
if [ $? != 0 ]; then
  tmux new-session -d -s $SESSION_NAME
  echo "Session $SESSION_NAME created."
else
  echo "Session $SESSION_NAME already exists."
  exit 1
fi

tmux new-window -t ${SESSION_NAME}:1 -n 'SCGAN_0_0' "python misalign/train.py model='scgan.yaml' data.misalign_x=0 data.misalign_y=0 trainer.devices=[0]"
sleep 5s
tmux new-window -t ${SESSION_NAME}:2 -n 'SCGAN_0_1' "python misalign/train.py model='scgan.yaml' data.misalign_x=0 data.misalign_y=1"
sleep 5s
tmux new-window -t ${SESSION_NAME}:3 -n 'SCGAN_0_2' "python misalign/train.py model='scgan.yaml'  data.misalign_x=0 data.misalign_y=2"