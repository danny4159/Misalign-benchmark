#!/bin/bash

SESSION_NAME=PGAN

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

tmux new-window -t ${SESSION_NAME}:1 -n 'PGAN_0_0_0_0_0' "python misalign/train.py model='pgan.yaml' data.misalign_x=0 data.misalign_y=0 data.degree=0 data.motion_prob=0 data.deform_prob=0"
sleep 5s
tmux new-window -t ${SESSION_NAME}:2 -n 'PGAN_0_1_0.5_0_0' "python misalign/train.py model='pgan.yaml' data.misalign_x=0 data.misalign_y=1 data.degree=0.5 data.motion_prob=0 data.deform_prob=0"
sleep 5s
tmux new-window -t ${SESSION_NAME}:3 -n 'PGAN_0_2_1_0_0' "python misalign/train.py model='pgan.yaml' data.misalign_x=0 data.misalign_y=2 data.degree=1 data.motion_prob=0 data.deform_prob=0"
sleep 5s
tmux new-window -t ${SESSION_NAME}:4 -n 'PGAN_0_3_1.5_0_0' "python misalign/train.py model='pgan.yaml' data.misalign_x=0 data.misalign_y=3 data.degree=1.5 data.motion_prob=0 data.deform_prob=0"
sleep 5s
tmux new-window -t ${SESSION_NAME}:5 -n 'PGAN_0_4_2_0_0' "python misalign/train.py model='pgan.yaml' data.misalign_x=0 data.misalign_y=4 data.degree=2 data.motion_prob=0 data.deform_prob=0"
sleep 5s
tmux new-window -t ${SESSION_NAME}:6 -n 'PGAN_0_5_2.5_0_0' "python misalign/train.py model='pgan.yaml' data.misalign_x=0 data.misalign_y=5 data.degree=2.5 data.motion_prob=0 data.deform_prob=0"
sleep 5s
tmux new-window -t ${SESSION_NAME}:7 -n 'PGAN_0_0_0_0.05_0' "python misalign/train.py model='pgan.yaml' data.misalign_x=0 data.misalign_y=0 data.degree=0 data.motion_prob=0.05 data.deform_prob=0"
sleep 5s
tmux new-window -t ${SESSION_NAME}:8 -n 'PGAN_0_0_0_0_0.05' "python misalign/train.py model='pgan.yaml' data.misalign_x=0 data.misalign_y=0 data.degree=0 data.motion_prob=0 data.deform_prob=0.05"
