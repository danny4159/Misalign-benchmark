#!/bin/bash

# Check if there are less than 2 arguments
if [ $# -lt 2 ]; then
  echo "Usage: $0 <misalign_x> <misalign_y>"
  echo "Both <misalign_x> and <misalign_y> are required."
  exit 1
fi

X=$1
Y=$2
echo "Starting training (background) for X${X}Y${Y} misalignment"

# The name of the session to start
SESSION_NAME="my_session"

# Check if the session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

# If the session does not exist, create it
if [ $? != 0 ]; then
  tmux new-session -d -s $SESSION_NAME
  echo "Session $SESSION_NAME created."
else
  echo "Session $SESSION_NAME already exists."
  RANDOM_LETTER=$(head /dev/urandom | tr -dc 'a-z' | head -c 1 ; echo '')
  SESSION_NAME="${SESSION_NAME}_${RANDOM_LETTER}"
  tmux new-session -d -s $SESSION_NAME
  echo "Created a new session with $SESSION_NAME"
fi

tmux new-window -t ${SESSION_NAME}:1 -n 'CycleGAN' "python misalign/train.py model='cgan.yaml' data.misalign_x=${X} data.misalign_y=${Y}"
sleep 20s
tmux new-window -t ${SESSION_NAME}:2 -n 'GC_CycleGAN' "python misalign/train.py model='gcgan.yaml' data.misalign_x=${X} data.misalign_y=${Y}"
sleep 20s
tmux new-window -t ${SESSION_NAME}:3 -n 'SC_CycleGAN' "python misalign/train.py model='scgan.yaml' data.misalign_x=${X} data.misalign_y=${Y}"
sleep 20s
tmux new-window -t ${SESSION_NAME}:4 -n 'PGAN' "python misalign/train.py model='pgan.yaml' data.misalign_x=${X} data.misalign_y=${Y}"