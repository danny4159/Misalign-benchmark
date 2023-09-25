#!/bin/bash

mamba activate misalign

python misalign/train.py model='proposed_A_to_B.yaml' data.misalign_x=0 data.misalign_y=5 data.degree=2.5 data.motion_prob=0 data.deform_prob=0 model.params.lambda_l1=50 model.params.lambda_style=25 model.params.lambda_smooth=100 data.reverse=True tags='LRE_CTX_noise5';
python misalign/train.py model='proposed_A_to_B.yaml' data.misalign_x=0 data.misalign_y=0 data.degree=0 data.motion_prob=0.05 data.deform_prob=0 model.params.lambda_l1=50 model.params.lambda_style=50 model.params.lambda_smooth=500 data.reverse=True tags='LRE_CTX_motion';
python misalign/train.py model='proposed_A_to_B.yaml' data.misalign_x=0 data.misalign_y=0 data.degree=0 data.motion_prob=0 data.deform_prob=0.05 model.params.lambda_l1=50 model.params.lambda_style=50 model.params.lambda_smooth=500 data.reverse=True tags='LRE_CTX_deform'
