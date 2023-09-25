#!/bin/bash

python misalign/pretrain.py --nce True --nce_weight 0.01
python misalign/pretrain.py --nce True --nce_weight 0.005
python misalign/pretrain.py --nce True --nce_weight 0.05
python misalign/pretrain.py --nce False --nce_weight 0.0
