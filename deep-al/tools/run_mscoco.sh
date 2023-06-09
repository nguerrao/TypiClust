#!/bin/bash

python train_al.py --cfg ../configs/mscoco/al/RESNET50.yaml --al probcover --exp-name auto --initial_size 0 --budget 1574 --delta 0.56 --clip_selection True --const_threshold 1.3

python train_al.py --cfg ../configs/mscoco/al/RESNET50.yaml --al probcover --exp-name auto --initial_size 0 --budget 1574 --delta 0.6 --clip_selection True --const_threshold 1.3

python train_al.py --cfg ../configs/mscoco/al/RESNET50.yaml --al probcover --exp-name auto --initial_size 0 --budget 1574 --delta 0.7 --clip_selection True --const_threshold 1.3