#!/bin/bash
'========================= # Example of commande lines to run the different methods for both datasets ========================='

'========================= PASCAL VOC ========================='

# ProbCover Only 
python train_al.py --cfg ../configs/pascalvoc/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 1000 \
    --delta 0.6 \
    --model_features clip \
    --method probcover 
   

# Random 
python train_al.py --cfg ../configs/pascalvoc/al/RESNET50.yaml \
    --al random \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 1000 \
    --seed 1

# ProbCover with CLIP
python train_al.py --cfg ../configs/pascalvoc/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 4 \
    --delta 0.6 \
    --model_features clip \
    --method clip_selection_max_object_v2 \
    --number_of_samples 20 \
    --seed 1 \
    --const_threshold 0.4 \
    --number_of_smallest_values_to_consider 18


# ProbCover with OWL-ViT
python train_al.py --cfg ../configs/pascalvoc/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 4 \
    --delta 0.6 \
    --model_features clip \
    --method clip_selection_max_object \
    --number_of_samples 20 \
    --topline_count_method per_object_weighted \
    --seed 1 \
    --top_line True \
    --topline_path '/home/ubuntu/master_thesis/covering_lens/TypiClust/topline_csv/zero_shot_model_th_0.3.csv' 


# ProbCover with CLIP Balancing
python train_al.py --cfg ../configs/pascalvoc/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 4 \
    --delta 0.6 \
    --model_features clip \
    --method clip_selection_balanced_classes \
    --seed 1 \
    --const_threshold 1.2

# ProbCover with TOP Line Balancing
python train_al.py --cfg ../configs/pascalvoc/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 4 \
    --delta 0.6 \
    --model_features clip \
    --method clip_selection_balanced_classes \
    --seed 1 \
    --const_threshold 1.2 \
    --top_line True \
    --topline_path '/home/ubuntu/master_thesis/covering_lens/TypiClust/topline_csv/df_top_line.csv' 



'========================= MS COCO ========================='

# ProbCover Only 
python train_al.py --cfg ../configs/mscoco/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 1000 \
    --delta 0.56 \
    --model_features clip \
    --method probcover 

# Random
python train_al.py --cfg ../configs/mscoco/al/RESNET50.yaml \
    --al random \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 1000 \
    --seed 1


# ProbCover with CLIP 
python train_al.py --cfg ../configs/mscoco/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 4 \
    --delta 0.6 \
    --model_features clip \
    --method clip_selection_max_object_v2 \
    --number_of_samples 20 \
    --seed 1 \
    --const_threshold 0.4 \
    --number_of_smallest_values_to_consider 18


# ProbCover with OWL-ViT
python train_al.py --cfg ../configs/mscoco/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 4 \
    --delta 0.56 \
    --model_features clip \
    --method clip_selection_max_object \
    --number_of_samples 20 \
    --topline_count_method per_object_weighted \
    --seed 1 \
    --top_line True \
    --topline_path '/home/ubuntu/master_thesis/covering_lens/TypiClust/topline_csv/zero_shot_mscoco_02.csv' 



# ProbCover with CLIP Balancing
python train_al.py --cfg ../configs/mscoco/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 4 \
    --delta 0.56 \
    --model_features clip \
    --method clip_selection_balanced_classes \
    --seed 1 \
    --const_threshold 1.2

# ProbCover with TOP Line Balancing
python train_al.py --cfg ../configs/mscoco/al/RESNET50.yaml \
    --al probcover \
    --exp-name auto \
    --initial_size 0 \
    --budget 1000 \
    --num_cycles 4 \
    --delta 0.56 \
    --model_features clip \
    --method clip_selection_balanced_classes \
    --seed 1 \
    --const_threshold 1.2 \
    --top_line True \
    --topline_path '/home/ubuntu/master_thesis/covering_lens/TypiClust/topline_csv/df_top_line_mscoco.csv' 




