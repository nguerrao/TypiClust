# Data selection for Object Detection using Vision Language Models

The project consists of the following steps:

1. **Features extraction**: Image embeddings are extracted using CLIP model. For a detailed walkthrough, refer to the[Jupyter notebook](deep-al/features_extraction/CLIP/CLIP_embeddings_pascalvoc.ipynb) in the project.
2. **Labels extraction**: Labels for each image are predicted using the OWL-ViT model. For a detailed walkthrough, refer to the[Jupyter notebook](deep-al/features_extraction/OWL-ViT/OWL-ViT.ipynb) in the project.
3. **Selection**: 
    1. **Using CLIP**: Selection Using ProbCover algorithm with CLIP supervision. For more details about this method, see the [PDF report](https://drive.google.com/file/d/1lwNdwZGJDSWM0PaDQKE1tZvTP5_a1DOG/view) provided on Google Drive.

    This can be done by running (for PASCAL VOC by default):
    ```sh
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
    ```
    
    2. **Using OWL-ViT**: Selection Using ProbCover algorithm with OWL-ViT supervision. For more details about this method, see the [PDF report](https://drive.google.com/file/d/1lwNdwZGJDSWM0PaDQKE1tZvTP5_a1DOG/view) provided on Google Drive.
    This can be done by running (for PASCAL VOC by default):
    ```sh
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
    ```
To learn more about the final optimal parameters and how we chose them, please refer to the [PDF report](https://drive.google.com/file/d/1lwNdwZGJDSWM0PaDQKE1tZvTP5_a1DOG/view) provided on Google Drive.

You can also find additional command-line instructions in the [bash file](deep-al/tools/run_AL.sh).
