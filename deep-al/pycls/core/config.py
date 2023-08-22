# This file is modified from code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564

# ----------------------------------------------------------
# This file is modified from official pycls repository to adapt in AL settings.

"""Configuration file (powered by YACS)."""

import os
import json
from yacs.config import CfgNode as CN
from tqdm import tqdm

# Global config object
_C = CN()

# Example usage:
#   from core.config import cfg
cfg = _C

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Number of GPUs to use (applies to both training and testing)
_C.NUM_GPUS = 1
# Output directory (will be created at the projec root)
_C.OUT_DIR = 'output'
# Experiment directory
_C.EXP_DIR = ''
# Episode directory
_C.EPISODE_DIR = ''
# Config destination (in OUT_DIR)
_C.CFG_DEST = 'config.yaml'
# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
_C.RNG_SEED = None
# Folder name where best model logs etc are saved. "auto" creates a timestamp based folder 
_C.EXP_NAME = 'auto' 
# Which GPU to run on
_C.GPU_ID = 0
# Log destination ('stdout' or 'file')
_C.LOG_DEST = 'file'
# Log period in iters
_C.LOG_PERIOD = 10


#------------------------------------------------------------------------------#
# VAAL Options (Taken from https://arxiv.org/abs/1904.00370)
#------------------------------------------------------------------------------#
_C.VAAL = CN()
_C.VAAL.TRAIN_VAAL = False
_C.VAAL.Z_DIM = 32
_C.VAAL.VAE_BS = 64
_C.VAAL.VAE_EPOCHS = 100
_C.VAAL.VAE_LR = 5e-4
_C.VAAL.DISC_LR = 5e-4
_C.VAAL.BETA = 1.0
_C.VAAL.ADVERSARY_PARAM = 1.0
_C.VAAL.IM_SIZE = 32

#------------------------------------------------------------------------------#
# Ensemble Options
#------------------------------------------------------------------------------#
_C.ENSEMBLE = CN()
_C.ENSEMBLE.NUM_MODELS = 3
_C.ENSEMBLE.SAME_MODEL = True
_C.ENSEMBLE.MODEL_TYPE = ['resnet18']

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
# Model type. 
# Choose from vgg style ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',]
# or from resnet style ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 
# 'wide_resnet50_2', 'wide_resnet101_2']
_C.MODEL.TYPE = '' 
# Number of classes
_C.MODEL.NUM_CLASSES = 10
# Loss function (see pycls/models/loss.py for options)
_C.MODEL.LOSS_FUN = 'cross_entropy'


# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CN()
# BN epsilon
_C.BN.EPS = 1e-5
# BN momentum (BN momentum in PyTorch = 1 - BN momentum in Caffe2)
_C.BN.MOM = 0.1
# Precise BN stats
_C.BN.USE_PRECISE_STATS = False
_C.BN.NUM_SAMPLES_PRECISE = 1024
# Initialize the gamma of the final BN of each block to zero
_C.BN.ZERO_INIT_FINAL_GAMMA = False

# Use a different weight decay for BN layers
_C.BN.USE_CUSTOM_WEIGHT_DECAY = False
_C.BN.CUSTOM_WEIGHT_DECAY = 0.0


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.OPTIM = CN()
_C.OPTIM.TYPE='sgd'
# Learning rate ranges from BASE_LR to MIN_LR*BASE_LR according to the LR_POLICY
_C.OPTIM.BASE_LR = 0.1
_C.OPTIM.MIN_LR = 0.0
# Learning rate policy select from {'cos', 'exp', 'steps'}
_C.OPTIM.LR_POLICY = 'cos'
# Exponential decay factor
_C.OPTIM.GAMMA = 0.1
# Steps for 'steps' policy (in epochs)
_C.OPTIM.STEPS = []
# Learning rate multiplier for 'steps' policy
_C.OPTIM.LR_MULT = 0.1
# Maximal number of epochs
_C.OPTIM.MAX_EPOCH = 200
# Momentum
_C.OPTIM.MOMENTUM = 0.9
# Momentum dampening
_C.OPTIM.DAMPENING = 0.0
# Nesterov momentum
_C.OPTIM.NESTEROV = False
# L2 regularization
_C.OPTIM.WEIGHT_DECAY = 5e-4
# Start the warm up from OPTIM.BASE_LR * OPTIM.WARMUP_FACTOR
_C.OPTIM.WARMUP_FACTOR = 0.1
# Gradually warm up the OPTIM.BASE_LR over this number of epochs
_C.OPTIM.WARMUP_EPOCHS = 0

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
# Dataset and split
_C.TRAIN.DATASET = ''
_C.TRAIN.SPLIT = 'train'
_C.TRAIN.IMBALANCED = False
# Total mini-batch size
_C.TRAIN.BATCH_SIZE = 128
# Image size
_C.TRAIN.IM_SIZE = 224
_C.TRAIN.IM_CHANNELS = 3
# Evaluate model on test data every eval period epochs
_C.TRAIN.EVAL_PERIOD = 1
# Save model checkpoint every checkpoint period epochs
_C.TRAIN.CHECKPOINT_PERIOD = 1
# Resume training from the latest checkpoint in the output directory
_C.TRAIN.AUTO_RESUME = False
# Weights to start training from
_C.TRAIN.WEIGHTS = ''
_C.TRAIN.TRANSFER_EXP = False

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Dataset and split
_C.TEST.DATASET = ''
_C.TEST.SPLIT = 'val'
# Total mini-batch size
_C.TEST.BATCH_SIZE = 200
# Image size
_C.TEST.IM_SIZE = 256
# Weights to use for testing
_C.TEST.WEIGHTS = ''
# Saved model to use for testing
_C.TEST.MODEL_PATH = ''

# ---------------------------------------------------------------------------- #
# Random Augmentation options
# ---------------------------------------------------------------------------- #
_C.RANDAUG = CN()
_C.RANDAUG.ACTIVATE = False
_C.RANDAUG.N = 1
_C.RANDAUG.M = 5

# #-------------------------------------------------------------------------------#
# #  ACTIVE LEARNING options
# #-------------------------------------------------------------------------------#
_C.ACTIVE_LEARNING = CN()
_C.ACTIVE_LEARNING.SAMPLING_FN = 'random' # 'entropy', 'margin', 'vaal', 'coreset', 'ensemble_var_R'
_C.ACTIVE_LEARNING.ACTIVATE = False
_C.ACTIVE_LEARNING.LSET_PATH = ''
_C.ACTIVE_LEARNING.USET_PATH = ''
_C.ACTIVE_LEARNING.VALSET_PATH = ''
_C.ACTIVE_LEARNING.MODEL_LOAD_DIR = ''
_C.ACTIVE_LEARNING.MODEL_SAVE_DIR = ''
_C.ACTIVE_LEARNING.DATA_SPLIT = 0
_C.ACTIVE_LEARNING.BUDGET_SIZE = 5000 # 10% of initial lSet
_C.ACTIVE_LEARNING.N_BINS = 500 # Used by UC_uniform
_C.ACTIVE_LEARNING.DROPOUT_ITERATIONS = 25 # Used by DBAL and BALD
_C.ACTIVE_LEARNING.INIT_L_RATIO = 0.1 # Initial labeled pool ration
_C.ACTIVE_LEARNING.MAX_ITER = 5 # Max AL iterations
_C.ACTIVE_LEARNING.FINE_TUNE = True # continue after AL from existing model or from scratch

# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()
# Number of data loader workers per training process
_C.DATA_LOADER.NUM_WORKERS = 4
# Load data to pinned host memory
_C.DATA_LOADER.PIN_MEMORY = True

# ---------------------------------------------------------------------------- #
# CUDNN options
# ---------------------------------------------------------------------------- #
_C.CUDNN = CN()
# Perform benchmarking to select the fastest CUDNN algorithms to use
# Note that this may increase the memory usage and will likely not result
# in overall speedups when variable size inputs are used (e.g. COCO training)
_C.CUDNN.BENCHMARK = False

# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.NAME = None
# For Tiny ImageNet dataset, ROOT_DIR must be set to the dataset folder ("data/tiny-imagenet-200/"). For others, the outder "data" folder where all datasets can be stored is expected.
_C.DATASET.ROOT_DIR = None
# Specifies the proportion of data in train set that should be considered as the validation data
_C.DATASET.VAL_RATIO = 0.1
# Data augmentation methods - 'simclr', 'randaug', 'hflip'
_C.DATASET.AUG_METHOD = 'hflip' 
# Accepted Datasets
_C.DATASET.ACCEPTED = ['MNIST','SVHN','CIFAR10','CIFAR100','TINYIMAGENET', 'IMBALANCED_CIFAR10', 'IMBALANCED_CIFAR100', 'IMAGENET50', 'IMAGENET100', 'IMAGENET200', 'PASCALVOC', 'MSCOCO']

def assert_cfg():
    """Checks config values invariants."""
    assert not _C.OPTIM.STEPS or _C.OPTIM.STEPS[0] == 0, \
        'The first lr step must start at 0'
    assert _C.TRAIN.SPLIT in ['train', 'val', 'test'], \
        'Train split \'{}\' not supported'.format(_C.TRAIN.SPLIT)
    assert _C.TRAIN.BATCH_SIZE % _C.NUM_GPUS == 0, \
        'Train mini-batch size should be a multiple of NUM_GPUS.'
    assert _C.TEST.SPLIT in ['train', 'val', 'test'], \
        'Test split \'{}\' not supported'.format(_C.TEST.SPLIT)
    assert _C.TEST.BATCH_SIZE % _C.NUM_GPUS == 0, \
        'Test mini-batch size should be a multiple of NUM_GPUS.'

    #our assertions
    if _C.ACTIVE_LEARNING.SAMPLING_FN == "uncertainty_uniform_discretize":
        assert _C.ACTIVE_LEARNING.N_BINS !=0, \
        "The number of bins used cannot be 0. Please provide a number >0 for {} sampling function"\
            .format(_C.ACTIVE_LEARNING.SAMPLING_FN)

def custom_dump_cfg(temp_cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(temp_cfg.EXP_DIR, temp_cfg.CFG_DEST)
    with open(cfg_file, 'w') as f:
        _C.dump(stream=f)


def dump_cfg(cfg):
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(cfg.EXP_DIR, cfg.CFG_DEST)
    with open(cfg_file, 'w') as f:
        cfg.dump(stream=f)

def dump_file(cfg, selected_files):
    """Dumps the file to the output directory."""

    if cfg.DATASET.NAME == "MSCOCO":
        
        if cfg.MODEL_FEATURES == 'clip':
            selected_filenames = "train_active_model_{}_budget_{}_delta_{}_method_{}_normalize_{}_top_line_{}_th_{}_N_{}.txt".format(cfg.MODEL_FEATURES, cfg.ACTIVE_LEARNING.BUDGET_SIZE, cfg.ACTIVE_LEARNING.DELTA, cfg.METHOD, cfg.NORMALIZE, cfg.TOP_LINE, cfg.CONST_THRESHOLD, cfg.NUMBER_OF_SAMPLES)
        else: 
            selected_filenames= "train_active_model_{}_budget_{}_delta_{}.txt".format(cfg.MODEL_FEATURES, cfg.ACTIVE_LEARNING.BUDGET_SIZE, cfg.ACTIVE_LEARNING.DELTA)
        
        file = os.path.join(cfg.EXP_DIR, selected_filenames)

        with open(file, 'w') as f:
            for selected_image_file_name in selected_files:
                f.write(selected_image_file_name + '\n')

        convert_to_json(cfg,  selected_files)
    else:
        # reading the filenames.txt file
        filename_2012 = []
        with open(os.path.join(cfg.DATASET.ROOT_DIR,'VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'), 'r') as f:
            lines = f.readlines()
        for line in lines:
            filename_2012.append(line.strip())

        if cfg.MODEL_FEATURES == 'clip':
            selected_filenames_2007 = "train_active_2007_model_{}_budget_{}_delta_{}_method_{}_normalize_{}_top_line_{}_th_{}_N_{}.txt".format(cfg.MODEL_FEATURES, cfg.ACTIVE_LEARNING.BUDGET_SIZE, cfg.ACTIVE_LEARNING.DELTA, cfg.METHOD, cfg.NORMALIZE, cfg.TOP_LINE, cfg.CONST_THRESHOLD, cfg.NUMBER_OF_SAMPLES)
            selected_filenames_2012 = "train_active_2012_model_{}_budget_{}_delta_{}_method_{}_normalize_{}_top_line_{}_th_{}_N_{}.txt".format(cfg.MODEL_FEATURES, cfg.ACTIVE_LEARNING.BUDGET_SIZE, cfg.ACTIVE_LEARNING.DELTA, cfg.METHOD, cfg.NORMALIZE, cfg.TOP_LINE, cfg.CONST_THRESHOLD, cfg.NUMBER_OF_SAMPLES)
        else: 
            selected_filenames_2007 = "train_active_2007_model_{}_budget_{}_delta_{}.txt".format(cfg.MODEL_FEATURES, cfg.ACTIVE_LEARNING.BUDGET_SIZE, cfg.ACTIVE_LEARNING.DELTA)
            selected_filenames_2012 = "train_active_2012_model_{}_budget_{}_delta_{}.txt".format(cfg.MODEL_FEATURES, cfg.ACTIVE_LEARNING.BUDGET_SIZE, cfg.ACTIVE_LEARNING.DELTA)
        file_1= os.path.join(cfg.EXP_DIR, selected_filenames_2007)
        file_2= os.path.join(cfg.EXP_DIR, selected_filenames_2012)
        with open(file_1, 'w') as file_1, open(file_2, 'w') as file_2:
            for selected_image_file_name in selected_files:
                selected_image_file_name = selected_image_file_name.strip(".jpg")
                if selected_image_file_name in filename_2012:
                    file_2.write(selected_image_file_name + '\n')
                else:
                    file_1.write(selected_image_file_name + '\n')

def convert_to_json(cfg,  selected_files):
    """Converts a COCO format annotation file to a new file containing only the information for selected images."""

    if cfg.MODEL_FEATURES == 'clip':
        json_filename= "instances_train2017_selected_model_{}_budget_{}_delta_{}_method_{}_normalize_{}_top_line_{}_th_{}_N_{}.json".format(cfg.MODEL_FEATURES, cfg.ACTIVE_LEARNING.BUDGET_SIZE, cfg.ACTIVE_LEARNING.DELTA, cfg.METHOD, cfg.NORMALIZE, cfg.TOP_LINE, cfg.CONST_THRESHOLD, cfg.NUMBER_OF_SAMPLES)
    else: 
        json_filename= "instances_train2017_selected_model_{}_budget_{}_delta_{}.json".format(cfg.MODEL_FEATURES, cfg.ACTIVE_LEARNING.BUDGET_SIZE, cfg.ACTIVE_LEARNING.DELTA)
    json_file = os.path.join(cfg.EXP_DIR, json_filename)
    # Load initial annotations_train2017.json file
    with open(os.path.join(cfg.DATASET.ROOT_DIR,'coco/annotations/instances_train2017.json'), 'r') as f:
        coco_json = json.load(f)

    # Create new dictionary with only the information for the selected images
    selected_coco_json = {}
    selected_coco_json['info'] = coco_json['info']
    selected_coco_json['licenses'] = coco_json['licenses']
    selected_coco_json['images'] = []
    selected_coco_json['annotations'] = []
    selected_coco_json['categories'] = coco_json['categories']

    for image in tqdm(coco_json['images']):
        if image['file_name'] in  selected_files:
            selected_coco_json['images'].append(image)
            for annotation in coco_json['annotations']:
                if annotation['image_id'] == image['id']:
                    selected_coco_json['annotations'].append(annotation)

    # Write the new COCO annotation file
    with open(json_file, 'w') as f:
        json.dump(selected_coco_json, f)


def load_cfg(out_dir, cfg_dest='config.yaml'):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    _C.merge_from_file(cfg_file)