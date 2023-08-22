import os
import sys
from datetime import datetime
import argparse
import numpy as np

import torch
from copy import deepcopy

# local

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(os.path.abspath('..'))

from pycls.al.ActiveLearning import ActiveLearning
import pycls.core.builders as model_builder
from pycls.core.config import cfg, dump_cfg, dump_file
import pycls.core.losses as losses
import pycls.core.optimizer as optim
from pycls.datasets.data import Data
import pycls.utils.checkpoint as cu
import pycls.utils.logging as lu
import pycls.utils.metrics as mu
import pycls.utils.net as nu
from pycls.utils.meters import TestMeter
from pycls.utils.meters import TrainMeter
from pycls.utils.meters import ValMeter

logger = lu.get_logger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argparser():
    parser = argparse.ArgumentParser(description='Active Learning - Image Classification')
    parser.add_argument('--cfg', dest='cfg_file', help='Config file', required=True, type=str)
    parser.add_argument('--exp-name', help='Experiment Name', required=True, type=str)
    parser.add_argument('--al', help='AL Method', required=True, type=str)
    parser.add_argument('--budget', help='Budget Per Round', required=True, type=int)
    parser.add_argument('--initial_size', help='Size of the initial random labeled set', default=0, type=int)
    parser.add_argument('--seed', help='Random seed', default=1, type=int)
    parser.add_argument('--model_features', help='Model for features extraction', default='clip', type=str)
    parser.add_argument('--method', help='Method used to do the selection with clip and ProbCover (e.g. probcover, clip_selection_balanced_classes, clip_selection_max_object, clip_selection_max_object_v2, top_line(used ground truth instead of clip predictions)', default='probcover', type=str)
    parser.add_argument('--finetune', help='Whether to continue with existing model between rounds', type=str2bool, default=False)
    parser.add_argument('--linear_from_features', help='Whether to use a linear layer from self-supervised features', action='store_true')
    parser.add_argument('--delta', help='Relevant only for ProbCover', default=0.6, type=float)
    parser.add_argument('--top_line', help='Relevant only to get the top line with ProbCover', default=False, type=bool)
    parser.add_argument('--const_threshold', help='Relevant only to use CLIP selection with ProbCover to balance the dataset', default=1, type=float)
    parser.add_argument('--const_threshold_mean', help='Relevant only to use CLIP selection with ProbCover to balance the dataset', default=1, type=float)
    parser.add_argument('--alpha', help='Relevant only to use CLIP new distance with ProbCover', default=1, type=float)
    parser.add_argument('--number_of_samples', help='Relevant only to set the number of the selected samples with ProbCover for each iteration', default=10, type=int)
    parser.add_argument('--number_of_smallest_values_to_consider', help='Relevant only to set the number of smallest values to consider for average for clip selection count method with ProbCover', default=3, type=int)
    parser.add_argument('--normalize', help='Relevant only to normalize the cosine similarities between images and text features when using clip selection method', default=False, type=bool)
    parser.add_argument('--text_embedding_pascalvoc', help='Relevant only to use clip methods with probcover', default='/home/ubuntu/master_thesis/covering_lens/TypiClust/scan/results/pascalvoc/pretext/text_embedding_pascalvoc_classes_human_RN50.npy', type=str)
    parser.add_argument('--text_embedding_coco', help='Relevant only to use clip methods with probcover', default='/home/ubuntu/master_thesis/covering_lens/TypiClust/scan/results/mscoco/pretext/text_embedding_mscoco_classes_human_RN50.npy', type=str)
    return parser


def main(cfg):

    # Setting up GPU args
    use_cuda = (cfg.NUM_GPUS > 0) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': cfg.DATA_LOADER.NUM_WORKERS, 'pin_memory': cfg.DATA_LOADER.PIN_MEMORY} if use_cuda else {}

    # Auto assign a RNG_SEED when not supplied a value
    if cfg.RNG_SEED is None:
        cfg.RNG_SEED = np.random.randint(100)

    # Using specific GPU
    # os.environ['NVIDIA_VISIBLE_DEVICES'] = str(cfg.GPU_ID)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # print("Using GPU : {}.\n".format(cfg.GPU_ID))

    # Getting the output directory ready (default is "/output")
    cfg.OUT_DIR = os.path.join(os.path.abspath('../..'), cfg.OUT_DIR)
    if not os.path.exists(cfg.OUT_DIR):
        os.mkdir(cfg.OUT_DIR)
    # Create "DATASET/MODEL TYPE" specific directory
    dataset_out_dir = os.path.join(cfg.OUT_DIR, cfg.DATASET.NAME, cfg.MODEL.TYPE)
    if not os.path.exists(dataset_out_dir):
        os.makedirs(dataset_out_dir)
    # Creating the experiment directory inside the dataset specific directory 
    # all logs, labeled, unlabeled, validation sets are stroed here 
    # E.g., output/CIFAR10/resnet18/{timestamp or cfg.EXP_NAME based on arguments passed}
    if cfg.EXP_NAME == 'auto':
        now = datetime.now()
        exp_dir = f'{now.year}_{now.month}_{now.day}_{now.hour:02}{now.minute:02}{now.second:02}_{now.microsecond}'
    else:
        exp_dir = cfg.EXP_NAME

    exp_dir = os.path.join(dataset_out_dir, exp_dir)
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print("Experiment Directory is {}.\n".format(exp_dir))
    else:
        print("Experiment Directory Already Exists: {}. Reusing it may lead to loss of old logs in the directory.\n".format(exp_dir))
    cfg.EXP_DIR = exp_dir

    # Save the config file in EXP_DIR
    dump_cfg(cfg)

    # Setup Logger
    lu.setup_logging(cfg)

    # Dataset preparing steps
    print("\n======== PREPARING DATA AND MODEL ========\n")
    cfg.DATASET.ROOT_DIR = os.path.join(os.path.abspath('../..'), cfg.DATASET.ROOT_DIR)
    data_obj = Data(cfg)
    # Print the directory path
    print("Directory path:", cfg.DATASET.ROOT_DIR)
    train_data, train_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=True, isDownload=True)
    test_data, test_size = data_obj.getDataset(save_dir=cfg.DATASET.ROOT_DIR, isTrain=False, isDownload=True)
    cfg.ACTIVE_LEARNING.INIT_L_RATIO = args.initial_size / train_size
    print("\nDataset {} Loaded Sucessfully.\nTotal Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))
    logger.info("Dataset {} Loaded Sucessfully. Total Train Size: {} and Total Test Size: {}\n".format(cfg.DATASET.NAME, train_size, test_size))

    lSet_path, uSet_path, valSet_path = data_obj.makeLUVSets(train_split_ratio=cfg.ACTIVE_LEARNING.INIT_L_RATIO, \
        val_split_ratio=cfg.DATASET.VAL_RATIO, data=train_data, seed_id=cfg.RNG_SEED, save_dir=cfg.EXP_DIR)

    cfg.ACTIVE_LEARNING.LSET_PATH = lSet_path
    cfg.ACTIVE_LEARNING.USET_PATH = uSet_path
    cfg.ACTIVE_LEARNING.VALSET_PATH = valSet_path
  
    cfg.coverage=0
    if cfg.DATASET.NAME=="MSCOCO":
        cfg.num_class=80
        class_list=[
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
                    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
        
        
        cfg.count_class=[{class_name: 0 for class_name in class_list}] # counter updated every time a new image is selected to get the number of different class objects in the labeled set 
        cfg.counts=[{i: 0 for i in range(cfg.num_class)}] # dict used to evaluate the number of object per class in the dataset (using CLIP predictions)
        cfg.threshold=[{i: 0 for i in range(cfg.num_class)}]
        
    elif cfg.DATASET.NAME=="PASCALVOC":
        cfg.num_class=20
        class_list=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
        
        cfg.count_class=[{class_name: 0 for class_name in class_list}]  # counter updated every time a new image is selected to get the number of different class objects in the labeled set 
        cfg.counts=[{i: 0 for i in range(cfg.num_class)}] # dict used to evaluate the number of object per class in the dataset (using CLIP predictions)
        cfg.threshold=[{i: 0 for i in range(cfg.num_class)}]

    lSet, uSet, valSet = data_obj.loadPartitions(lSetPath=cfg.ACTIVE_LEARNING.LSET_PATH, \
            uSetPath=cfg.ACTIVE_LEARNING.USET_PATH, valSetPath = cfg.ACTIVE_LEARNING.VALSET_PATH)
    
    model = model_builder.build_model(cfg).cuda()
    
    if len(lSet) == 0:
        print('Labeled Set is Empty - Sampling an Initial Pool')
        al_obj = ActiveLearning(data_obj, cfg)
        activeSet, new_uSet = al_obj.sample_from_uSet(model, lSet, uSet, train_data)
        print(f'Initial Pool is {activeSet}')
        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet

    print("Data Partitioning Complete. \nLabeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))
    logger.info("Labeled Set: {}, Unlabeled Set: {}, Validation Set: {}\n".format(len(lSet), len(uSet), len(valSet)))

    print("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    logger.info("AL Query Method: {}\nMax AL Episodes: {}\n".format(cfg.ACTIVE_LEARNING.SAMPLING_FN, cfg.ACTIVE_LEARNING.MAX_ITER))
    
    for cur_episode in range(0, cfg.ACTIVE_LEARNING.MAX_ITER+1):

        print("======== EPISODE {} BEGINS ========\n".format(cur_episode))
        logger.info("======== EPISODE {} BEGINS ========\n".format(cur_episode))

        # Creating output directory for the episode
        episode_dir = os.path.join(cfg.EXP_DIR, f'episode_{cur_episode}')
        if not os.path.exists(episode_dir):
            os.mkdir(episode_dir)
        cfg.EPISODE_DIR = episode_dir

        file_names=[]
        #reading the filenames.txt file 
        if cfg.DATASET.NAME=="MSCOCO":
            with open('../../scan/results/mscoco/pretext/filenames.txt', 'r') as f:
                lines = f.readlines()

        elif cfg.DATASET.NAME=="PASCALVOC":
            with open('../../scan/results/pascalvoc/pretext/filenames.txt', 'r') as f:
                lines = f.readlines()
        
        for line in lines:
            file_names.append(line.strip())
        
    
        # No need to perform active sampling in the last episode iteration
        if cur_episode == cfg.ACTIVE_LEARNING.MAX_ITER:
            # Save current lSet, uSet in the final episode directory
            data_obj.saveSet(lSet, 'lSet', cfg.EPISODE_DIR)
            data_obj.saveSet(uSet, 'uSet', cfg.EPISODE_DIR)
            
            selected_files = [file_names[i] for i in lSet]
            dump_file(cfg, selected_files)
          
            break

        # Active Sample 
        print("======== ACTIVE SAMPLING ========\n")
        logger.info("======== ACTIVE SAMPLING ========\n")
        al_obj = ActiveLearning(data_obj, cfg)
        clf_model = model_builder.build_model(cfg)
        #clf_model = cu.load_checkpoint(checkpoint_file, clf_model) #work with training 
        activeSet, new_uSet = al_obj.sample_from_uSet(clf_model, lSet, uSet, train_data)

        # Save current lSet, new_uSet and activeSet in the episode directory
        data_obj.saveSets(lSet, uSet, activeSet, cfg.EPISODE_DIR)

        # Add activeSet to lSet, save new_uSet as uSet and update dataloader for the next episode
        lSet = np.append(lSet, activeSet)
        uSet = new_uSet

        print("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        logger.info("Active Sampling Complete. After Episode {}:\nNew Labeled Set: {}, New Unlabeled Set: {}, Active Set: {}\n".format(cur_episode, len(lSet), len(uSet), len(activeSet)))
        logger.info('Counts per images is: {}'.format(cfg.counts[0]))
        logger.info('Threshold is: {}'.format(cfg.threshold[0]))
        logger.info('CLASS count: {}'.format(cfg.count_class[0]))
        logger.info('Coverage is: {}'.format(cfg.coverage))
        print("================================\n\n")
        logger.info("================================\n\n")


if __name__ == "__main__":
    args = argparser().parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_file(args.cfg_file)
    cfg.EXP_NAME = args.exp_name
    cfg.ACTIVE_LEARNING.SAMPLING_FN = args.al
    cfg.ACTIVE_LEARNING.BUDGET_SIZE = args.budget
    cfg.ACTIVE_LEARNING.DELTA = args.delta
    cfg.TOP_LINE=args.top_line
    cfg.CONST_THRESHOLD = args.const_threshold
    cfg.CONST_THRESHOLD_MEAN = args.const_threshold_mean
    cfg.ALPHA = args.alpha
    cfg.NUMBER_OF_SAMPLES = args.number_of_samples
    cfg.NUMBER_OF_SMALLEST_VALUES_TO_CONSIDER = args.number_of_smallest_values_to_consider
    cfg.RNG_SEED = args.seed
    cfg.MODEL_FEATURES = args.model_features
    cfg.METHOD = args.method
    cfg.MODEL.LINEAR_FROM_FEATURES = args.linear_from_features
    cfg.NORMALIZE = args.normalize
    cfg.TEXT_EMBEDDING_PASCALVOC = args.text_embedding_pascalvoc
    cfg.TEXT_EMBEDDING_COCO = args.text_embedding_coco

    main(cfg)
