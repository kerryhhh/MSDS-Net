from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import argparse

from yacs.config import CfgNode as CN


_C = CN()

_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.GPUS = (0,)
_C.WORKERS = 4
_C.PRINT_FREQ = 20
_C.AUTO_RESUME = False
_C.PIN_MEMORY = True
_C.RANK = 0

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'model'
_C.MODEL.DESCRIPTION = ''
_C.MODEL.EXTRA = CN(new_allowed=True)

_C.LOSS = CN(new_allowed=True)
# _C.LOSS.LAMBDA_ATTR = 1
_C.LOSS.LAMBDA_PRED = 1

# DATASET related params
_C.DATASET = CN()
_C.DATASET.ROOT = ''
_C.DATASET.DATASET = 'celeba'
_C.DATASET.NUM_ATTR = 4
# _C.DATASET.TRAIN_LIST = ''
# _C.DATASET.VAL_LIST = ''
# _C.DATASET.TEST_LIST = ''
_C.DATASET.DATA_LIST = ''
_C.DATASET.ATTR_PATH = ''

# training
_C.TRAIN = CN()

_C.TRAIN.IMAGE_SIZE = [256, 256]

_C.TRAIN.LR = 1e-4
_C.TRAIN.WEIGHT_DECAY = 0.

_C.TRAIN.OPTIMIZER = 'adam'

_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.EPOCHES = 150

_C.TRAIN.RESUME = False
_C.TRAIN.CONTINUE_PATH = ''

_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.SHUFFLE = True

# testing
_C.TEST = CN()

_C.TEST.IMAGE_SIZE = [256, 256]

_C.TEST.BATCH_SIZE = 10
# only testing some samples
_C.TEST.NUM_SAMPLES = 0

_C.TEST.MODEL_FILE = ''

# debug
_C.DEBUG = CN()
_C.DEBUG.DEBUG = False
_C.DEBUG.SAVE_BATCH_IMAGES_GT = False
_C.DEBUG.SAVE_BATCH_IMAGES_PRED = False
_C.DEBUG.SAVE_HEATMAPS_GT = False
_C.DEBUG.SAVE_HEATMAPS_PRED = False


def get_cfg_defaults():
    return _C.clone()


def update_config(cfg, args):
    cfg.defrost()
    
    if isinstance(args, argparse.Namespace):
        cfg.merge_from_file(args.cfg)
        if args.opts:
            cfg.merge_from_list(args.opts)
    else:
        cfg.merge_from_file(args)

    cfg.freeze()


if __name__ == '__main__':
    import sys
    # with open(sys.argv[1], 'w') as f:
    #     print(_C, file=f)
    print(_C.clone())
