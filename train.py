import sys
import os
import math
import time
import numpy as np
import logging
from collections import defaultdict
import pprint
import shutil
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import torch
import torch.nn as nn
import torch.optim
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from tqdm.contrib import tzip, tenumerate
# from skimage.metrics import structural_similarity as compare_ssim
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sklearn import metrics


from config import config, update_config, get_cfg_defaults
from models.models import Generator
# from data.dataset import trainloader, valloader
from data.dataset_CelebA import CelebA
# from data.dataset_CASIA import CASIA
# from data.dataset_inpainting import Inpainting
from common.utils import AverageMeter
import common.utils as utils
from common.metrics import calculate_pixel_score,calculate_img_score_np
from common.loss import DiceLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('-c', '--cfg',
                        help='experiment configure file name',
                        # required=True,
                        default='/home/huangjinkun/mycode/Detection/experiment/config.yaml',
                        type=str)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    # update_config(config, args)

    return args

def main(args):
    config = get_cfg_defaults()
    update_config(config, args)
    # config = utils.get_config(config_path)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    net = Generator()
    net = net.to(device)
    # # pre-train
    # state_dicts = torch.load('/data/hjk/result/Detection/runs/model_aotgan_dct_h--2022.08.12--14-30-36/model_best.pt')
    # net.load_state_dict(state_dicts['net'])

    optim = torch.optim.Adam(net.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY)

    if config.TRAIN.RESUME is True:
        # state_dicts = torch.load(os.path.join(config.CONTINUE_PATH, 'model.pt'))
        # net.load_state_dict(state_dicts['net'])
        # optim.load_state_dict(state_dicts['opt'])
        model_path = config.TRAIN.CONTINUE_PATH
        # utils.load_model(os.path.join(model_path, 'model.pt'), net, optim)
        state_dicts = torch.load(os.path.join(model_path, 'model.pt'))
        net.load_state_dict(state_dicts['net'])
        optim.load_state_dict(state_dicts['opt'])
        begin_epoch = state_dicts['epoch'] + 1
        best_loss = state_dicts['best_loss']
        best_p_f1 = state_dicts['best_p_f1']

    else:
        model_path = os.path.join(config.OUTPUT_DIR, config.MODEL.NAME + '--' + time.strftime("%Y.%m.%d--%H-%M-%S"))
        os.makedirs(model_path)
        os.makedirs(os.path.join(model_path, 'images'))
        # 保存配置文件和模型代码
        shutil.copy2(args.cfg, os.path.join(model_path, 'config.yaml'))
        shutil.copy2('/home/huangjinkun/mycode/Detection/models/models.py', os.path.join(model_path, 'models.py'))

        begin_epoch = config.TRAIN.BEGIN_EPOCH
        best_loss = float('inf')
        best_p_f1 = 0.

    net = nn.DataParallel(net, device_ids=list(config.GPUS))

    # log
    utils.setup_logger('train', model_path, 'train', level=logging.INFO, screen=True, tofile=True)
    logger_train = logging.getLogger('train')

    # logger_train.info(pprint.pformat(vars(config)))
    logger_train.info(pprint.pformat(args))
    logger_train.info(config)
    logger_train.info(net)

    writer = SummaryWriter(log_dir=os.path.join(model_path, 'tb-logs'))

    # loss function
    bce_loss = nn.BCEWithLogitsLoss().to(device)
    mse_loss = nn.MSELoss().to(device)
    ce_loss = nn.CrossEntropyLoss().to(device)
    dice_loss = DiceLoss().to(device)

    # load dataset
    transform_train = T.Compose([
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        T.Resize(config.TRAIN.IMAGE_SIZE),
        T.ToTensor(),
    ])

    transform_val = T.Compose([
        T.Resize(config.TEST.IMAGE_SIZE),
        T.ToTensor()
    ])

    train_dataset = CelebA(config.DATASET.ROOT, config.DATASET.DATA_LIST, config.DATASET.ATTR_PATH, mode='train', transforms=transform_train, origin=True)
    # train_dataset = CASIA(config.DATASET.ROOT, mode='train', transforms=transform_train)
    # train_dataset = Inpainting(config.DATASET.ROOT, mode='train', transforms=transform_train)

    trainloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        num_workers=config.WORKERS
    )

    val_dataset = CelebA(config.DATASET.ROOT, config.DATASET.DATA_LIST, config.DATASET.ATTR_PATH, mode='val', transforms=transform_val, origin=True)
    # val_dataset = CASIA(config.DATASET.ROOT, mode='val', transforms=transform_val)
    # val_dataset = Inpainting(config.DATASET.ROOT, mode='val', transforms=transform_val)

    valloader = DataLoader(
        val_dataset,
        batch_size=config.TEST.BATCH_SIZE,
        shuffle=False,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        num_workers=config.WORKERS
    )

    # min_loss = float('inf')
    # best_p_f1 = 0.
    for epoch in range(begin_epoch, config.TRAIN.EPOCHES):
        #-------------------------train-----------------------
        net.train()

        train_loss_meter = defaultdict(AverageMeter)

        for data in tqdm(trainloader, dynamic_ncols=True):
            # imgs, masks, labels = [], [], []
            # for (img, mask, label) in data:
            #     imgs.append(img)
            #     masks.append(mask)
            #     labels.append(label)

            # imgs = torch.concat(imgs, dim=0)
            # masks = torch.concat(masks, dim=0)
            # labels = torch.concat(labels, dim=0)
            imgs, masks, labels = data
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            reveal_masks, pred = net(imgs) 

            #-------------------------train gen-----------------------
            optim.zero_grad()

            g_loss_mask = bce_loss(reveal_masks, masks)
            # g_loss_mask = dice_loss(reveal_masks, masks)
            # g_loss_pred = bce_loss(attrs, labels)
            # g_loss_pred = mse_loss(attrs, labels)
            g_loss_pred = ce_loss(pred, labels)
            
            g_loss = g_loss_mask + config.LOSS.LAMBDA_PRED * g_loss_pred

            g_loss.backward()
            optim.step()
        
            train_loss_meter['loss'].update(g_loss.item())
            train_loss_meter['loss_mask'].update(g_loss_mask.item())
            train_loss_meter['loss_pred'].update(g_loss_pred.item())

        # tensorboard
        for k, v in train_loss_meter.items():
            writer.add_scalar(f'train/{k}', v.avg, epoch)

        msg = f"Train epoch {epoch}: "
        for k, v in train_loss_meter.items():
            msg += f"{k}: {v.avg:.6f} |"
        msg += f"lr: {optim.param_groups[0]['lr']} |"
        logger_train.info(msg)


        #-------------------------val-----------------------
        net.eval()

        val_loss_meter = defaultdict(AverageMeter)
        img_list, mask_list, pre_list = [], [], []
        # for image level metrics
        img_pred_list, img_socre_list, img_label_list = [], [], []
        with torch.no_grad():
            for i, data in tenumerate(valloader, dynamic_ncols=True):
                # imgs, masks, labels = [], [], []
                # for (img, mask, label) in data:
                #     imgs.append(img)
                #     masks.append(mask)
                #     labels.append(label)

                # imgs = torch.concat(imgs, dim=0)
                # masks = torch.concat(masks, dim=0)
                # labels = torch.concat(labels, dim=0)
                imgs, masks, labels = data
                imgs = imgs.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                reveal_masks, pred = net(imgs)

                g_loss_mask = bce_loss(reveal_masks, masks)
                # g_loss_mask = dice_loss(reveal_masks, masks)
                # g_loss_pred = bce_loss(attrs, labels)
                # g_loss_pred = mse_loss(attrs, labels)
                g_loss_pred = ce_loss(pred, labels)

                g_loss = g_loss_mask + config.LOSS.LAMBDA_PRED * g_loss_pred

                reveal_masks = torch.sigmoid(reveal_masks)

                p_f1 = torch.zeros(config.TEST.BATCH_SIZE)
                mIoU = torch.zeros(config.TEST.BATCH_SIZE)
                mcc = torch.zeros(config.TEST.BATCH_SIZE)
                for j in range(config.TEST.BATCH_SIZE):
                    reveal_masks_ = reveal_masks[j].squeeze(0)
                    reveal_masks_ = (reveal_masks_ >= 0.5).float()
                    masks_ = masks[j].squeeze(0)
                    p_f1[j], _, _, mIoU[j], mcc[j] = calculate_pixel_score(reveal_masks_.flatten(), masks_.flatten())
                p_f1 = p_f1.mean()
                mIoU = mIoU.mean()
                mcc = mcc.mean()

                val_loss_meter['loss'].update(g_loss.item())
                val_loss_meter['loss_mask'].update(g_loss_mask.item())
                val_loss_meter['loss_pred'].update(g_loss_pred.item())
                val_loss_meter['p_f1'].update(p_f1.item())
                val_loss_meter['mIoU'].update(mIoU.item())
                val_loss_meter['mcc'].update(mcc.item())

                # image_level
                img_pred_list += torch.max(torch.softmax(pred, dim=1), dim=1)[1].tolist()
                img_socre_list += torch.softmax(pred, dim=1)[:, 1].tolist()
                img_label_list += labels.tolist()

                # batch size 需要能被整除，不然保存的图片会有问题
                if i % (500 // config.TEST.BATCH_SIZE)  == 0:
                    # reveal_masks = torch.sigmoid(reveal_masks)
                    img_list.append(imgs)
                    mask_list.append(masks)
                    pre_list.append(reveal_masks)

            try:
                img_auc = metrics.roc_auc_score(img_label_list, img_socre_list)
            except:
                img_auc = np.zeros(1)
            _, _, _, img_f1, _, _, _, _ = calculate_img_score_np(img_pred_list, img_label_list)
            val_loss_meter['img_f1'].update(img_f1.item())
            val_loss_meter['img_auc'].update(img_auc.item())

            utils.save_images(torch.concat(img_list, dim=0), torch.concat(mask_list, dim=0), torch.concat(pre_list, dim=0), config.TEST.BATCH_SIZE, epoch, os.path.join(model_path, 'images'), 256)
            
            # tensorboard
            for k, v in val_loss_meter.items():
                writer.add_scalar(f'val/{k}', v.avg, epoch)

            msg = f"Val epoch {epoch}: "
            for k, v in val_loss_meter.items():
                msg += f"{k}: {v.avg:.6f} |"
            logger_train.info(msg)

        # save checkpoint
        torch.save({'net': net.module.state_dict(),
                    'opt': optim.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'best_p_f1': best_p_f1}, os.path.join(model_path, 'model.pt'))
        
        # save best checkpoint
        # if val_loss_meter['loss'].avg < min_loss:
        if val_loss_meter['loss'].avg < best_loss:
            torch.save({'net': net.module.state_dict(),
                        'opt': optim.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'best_p_f1': best_p_f1}, os.path.join(model_path, 'model_best_loss.pt'))
            # min_loss = val_loss_meter['loss'].avg
            best_loss = val_loss_meter['loss'].avg
            logger_train.info('Save best loss checkpoint.')
        if val_loss_meter['p_f1'].avg > best_p_f1:
            torch.save({'net': net.module.state_dict(),
                        'opt': optim.state_dict(),
                        'epoch': epoch,
                        'best_loss': best_loss,
                        'best_p_f1': best_p_f1}, os.path.join(model_path, 'model_best_f1.pt'))
            # min_loss = val_loss_meter['loss'].avg
            best_p_f1 = val_loss_meter['p_f1'].avg
            logger_train.info('Save best f1 checkpoint.')
        
    writer.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)