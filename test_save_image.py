import argparse
import logging
import math
import os
import sys
import time
from collections import defaultdict
from tkinter import image_names

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torchvision
import torchvision.transforms as T
# from skimage.metrics import peak_signal_noise_ratio as compare_psnr
# from skimage.metrics import structural_similarity as compare_ssim
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from torchsummary import summary

from tqdm import tqdm
from tqdm.contrib import tenumerate, tzip

import common.utils as utils
from common.metrics import calculate_img_score_np, calculate_pixel_score, calculate_pixel_score_np, compute_eer
from common.utils import AverageMeter
from config import config, get_cfg_defaults, update_config
from data.dataset_CelebA import CelebA
# from data.dataset_CASIA import CASIA
from data.dataset_inpainting import Inpainting
from models.models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description='Test segmentation network')

    parser.add_argument('-d', '--dir',
                        help='experiment dir',
                        # required=True,
                        default='/data/hjk/result/Detection/runs/ab_model_aot_dct_h_1_m_3x4_cbam--2022.11.14--17-42-48',
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
    model_path = args.dir
    config = get_cfg_defaults()
    update_config(config, os.path.join(model_path, 'config.yaml'))

    # log
    utils.setup_logger('test', model_path, 'test', level=logging.INFO, screen=True, tofile=True)
    logger_test = logging.getLogger('test')

    net = Generator().to(device)
    logger_test.info(summary(net, (3, 256, 256)))
    optim = torch.optim.Adam(net.parameters(), lr=config.TRAIN.LR)
    # 加载模型 
    print(args.dir)
    utils.load_model(os.path.join(model_path, 'model_best_f1.pt'), net, optim)
    net = nn.DataParallel(net)

    bce_loss = nn.BCEWithLogitsLoss()
    # mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    transform_test = T.Compose([
        T.Resize(config.TEST.IMAGE_SIZE),
        T.ToTensor()
    ])

    testloader = DataLoader(
        CelebA(config.DATASET.ROOT, config.DATASET.DATA_LIST, config.DATASET.ATTR_PATH, mode='test', transforms=transform_test, origin=True),
        # CASIA('/data/hjk/CASIA/', mode='CASIA1', transforms=transform_test),
        # Inpainting(config.DATASET.ROOT, mode='test', transforms=transform_test),
        batch_size=1,
        shuffle=False,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        num_workers=config.WORKERS
    )

    with torch.no_grad():
        net.eval()

        if not os.path.exists(os.path.join(model_path, 'result')):
                os.makedirs(os.path.join(model_path, 'result'))
        test_meter = defaultdict(AverageMeter)
        # save image
        img_list, mask_list, pre_list = [], [], []
        # for image level metrics
        img_pred_list, img_socre_list, img_label_list = [], [], []
        for i, data in tenumerate(testloader, dynamic_ncols=True):
            # imgs, masks, labels = [], [], []
            # for (img, mask) in data:
            #     imgs.append(img)
            #     masks.append(mask)

            # imgs = torch.concat(imgs, dim=0)
            # masks = torch.concat(masks, dim=0)
            # labels = torch.concat(labels, dim=0)
            imgs, masks, labels, image_names = data
            imgs = imgs.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            reveal_masks, pred = net(imgs)

            g_loss_mask = bce_loss(reveal_masks, masks)
            # g_loss_pred = bce_loss(attrs, labels)
            g_loss_pred = ce_loss(pred, labels)
                
            g_loss = g_loss_mask + config.LOSS.LAMBDA_PRED * g_loss_pred

            # pixel level
            reveal_masks = torch.sigmoid(reveal_masks)

            reveal_masks_ = reveal_masks.squeeze(0)
            masks_ = masks.squeeze(0)

            try:
                p_auc = metrics.roc_auc_score(masks_.cpu().numpy().flatten(), reveal_masks_.cpu().numpy().flatten(), labels=[0, 1])
            except:
                p_auc = np.ones(1)

            # eer, th = compute_eer(reveal_masks_[masks_ == 1].cpu().numpy().flatten(), reveal_masks_[masks_ == 0].cpu().numpy().flatten())                

            reveal_masks_ = (reveal_masks_ >= 0.5).float()

            p_f1, precision, recall, mIoU, mcc = calculate_pixel_score(reveal_masks_.flatten(), masks_.flatten())

            test_meter['loss'].update(g_loss.item())
            test_meter['loss_mask'].update(g_loss_mask.item())
            test_meter['loss_pred'].update(g_loss_pred.item())
            test_meter['p_f1'].update(p_f1)
            test_meter['precision'].update(precision)
            test_meter['recall'].update(recall)
            test_meter['mIoU'].update(mIoU)
            test_meter['mcc'].update(mcc)
            test_meter['p_auc'].update(p_auc.item())

            # image_level
            img_pred_list += torch.max(torch.softmax(pred, dim=1), dim=1)[1].tolist()
            img_socre_list += torch.softmax(pred, dim=1)[:, 1].tolist()
            img_label_list += labels.tolist()

            # if i == 0:
            #     reveal_masks = torch.sigmoid(reveal_masks)
            #     reveal_masks[reveal_masks >= 0.25] = 1
            #     reveal_masks[reveal_masks < 0.25] = 0
            #     save_images(imgs, masks, reveal_masks, config.val_batch_size, os.path.join(config.TEST_PATH), 256)

            # batch size 需要能被整除，不然保存的图片会有问题
            # if i % (1500 // config.TEST.BATCH_SIZE)  == 0:
            #     img_list.append(imgs)
            #     mask_list.append(masks)
            #     pre_list.append(reveal_masks)
            torchvision.utils.save_image(imgs, os.path.join(model_path, 'result', f'{image_names[0]}.png'), normalize=False)
            torchvision.utils.save_image(masks, os.path.join(model_path, 'result', f'{image_names[0]}_mask.png'), normalize=False)
            torchvision.utils.save_image(reveal_masks, os.path.join(model_path, 'result', f'{image_names[0]}_pred_mask.png'), normalize=False)
            torchvision.utils.save_image(reveal_masks_, os.path.join(model_path, 'result', f'{image_names[0]}_pred_mask_0.5.png'), normalize=False)
            # save_images(imgs, masks, reveal_masks, config.TEST.BATCH_SIZE, os.path.join(model_path, 'result'), 256, i)



        
        try:
            img_auc = metrics.roc_auc_score(img_label_list, img_socre_list)
        except:
            img_auc = np.zeros(1)
        _, _, _, img_f1, _, _, _, _ = calculate_img_score_np(img_pred_list, img_label_list)
        test_meter['img_f1'].update(img_f1.item())
        test_meter['img_auc'].update(img_auc.item())

        if not os.path.exists(os.path.join(model_path, 'result')):
            os.makedirs(os.path.join(model_path, 'result'))
        # save_images(torch.concat(img_list, dim=0), torch.concat(mask_list, dim=0), torch.concat(pre_list, dim=0), config.TEST.BATCH_SIZE, model_path + os.makedirs(os.path.join(model_path, 'result')), 256)

        logger_test.info(f"Test result: "
            f"loss: {test_meter['loss'].avg:.6f} |"
            f"loss_mask: {test_meter['loss_mask'].avg:.6f} |"  
            f"loss_pred: {test_meter['loss_pred'].avg:.6f} |"
            f"p_f1: {test_meter['p_f1'].avg:.6f} |"
            f"precision: {test_meter['precision'].avg:.6f} |"
            f"recall: {test_meter['recall'].avg:.6f} |"
            f"mIoU: {test_meter['mIoU'].avg:.6f} |"
            f"mcc: {test_meter['mcc'].avg:.6f} |"
            f"p_auc: {test_meter['p_auc'].avg:.6f} |"
            f"img_f1: {test_meter['img_f1'].avg:.6f} |"
            f"img_auc: {test_meter['img_auc'].avg:.6f} |"
        )


def save_images(images, masks, reveal_masks, rows, folder, resize_to=None, name='test'):
    masks = masks.expand_as(images)
    reveal_masks = reveal_masks.expand_as(images)
    images_list = torch.split(images, rows, dim=0)
    masks_list = torch.split(masks, rows, dim=0)
    reveal_masks_list = torch.split(reveal_masks, rows, dim=0)
    stack_images = []
    for data in zip(images_list, masks_list, reveal_masks_list):
        stack_images.append(torch.concat(data, dim=0))
    stack_images = torch.concat(stack_images, dim=0)

    if resize_to is not None:
        stack_images = F.interpolate(stack_images, size=resize_to)
    torchvision.utils.save_image(stack_images, os.path.join(folder, f'{name}.png'), nrow=rows, normalize=False)


if __name__ == '__main__':
    args = parse_args()
    main(args)
