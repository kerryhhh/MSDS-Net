import math
import os
import logging
from datetime import datetime
import yaml
import struct
import shutil
import numpy as np

import torch
import torch.nn.functional as F
import torchvision


def calculate_img_score_np(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    TP = float(np.logical_and(pd, gt).sum())
    FP = np.logical_and(pd, gt_inv).sum()
    FN = np.logical_and(seg_inv, gt).sum()
    TN = float(np.logical_and(seg_inv, gt_inv).sum())
    acc = (TP + TN) / (TP + TN + FN + FP + 1e-6)
    sen = TP / (TP + FN + 1e-6)
    spe = TN / (TN + FP + 1e-6)
    # pre = TP / (TP + FP + 1e-6)
    # f1 = 2 * sen * spe / (sen + spe)
    # f1 = 2 * sen * pre / (sen + pre)
    f1 = 2 * TP / (2 * TP + FP + FN + 1e-6)
    return acc, sen, spe, f1, TP, TN, FP, FN


def calculate_pixel_score_np(pd, gt):
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, mIoU = 1.0, 1.0
        return f1, 0.0, 0.0, mIoU
    torch.max()
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    TP = float(np.logical_and(pd, gt).sum())
    TN = float(np.logical_and(seg_inv, gt_inv).sum())
    FP = float(np.logical_and(pd, gt_inv).sum())
    FN = float(np.logical_and(seg_inv, gt).sum())
    f1 = 2 * TP / (2 * TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    mIoU = 0.5 * (TP / (FN + FP + TP + 1e-6)) + 0.5 * (TN / (FN + FP + TN + 1e-6))
    mcc = (TP * TN - FP * FN) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-6)
    return f1, precision, recall, mIoU, mcc


def calculate_pixel_score(pd, gt):
    if torch.max(pd) == torch.max(gt) and torch.max(pd) == 0:
        f1, mIoU, mcc = 1.0, 1.0, 1.0
        return f1, 0.0, 0.0, mIoU, mcc
    seg_inv, gt_inv = torch.logical_not(pd), torch.logical_not(gt)
    TP = float(torch.logical_and(pd, gt).sum())
    TN = float(torch.logical_and(seg_inv, gt_inv).sum())
    FP = float(torch.logical_and(pd, gt_inv).sum())
    FN = float(torch.logical_and(seg_inv, gt).sum())
    f1 = 2 * TP / (2 * TP + FP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    mIoU = 0.5 * (TP / (FN + FP + TP + 1e-6)) + 0.5 * (TN / (FN + FP + TN + 1e-6))
    mcc = (TP * TN - FP * FN) / (math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-6)
    return f1, precision, recall, mIoU, mcc


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size)) # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)) # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])) # Thresholds are the sorted scores

    return frr, far, thresholds


def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]