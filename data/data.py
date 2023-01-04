# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Custom datasets for CelebA and CelebA-HQ."""

import numpy as np
import os
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


class Custom(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, selected_attrs):
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        self.images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        self.labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att
    
    def __len__(self):
        return len(self.images)

class CelebA(data.Dataset):
    def __init__(self, data_path, attr_path, image_size, mode, selected_attrs):
        super(CelebA, self).__init__()
        self.data_path = data_path    #数据集图像所在路径'/data1/huangwenmin/CelebA/CelebA/Img'
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()   #返回标注的属性值列表，每个元素对应一个属性名称，列表长为40
        atts = [att_list.index(att) + 1 for att in selected_attrs]  #要编辑的属性的索引列表，长度13，返回要学习的属性在属性值列表中的索引+1
        images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)  #提取celebA数据集的全部图像名，既*.jpg (202599)
        labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        #提取要编辑的13个属性的标注，长度为202599的列表，每个元素是长度13列表，属性值用-1,1标注
        
        if mode == 'train':
            self.images = images[:182000]   #取出前182000张图像作为训练集
            self.labels = labels[:182000]   #取出前182000张图像属性标注
        if mode == 'valid':
            self.images = images[182000:182637]  #637张图像
            self.labels = labels[182000:182637]
        if mode == 'test':
            self.images = images[182637:]   #19965
            self.labels = labels[182637:]
        
        self.tf = transforms.Compose([
            transforms.CenterCrop(170),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])    #预处理：178*218使用中心剪裁成170*170，然后调整为128*128
                                       
        self.length = len(self.images)  #图像数量202599
    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))  #读取图像
        att = torch.tensor((self.labels[index] + 1) // 2)   #读取属性标注，并将-1,1转化成0,1
        return img, att
    def __len__(self):
        return self.length


class CelebA_HQ(data.Dataset):
    def __init__(self, data_path, attr_path, image_list_path, image_size, mode, selected_attrs):
        super(CelebA_HQ, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        orig_images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        orig_labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        # total=np.sum(orig_labels+1,axis=0)/2
        indices = np.loadtxt(image_list_path, skiprows=1, usecols=[1], dtype=np.int)
        indices2 = np.loadtxt(image_list_path, skiprows=1, usecols=[2], dtype=np.str)

        images = [i for i in indices2]
        labels = orig_labels[indices]
        # total = np.sum(labels  + 1, axis=0) / 2
        if mode == 'train':
            self.images = images[:28000]
            self.labels = labels[:28000]
        if mode == 'valid':
            self.images = images[28000:28500]
            self.labels = labels[28000:28500]
        if mode == 'test':
            self.images = images[28500:]
            self.labels = labels[28500:]

        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att

    def __len__(self):
        return self.length

def check_attribute_conflict(att_batch, att_name, att_names):  #对可能冲突的属性标注进行验证，存在冲突时仅保留其中一个
    def _get(att, att_name):  #属性标注列表13，属性名，，作用：返回某属性的标签
        if att_name in att_names:
            return att[att_names.index(att_name)]
        return None
    def _set(att, value, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = value
    att_id = att_names.index(att_name)    #从属性名列表中取出属性所在的位置
    for att in att_batch:     #取出一张图像的属性标签，长度13
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            if _get(att, 'Bangs') != 0:
                _set(att, 1-att[att_id], 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            for n in ['Bald', 'Receding_Hairline']:
                if _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name and _get(att, n) != 0:
                    _set(att, 1-att[att_id], n)
    return att_batch

class CelebA_HQ_t(data.Dataset):
    def __init__(self, data_path, attr_path, image_list_path, image_size, mode, selected_attrs):
        super(CelebA_HQ_t, self).__init__()
        self.data_path = data_path
        att_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        atts = [att_list.index(att) + 1 for att in selected_attrs]
        orig_images = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
        orig_labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
        # total=np.sum(orig_labels+1,axis=0)/2
        indices = np.loadtxt(image_list_path, skiprows=1, usecols=[1], dtype=np.int)
        indices2 = np.loadtxt(image_list_path, skiprows=1, usecols=[2], dtype=np.str)

        images = [i for i in indices2]
        labels = orig_labels[indices]
        # total = np.sum(labels  + 1, axis=0) / 2
        if mode == 'train':
            self.images = images[:28000]
            self.labels = labels[:28000]
        if mode == 'valid':
            self.images = images[28000:28500]
            self.labels = labels[28000:28500]
        if mode == 'test':
            self.images = images[28500:]
            self.labels = labels[28500:]

        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.tf(Image.open(os.path.join(self.data_path, self.images[index])))
        att = torch.tensor((self.labels[index] + 1) // 2)
        return img, att, self.images[index]

    def __len__(self):
        return self.length


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils

    attrs_default = [
        'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
        'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
    ]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to test')
    parser.add_argument('--data_path', dest='data_path', type=str, required=True)
    parser.add_argument('--attr_path', dest='attr_path', type=str, required=True)
    args = parser.parse_args()
    
    dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=64, shuffle=False, drop_last=False
    )

    print('Attributes:')
    print(args.attrs)
    for x, y in dataloader:
        vutils.save_image(x, 'test.png', nrow=8, normalize=True, range=(-1., 1.))
        print(y)
        break
    del x, y
    
    dataset = CelebA(args.data_path, args.attr_path, 128, 'valid', args.attrs)
    dataloader = data.DataLoader(
        dataset, batch_size=16, shuffle=False, drop_last=False
    )