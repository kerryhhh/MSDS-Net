from glob import glob
import sys
import glob
from os.path import join
from attr import attr
import numpy as np
from PIL import Image
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from natsort import natsorted


class CelebA(Dataset):
    def __init__(self, data_path, data_list_path, attr_path, mode='train', transforms=None, origin=True,
                 selected_attrs=['Black_Hair', 'Bushy_Eyebrows', 'Mouth_Slightly_Open', 'Young']):
        super().__init__()
        self.data_path = data_path
        self.transforms = transforms
        self.selected_attrs = selected_attrs
        self.mode = mode
        attr_list = open(attr_path, 'r', encoding='utf-8').readlines()[1].split()
        attrs_index = [attr_list.index(attr) + 1 for attr in selected_attrs]
        orig_attrs = np.loadtxt(attr_path, skiprows=2, usecols=attrs_index, dtype=int)
        
        img_idx = np.loadtxt(data_list_path, skiprows=1, usecols=[1], dtype=int)
        images = np.loadtxt(data_list_path, skiprows=1, usecols=[2], dtype=str)

        if mode == 'train':
            images = images[: 28000]
            img_idx = img_idx[: 28000]

        elif mode == 'val':
            images = images[28000: 28500]
            img_idx = img_idx[28000: 28500]
        
        else:
            images = images[28500:]
            img_idx = img_idx[28500:]

        self.images = []
        self.labels = []
        # for img in images:
        #     self.images.append(os.path.join('origin', img))
        #     self.labels.append('origin')
        
        if origin == True:
            # origin
            self.images += images.tolist()
            self.labels += ['origin'] * len(images)
        for attr in selected_attrs:
            # for img in images:
            #     self.images.append(os.path.join(attr, img))
            #     self.labels.append(attr)
            self.images += images.tolist()
            self.labels += [attr] * len(images)

        self.attrs = orig_attrs[img_idx].tolist() * (len(selected_attrs) + 1)

    def __getitem__(self, index):
        # label = torch.zeros(len(self.selected_attrs), dtype=torch.float)
        img = Image.open(os.path.join(self.data_path, self.labels[index], self.images[index])).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        if self.labels[index] == 'origin':
            mask = torch.zeros((1, img.shape[1], img.shape[2]), dtype=torch.float)
            label = 0
        else:
            mask = Image.open(os.path.join(self.data_path, self.labels[index], self.images[index][:-4] + '.png')).convert('1')
            # label[self.selected_attrs.index(self.labels[index])] = 1 if self.attrs[index][self.selected_attrs.index(self.labels[index])] == -1 else -1
            # label[self.selected_attrs.index(self.labels[index])] = 1
            label = 1
            if self.transforms is not None:
                mask = self.transforms(mask)
        if self.mode != 'test':
            return img, mask, label
        else:
            img_name = self.labels[index] + self.images[index][:-4]
            return img, mask, label, img_name
    
    def __len__(self):
        return len(self.images)



if __name__ == '__main__':

    transform_train = T.Compose([
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    T.Resize(256),
    T.ToTensor(),
])
    dataloader = DataLoader(CelebA(data_path='/data/hjk/CelebA-HQ-STGAN-Mask/', data_list_path='/data/hjk/CelebA-HQ-STGAN-Mask/image_list.txt', attr_path='/data/hjk/CelebA-HQ-STGAN-Mask/list_attr_celeba.txt',mode='train', transforms=transform_train),
                            batch_size=1,
                            shuffle=False)
    for data in dataloader:
        print(data[2])