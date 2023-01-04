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


class Inpainting(Dataset):
    def __init__(self, data_path, mode='train', transforms=None):
        super().__init__()
        self.data_path = data_path
        self.transforms = transforms
        
        with open('/home/huangjinkun/mycode/Detection/data/map.txt', 'r') as f:
            data_list = [data.strip().split(' ') for data in f.readlines()]
            data_2933 = data_list[: 2993]
            data_36500 = data_list[2993:]
        # if mode == 'train':
        #     self.data = data_2933[:2493] + data_36500[:34000]
        # elif mode == 'val':
        #     self.data = data_2933[2493: 2593] + data_36500[34000: 34500]
        # else:
        #     self.data = data_2933[2593:] + data_36500[34500:]
        # if mode == 'train':
        #     self.data = data_2933[:2393] + data_36500[:27200]
        # elif mode == 'val':
        #     self.data = data_2933[2393: 2493] + data_36500[27200: 27700]
        # else:
        #     self.data = data_2933[2493:] + data_36500[27700:]
        if mode == 'train':
            self.data = data_36500
        elif mode == 'val':
            self.data = data_2933[2000:]
        else:
            self.data = data_2933[:2000]


    def __getitem__(self, index):
        # label = torch.zeros(len(self.selected_attrs), dtype=torch.float)
        img = Image.open(os.path.join(self.data_path, self.data[index][0])).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        mask = Image.open(os.path.join(self.data_path, self.data[index][1])).convert('1')
        if self.transforms is not None:
            mask = 1 - self.transforms(mask)
        return img, mask, 1
    
    def __len__(self):
        return len(self.data)



if __name__ == '__main__':

    transform_train = T.Compose([
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    T.Resize(256),
    T.ToTensor(),
])
    dataloader = DataLoader(Inpainting(data_path='/data/hjk/CelebA-HQ-STGAN-Mask/', data_list_path='/data/hjk/CelebA-HQ-STGAN-Mask/image_list.txt', attr_path='/data/hjk/CelebA-HQ-STGAN-Mask/list_attr_celeba.txt',mode='train', transforms=transform_train),
                            batch_size=1,
                            shuffle=False)
    for data in dataloader:
        print(data[2])