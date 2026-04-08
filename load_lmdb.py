import torch
import torch.nn as nn
import torch.utils.data
import random
from copy import deepcopy
import torch.utils.tensorboard
import torchvision
import timm
import tqdm
import os
import lmdb
import json
import cv2
import h5py
import pickle
from typing import Any, Callable, Optional, Tuple
import numpy as np
from PIL import Image
from torchvision.datasets.vision import VisionDataset
from torchvision.transforms import ToTensor, Lambda
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Dataset(VisionDataset):
    def __init__(
        self,
        lmdb_path,
        label_path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        size: int = 224,
    ) -> None:

        self.lmdb = lmdb_path
        self.label_path = label_path
        self.transform=transform
        self.target_transform=target_transform
        self.size = (size, size)
        self.index_imgjson = {}
        self.data = []
        self.targets = []
        self.img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        ])
        self.env = lmdb.open(self.lmdb, readonly=True, lock=False)
        self.idx = 0
        with open(self.label_path, 'r') as f:
            labels = json.load(f)
            image_names = list(labels.keys())
        with self.env.begin() as txn:
            #self.num_samples = self._get_num_samples(txn)
            for name in tqdm.tqdm(image_names, total = len(image_names)):
                img_key = f'img_{name}'.encode('utf-8')
                #self.data.append(img_key)
                
                img_bytes = txn.get(img_key)
                if img_bytes is None:
                    img_key_without_prefix = name.encode('utf-8')
                    img_bytes = txn.get(img_key_without_prefix)
                
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                img = Image.fromarray(img)

                self.data.append(img)
                label = labels[name]
                self.targets.append(label)

                self.index_imgjson[self.idx] = (name, label)
                self.idx += 1
                #print(self.idx)

    def _get_num_samples(self, txn):
        num_samples = 0
        cursor = txn.cursor()
        for key, _ in cursor:
            if key.startswith(b'label_'):
                num_samples += 1
        return num_samples


    def __getitem__(self, index: int):
        img = self.data[index]
        '''
        with self.env.begin() as txn:
            img_bytes = txn.get(self.data[index])
            img_np = np.frombuffer(img_bytes, dtype=np.uint8).reshape(self.size + (3,))
            img = Image.fromarray(img_np)
        '''
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target

    def __len__(self) -> int:
        return self.idx

    def index_img(self):
        return self.index_imgjson



class StringToIndexTransform:
    def __init__(self, class_mapping):
        self.class_mapping = class_mapping

    def __call__(self, label):
        return self.class_mapping[label]



class SubsetDataset(Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform
        self.index_imgjson = {}
        self.data = []
        self.original_idx = []
        self.targets = []
        for index in tqdm.tqdm(range(len(self.indices)), total = len(self.indices)):
            original_idx = self.indices[index]
            # Retrieve the corresponding sample from the original dataset
            sample = dataset[original_idx]
            self.data.append(sample[0])
            self.targets.append(sample[1])
            #print(index)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        #idx = self.indices[idx]
        #sample = self.dataset[idx]
        img = self.data[idx]
        target = self.targets[idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        
        return img, target

    
    def get_indices(self):
        return self.indices
    



if __name__ == '__main__':
    raise RuntimeError('[ERROR] The module vitcifar10 is not supposed to be run as an executable.')
