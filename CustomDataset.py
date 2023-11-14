import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder 
import os

import numpy as np
from glob import glob
import random
random.seed(0)


class CustomDataset(Dataset):
    def __init__(self, args, data_root, mode='train'):
        self.subdirs = list(sorted(os.listdir(data_root))[:args.num_class])
        self.dataindex = []
        for sub in self.subdirs:
            files = glob(os.path.join(data_root, sub, '*.npy'))
            for f in files:
                self.dataindex.append({
                    'data': f,
                    'label': np.array([self.label2index(sub)])
                })
        self.scale_factor = 0.18215
        random.shuffle(self.dataindex)

    def label2index(self, label):
        return self.subdirs.index(label)

    def __len__(self):
        return len(self.dataindex)

    def __getitem__(self, idx):
        feature = torch.from_numpy(np.load(self.dataindex[idx]['data']))
        # feature = feature * self.scale_factor
        label = torch.from_numpy(self.dataindex[idx]['label'])
        return feature, label



if __name__=='__main__':
    import pdb
    dataset_train = CustomDataset('/remote-home/share/yfsong/imageNet/train')
    dataset_test = CustomDataset('/remote-home/share/yfsong/imageNet/val')
    # print(dataset_train.subdirs[:100])
    # print(dataset_test.subdirs[:100])
    dataloader = DataLoader(dataset_test, batch_size=2)
    for x ,l in dataloader:
        print(x.shape)
        # pdb.set_trace()
        print(l.shape)