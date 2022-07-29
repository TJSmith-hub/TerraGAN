import glob, os, cv2
import matplotlib.pyplot as plt
import numpy as np
from attrdict import AttrDict

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from hyperparams import TRAIN_PARAMS
cfg = AttrDict(TRAIN_PARAMS)

class TerrainData(torch.utils.data.Dataset):
    def __init__(self, path, sourceTransform):
        self.rgbdata = np.array([np.array(fname) for fname in glob.glob(os.path.join(path, '*t.png'))])
        self.heightdata = np.array([np.array(fname) for fname in glob.glob(os.path.join(path, '*h.png'))])
        self.segdata = np.array([np.array(fname) for fname in glob.glob(os.path.join(path, '*i2.png'))])
        self.sourceTransform = sourceTransform
        return

    def __len__(self):
        return self.rgbdata.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rgb = cv2.imread(self.rgbdata[idx])
        rgb = cv2.resize(rgb, (cfg.img_size, cfg.img_size))
        rgbh = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGBA)
        height = cv2.imread(self.heightdata[idx], cv2.IMREAD_GRAYSCALE)
        height = cv2.resize(height, (cfg.img_size, cfg.img_size))
        label = cv2.imread(self.segdata[idx])
        label = cv2.resize(label, (cfg.img_size, cfg.img_size))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        #if self.sourceTransform:
            #image = self.sourceTransform(image)
            
        rgbh[:,:,3] = height
            
        rgbh = transforms.ToTensor()(np.uint8(rgbh))
        label = transforms.ToTensor()(np.uint8(label))

        return {'x':label, 'y': rgbh}
    
def make_dataloader(path, batch_size, n_workers, pin_memory): # A handy function to make our dataloaders
    dataset = TerrainData(path, None)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader