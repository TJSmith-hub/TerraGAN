import glob, os, cv2
import numpy as np
from PIL import Image, ImageOps

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class TerrainData(Dataset):
    def __init__(self, path, size):
        self.rgbdata = np.array([np.array(fname) for fname in glob.glob(os.path.join(path, '*t.png'))])
        self.heightdata = np.array([np.array(fname) for fname in glob.glob(os.path.join(path, '*h.png'))])
        self.segdata = np.array([np.array(fname) for fname in glob.glob(os.path.join(path, '*i2.png'))])
        self.transform_rgb = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_h = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        self.transform_s = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        return

    def __len__(self):
        return self.rgbdata.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        rgb = cv2.imread(self.rgbdata[idx])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = self.transform_rgb(rgb)
        
        height = cv2.imread(self.heightdata[idx], cv2.IMREAD_GRAYSCALE)
        height = self.transform_h(height)
        
        seg = cv2.imread(self.segdata[idx], cv2.IMREAD_GRAYSCALE)
        seg = self.transform_s(seg)
            
        return {'x':seg, 'yt': rgb, 'yh':height}
    
def make_dataloader(path, size, batch_size, n_workers, pin_memory): # A handy function to make our dataloaders
    dataset = TerrainData(path, size)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader