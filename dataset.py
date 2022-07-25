import glob, os, cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class TerrainData(torch.utils.data.Dataset):
    def __init__(self, path, sourceTransform):
        self.rgbdata = np.array([np.array(fname) for fname in glob.glob(os.path.join(path, '*t.png'))])
        self.segdata = np.array([np.array(fname) for fname in glob.glob(os.path.join(path, '*i2.png'))])
        self.sourceTransform = sourceTransform
        return

    def __len__(self):
        return self.rgbdata.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = cv2.imread(self.rgbdata[idx])
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(self.segdata[idx])
        label = cv2.resize(label, (256, 256))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        if self.sourceTransform:
            image = self.sourceTransform(image)
            
        image = transforms.ToTensor()(image)
        label = transforms.ToTensor()(label)

        return {'x':label, 'y': image}
    
def make_dataloader(path, batch_size=16, n_workers=4, pin_memory=True): # A handy function to make our dataloaders
    dataset = TerrainData(path, None)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader