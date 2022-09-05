# %% imports
import matplotlib.pyplot as plt
import torch
import mlflow
import mlflow.pytorch
from PIL import Image
from attrdict import AttrDict
from tqdm import tqdm
from pytorch_fid import fid_score
from utils import *
from dataset import *
from models.terraGan import MainModel

# set up config
import hyperparams
cfg = AttrDict(hyperparams.UNET_GAN_PARAMS)

if __name__ == '__main__':
    # create data loader
    print("Loading data...") 
    train_dl = make_dataloader(cfg.train_path, cfg.img_size, cfg.batch_size, cfg.n_workers, cfg.pin_memory)
    
    print("loading model...")
    logged_model = 'runs:/5f63a51f15e44ed7b3506396d74a221e/model'
    model = mlflow.pytorch.load_model(logged_model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    i = 1
    for data in train_dl:
        model.setup_input(data)
        model.forward()
        for j in range(cfg.batch_size):
            fake = model.y_fake[j].detach().cpu().numpy().transpose(1, 2, 0)
            real = model.y[j].detach().cpu().numpy().transpose(1, 2, 0)
            Image.fromarray((fake * 255).astype(np.uint8)).save('test_fake/' + str(i + j) + '_fake.png')
            Image.fromarray((real * 255).astype(np.uint8)).save('test_real/' + str(i + j) + '_real.png')
        i += cfg.batch_size
