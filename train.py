# %%
import attrdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorboard import notebook
import torch
import torchvision
from attrdict import AttrDict
from utils import *
from tqdm import tqdm

from dataset import *
from terraGan import MainModel

from hyperparams import *
cfg = AttrDict(TRAIN_PARAMS)
G_cfg = AttrDict(G_PARAMS)
D_cfg = AttrDict(D_PARAMS)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

import mlflow
import mlflow.pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dl, epochs, display_every):
    data = next(iter(train_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        print('Epoch: {}'.format(e+1))
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['x'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                visualize(model, data, save=False) # function displaying the model's outputs
                mlflow.log_metrics(get_results_dict(loss_meter_dict), step=e) # log losses to mlflow
        log_results(loss_meter_dict) # function to print out the losses

def main():
    print("Loading data...")
    train_dl = make_dataloader(cfg.train_path, cfg.batch_size, cfg.n_workers, cfg.pin_memory)
    
    model = MainModel()

    with mlflow.start_run():
        
        mlflow.log_params(cfg)
        mlflow.log_params(G_cfg)
        mlflow.log_params(D_cfg)
        print("Training model...")
        train_model(model, train_dl, cfg.epochs, cfg.display_every)
        

        
        #mlflow.pytorch.log_model(model, "model")

# %%
if __name__ == "__main__":  
    main()
