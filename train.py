# %%
import numpy as np
import matplotlib.pyplot as plt
import torch
from attrdict import AttrDict
from utils import *
from tqdm import tqdm

from dataset import *
from models.terraGan import MainModel

import hyperparams
cfg = AttrDict(hyperparams.UNET_GAN_PARAMS)

plt.rcParams['figure.dpi'] = 300    
plt.rcParams['savefig.dpi'] = 300

import mlflow
import mlflow.pytorch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, train_dl, epochs, display_every, save_fig_every):
    i = 0
    for e in range(epochs):
        print('Epoch: {}'.format(e))
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['x'].size(0)) # function updating the log objects
            if i % display_every == 0:
                mlflow.log_metrics(get_results_dict(loss_meter_dict), step=i)
            i += 1
        if e % save_fig_every == 0:
            fig = visualize(model, data, show=False) # function displaying the model's outputs
            mlflow.log_figure(fig, 'epoch'+str(e)+'.jpg') # log figure
            plt.close(fig)
        #mlflow.log_metrics(get_results_dict(loss_meter_dict), step=i) # log losses to mlflow
        #log_results(loss_meter_dict) # function to print out the losses
    #mlflow.log_metrics(get_results_dict(loss_meter_dict), step=e) # log losses to mlflow

def main():
    mlflow.set_experiment(experiment_name="Unet Generator")
    print("Loading data...")
    train_dl = make_dataloader(cfg.train_path, cfg.img_size, cfg.batch_size, cfg.n_workers, cfg.pin_memory)
    
    model = MainModel(lr_G=cfg.lr_G, lr_D=cfg.lr_D, beta1=cfg.beta1, beta2=cfg.beta2, lambda_L1=cfg.lambda_L1)
    print(model)
    with mlflow.start_run():
        
        mlflow.log_params(cfg)
        
        print("Training model...")
        train_model(model, train_dl, cfg.epochs, cfg.display_every, cfg.save_fig_every)
        
        mlflow.pytorch.log_model(model, "model")

# %%
if __name__ == "__main__":  
    main()
