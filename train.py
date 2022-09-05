# %% imports
import matplotlib.pyplot as plt
import torch
import mlflow
import mlflow.pytorch
from attrdict import AttrDict
from tqdm import tqdm

from utils import *
from dataset import *
from models.terraGan import MainModel

# set up config
import hyperparams
cfg = AttrDict(hyperparams.UNET_GAN_PARAMS)

# plot settings
plt.rcParams['figure.dpi'] = 300    
plt.rcParams['savefig.dpi'] = 300

# set training device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# main training loop
def train_model(model, train_dl, epochs, log_every, save_fig_every):
    i = 0
    for e in range(epochs): # loop over the dataset multiple times
        print('Epoch: {}'.format(e)) # print epoch number
        loss_meter_dict = create_loss_meters() # create loss meters for tracking losses
        
        for data in tqdm(train_dl): # loop over the data batches
            model.setup_input(data) # set up input data
            model.optimize() # optimize model
            
            update_losses(model, loss_meter_dict, count=data['x'].size(0)) # update loss meters
            
            if i % log_every == 0:
                mlflow.log_metrics(get_results_dict(loss_meter_dict), step=i) # log losses to mlflow for current step
                
            i += 1
            
        if e % save_fig_every == 0:
            fig = visualize(model, data, show=False) # create figure of current model performance
            mlflow.log_figure(fig, 'epoch'+str(e)+'.jpg') # log figure to mlflow
            plt.close(fig)

def main():
    mlflow.set_experiment(experiment_name="Unet Generator") # set mlflow experiment name
    
    # create data loader
    print("Loading data...") 
    train_dl = make_dataloader(cfg.train_path, cfg.img_size, cfg.batch_size, cfg.n_workers, cfg.pin_memory)
    
    # create model
    model = MainModel(lr_G=cfg.lr_G, lr_D=cfg.lr_D, beta1=cfg.beta1, beta2=cfg.beta2, lambda_L1=cfg.lambda_L1)
    print(model)
    
    # train model with mlflow
    with mlflow.start_run():
        
        mlflow.log_params(cfg) # log hyperparameters to mlflow
        
        # train model
        print("Training model...")
        train_model(model, train_dl, cfg.epochs, cfg.log_every, cfg.save_fig_every)
        
        # save model to mlflow
        mlflow.pytorch.log_model(model, "model")

# %% run training
if __name__ == "__main__":  
    main()
