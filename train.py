# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from tensorboard import notebook
import torch
import torchvision
from utils import *
from tqdm import tqdm

from dataset import *
from terraGan import MainModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_path = "dataset/data_train"
val_path = "dataset/data_val"

def train_model(model, train_dl, epochs, display_every=100):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['x'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs

# %%
if __name__ == "__main__":  
    
    print("Loading data...")
    train_dl = make_dataloader(train_path)
    val_dl = make_dataloader(val_path)

    model = MainModel()

    print("Training model...")
    train_model(model, train_dl, 100)