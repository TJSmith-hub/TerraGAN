import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb
import time

class AverageMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3
    
    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()
    
    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)
    
def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake = model.y_fake.detach()
    real = model.y
    x = model.x
    for i in range(3):
        ax = plt.subplot(3, 3, i + 1)
        ax.imshow(x[i].to('cpu').numpy().transpose(1, 2, 0))
        ax.axis("off")
        ax = plt.subplot(3, 3, i + 1 + 3)
        ax.imshow(fake[i].to('cpu').numpy().transpose(1, 2, 0))
        ax.axis("off")
        ax = plt.subplot(3, 3, i + 1 + 6)
        ax.imshow(real[i].to('cpu').numpy().transpose(1, 2, 0))
        ax.axis("off")
    plt.show()
        
def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")