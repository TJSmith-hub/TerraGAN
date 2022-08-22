import torch
import torch.nn as nn
from torch import nn, optim

import models.unet_g as old_unet_g
import models.unet_generator as unet_g
import spade.models.generator as SPADE_G
from models.descriminator import PatchDiscriminator
from models.ganloss import GANLoss

from attrdict import AttrDict
import hyperparams
cfg = AttrDict(hyperparams.UNET_GAN_PARAMS)
    
def init_weights(net, init='norm', gain=0.02):
    
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)
            
    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model
    
class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, 
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1
        
        if net_G == 'unet':
            self.net_G = init_model(unet_g.UNet(cfg.input_c_G, cfg.output_c_G, cfg.n_down_G, cfg.n_filters_G), self.device)
        elif net_G == 'old_unet':
            self.net_G = init_model(old_unet_g.UNet(cfg.input_c_G, cfg.output_c_G), self.device)
        elif net_G == 'spade':
            from spade.args import get_parser
            parser = get_parser()
            args, _ = parser.parse_known_args()
            self.net_G = init_model(SPADE_G.SPADEGenerator(args), self.device)
            
        self.net_D = init_model(PatchDiscriminator(cfg.input_c_D, cfg.n_down_D, cfg.n_filters_D), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad
        
    def setup_input(self, data):
        self.x = data['x'].to(self.device)
        self.y = torch.cat((data['yt'], data['yh']), 1).to(self.device) 
        #self.y = data['yh'].to(self.device)
        
    def forward(self):
        self.y_fake = self.net_G(self.x)
    
    def backward_D(self):
        fake_cat = torch.cat((self.x, self.y_fake), 1)
        fake_preds = self.net_D(fake_cat.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_cat = torch.cat((self.x, self.y), 1)
        real_preds = self.net_D(real_cat)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_cat = torch.cat((self.x, self.y_fake), 1)
        fake_preds = self.net_D(fake_cat)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.y_fake, self.y) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()