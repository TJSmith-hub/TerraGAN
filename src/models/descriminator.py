import torch.nn as nn

# discriminator class
class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, n_down, num_filters):
        super().__init__()
        
        # define first block with no batchnorm
        model = [self.get_layers(input_c, num_filters, norm=False)]
        
        # define middle blocks
        for i in range(n_down):
            model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2)]
        
        # define last block with no batchnorm or activation
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)]
        
        self.model = nn.Sequential(*model)
    
    # function to create a layer block of conv, batchnorm, and leaky relu              
    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True):
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)
    
    # forward pass
    def forward(self, x):
        return self.model(x)