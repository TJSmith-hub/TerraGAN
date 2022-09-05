import torch
from torch import nn

# class for constructing a U-Net
class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        
        # define variables
        self.outermost = outermost
        if input_c is None: input_c = nf
        
        # define layers
        downconv = nn.Conv2d(input_c, ni, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)
        
        # create outermost block
        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        # create innermost block
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        # create intermediate block
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            # add dropout
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        
        # set complete model
        self.model = nn.Sequential(*model)
    
    # forward pass
    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

# generator class
class UNet(nn.Module):
    def __init__(self, input_c, output_c, n_down, num_filters):
        super().__init__()
        
        # define innermost block
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        
        # define intermediate blocks with max number of filters
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
            
        out_filters = num_filters * 8
        
        # define intermediate blocks with decreasing number of filters
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        
        # define model with outermost block including all previous blocks
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)
        
    # forward pass
    def forward(self, x):
        return self.model(x)