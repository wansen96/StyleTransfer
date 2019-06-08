import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import lr_scheduler

###########################

# Generator

###########################
class ResnetBlock(nn.Module):
    def __init__(self, nic):
        '''
        :param nic: number of input feature
        '''
        super(ResnetBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(nic, nic, 3),
                      nn.BatchNorm2d(nic),
                      nn.ReLU(True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(nic, nic, 3),
                      nn.BatchNorm2d(nic)
                      ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x+self.conv_block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, nic, out_channels, num_residual_blocks, ngf=64):
        '''

        :param nic: number of Input Channels
        :param out_channels:
        :param num_residual_blocks:
        :param ngf: number of filter in generator
        '''
        super(ResnetGenerator, self).__init__()

        # Initial Convolution Block

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(nic, ngf, kernel_size=7, padding=0, bias=True),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]

        # Encoder
        number_of_encoder = 2
        nic = ngf
        for _ in range(number_of_encoder):
            noc = nic * 2 #number of output channels
            model += [nn.Conv2d(nic, noc, kernel_size=3, stride=2, padding=1, bias=True),
                      nn.BatchNorm2d(noc),
                      nn.ReLU(True)]
            nic = noc #256

        # Transformer
        for _ in range(num_residual_blocks): # add resnet block
            model+=[ResnetBlock(nic)]

        # Decoder
        for _ in range(number_of_encoder):
            noc = int(nic/2)
            model += [nn.ConvTranspose2d(nic, noc, kernel_size=3, stride=2, 
                                         padding=1, output_padding=1,bias=True),
                      nn.BatchNorm2d(noc),
                      nn.ReLU(True)
                      ]
            nic = noc

        #Output Layer
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, out_channels,7,padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)


    def forward(self, x):
        return self.model(x)


###########################

# Discriminator

###########################


class Discriminator(nn.Module):
    """ PatchGAN discriminator"""
    def __init__(self, nic, input_shape):
        '''
        :param nic: number of input channels
        :param nlayers: number of layers
        '''
        super(Discriminator,self).__init__()
        height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(nic, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1))

    def forward(self,x):
        return self.model(x)

    
    
##########################

# lr scheduler
def get_scheduler(optimizer, step_size):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    return scheduler
