import torch
import torch.nn as nn
from torch.nn import init

from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

import math
import numpy as np


class TestNet(nn.Module):   
    def __init__(self):
        super(TestNet, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(6, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            # MaxPool2d(kernel_size=2, stride=2),
        )

    # Defining the forward pass    
    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x=(inputs - rgb_mean) /255
        x1 = inputs[:,:,0,:,:]
        x2 = inputs[:,:,1,:,:]
        x = torch.cat((x1,x2), dim = 1)
        x = self.cnn_layers(x)
        return x