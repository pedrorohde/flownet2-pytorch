'''
Portions of this code copyright 2017, Clement Pinard
'''

# freda (todo) : adversarial loss 

import torch
import torch.nn as nn
import math
import pytorch_msssim

def EPE(input_flow, target_flow):
    return torch.norm(target_flow-input_flow,p=2,dim=1).mean()

class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.abs(output - target).mean()
        return lossvalue

class L2(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return lossvalue

class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()
        self.args = args
        self.loss = L1()
        self.loss_labels = ['L1', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class L2Loss(nn.Module):
    def __init__(self, args):
        super(L2Loss, self).__init__()
        self.args = args
        self.loss = L2()
        self.loss_labels = ['L2', 'EPE']

    def forward(self, output, target):
        lossvalue = self.loss(output, target)
        epevalue = EPE(output, target)
        return [lossvalue, epevalue]

class MultiScale(nn.Module):
    def __init__(self, args, startScale = 4, numScales = 5, l_weight= 0.32, norm= 'L1'):
        super(MultiScale,self).__init__()

        self.startScale = startScale
        self.numScales = numScales
        self.loss_weights = torch.FloatTensor([(l_weight / 2 ** scale) for scale in range(self.numScales)])
        self.args = args
        self.l_type = norm
        self.div_flow = 0.05
        assert(len(self.loss_weights) == self.numScales)

        if self.l_type == 'L1':
            self.loss = L1()
        else:
            self.loss = L2()

        self.multiScales = [nn.AvgPool2d(self.startScale * (2**scale), self.startScale * (2**scale)) for scale in range(self.numScales)]
        self.loss_labels = ['MultiScale-'+self.l_type, 'EPE'],

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0

        if type(output) is tuple:
            target = self.div_flow * target
            for i, output_ in enumerate(output):
                target_ = self.multiScales[i](target)
                epevalue += self.loss_weights[i]*EPE(output_, target_)
                lossvalue += self.loss_weights[i]*self.loss(output_, target_)
            return [lossvalue, epevalue]
        else:
            epevalue += EPE(output, target)
            lossvalue += self.loss(output, target)
            return  [lossvalue, epevalue]

class L2InterpolLoss(nn.Module):
    def __init__(self, args):
        super(L2, self).__init__()
        self.args = args
        self.loss_labels = ['L2']

    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return [ lossvalue ]

class L2InterpolLoss(nn.Module):
    def __init__(self, args):
        super(L2InterpolLoss, self).__init__()
        self.args = args
        self.loss_labels = ['L2']

    def forward(self, output, target):
        lossvalue = torch.norm(output-target,p=2,dim=1).mean()
        return [lossvalue]

class MSSSIMLoss(nn.Module):
    def __init__(self, args):
        super(MSSSIMLoss, self).__init__()
        self.args = args
        self.loss_labels = ['MS-SSIM']

    def forward(self, output, target):
        lossvalue = (1 - pytorch_msssim.ms_ssim(output, target, data_range=255.0, size_average=True))
        
        return [ lossvalue ]

class MSSSIML1Loss(nn.Module):
    def __init__(self, args):
        super(MSSSIML1Loss, self).__init__()
        self.args = args
        self.w = 0.84 # empirically set (see paper) 
        self.loss_labels = ['MS-SSIM_L1']

        self.MS_SSIM = pytorch_msssim.MS_SSIM(
            data_range=255.0,
            size_average=True,
            weights=[1.]*5 # no different weights for each level (check if true)
        )

        # use default values from pytorch-msssim (TODO: maybe change this?)
        win = pytorch_msssim._fspecial_gauss_1d(size=11, sigma=1.5)
        # apply gaussian filter and cast window to input's device/dtype
        self.gaussian_filter = lambda X: pytorch_msssim.gaussian_filter(X, win.to(X.device, dtype=X.dtype))

    def forward(self, output, target):
        loss_mssim = 1 - self.MS_SSIM(output, target)
        loss_l1 = torch.abs(output - target)
        # TODO: check if it should be .mean or .sum
        lossvalue = self.w*loss_mssim + (1-self.w)*self.gaussian_filter(loss_l1).mean()
        return [ lossvalue ]