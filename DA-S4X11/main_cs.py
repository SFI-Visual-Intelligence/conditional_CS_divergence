import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.optim as optim
import pandas as pd
from torchsummary import summary
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.nn import init
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
import time

from data_reader import * 
from losses import *
from train_util_cs import *

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA = 1


# init weights
def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)



class EEGNet_DA1(torch.nn.Module):
    """ The model for DJP-MMD, CS, CS + CCS
    """
    def __init__(self, n_output):
        super(EEGNet_DA1, self).__init__()
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=8,bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=0),
            nn.Dropout(p=0.35)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7),bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=0),
            nn.Dropout(p=0.35),
            nn.Flatten()
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(736, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
        self.classify = nn.Sequential(
            nn.Linear(256, n_output, bias=True)
        )

    def forward(self, source, target):
        out = self.firstConv(source)
        out = self.depthwiseConv(out)
        out = self.separableConv(out)
        features_src = self.bottleneck(out)
        out_source = self.classify(features_src)

        target = self.firstConv(target)
        target = self.depthwiseConv(target)
        target = self.separableConv(target)
        features_tar = self.bottleneck(target)
        out_target = self.classify(features_tar)

        return out_source, out_target, F.normalize(features_src, p=2.0, dim=-1), F.normalize(features_tar, p=2.0, dim=-1)
        #return out_source, out_target,features_src, features_tar


# ddc_lambda_list, ddc_acc_list, ddc_disc_list = main(model, path='DDC.jpg')

source_data, source_label, val_data, val_label, target_data, target_label = read_bci_data()

print(source_data.min(), source_data.max())

switch = False
#switch = True

if switch:
    target_data, target_label = source_data, source_label
    source_data, source_label = val_data, val_label
    val_data, val_label = target_data, target_label

# lambda_factor =  or 1
model = EEGNet_DA1(n_output=2)
# djp_lambda_list, djp_acc_list, djp_disc_list = main(model, 'djpmmd')
#_, _, djp_acc = train(model, source_data, source_label, val_data, val_label, target_data, target_label, mmd_type='CS', lambda_factor=30, batch_size=1080, epochs=200, lr=1e-2, path='CS', format='jpg')

_, _, djp_acc = train(model, source_data, source_label, val_data, val_label, target_data, target_label, mmd_type='CCS', lambda_factor=10, batch_size=1080, epochs=200, lr=1e-2, path='CS', format='jpg')

print(switch)
plt.plot(range(len(djp_acc)), djp_acc)

plt.savefig('./acc_curve.png')
plt.close()


