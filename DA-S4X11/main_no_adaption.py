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



# reference.
# 1. https://github.com/agrija9/Deep-Unsupervised-Domain-Adaptation/blob/master/DDC/model.py
class EEGNet_ReLU(torch.nn.Module):
    """
    EEGNet as defined in the paper:
        https://arxiv.org/abs/1611.08024
    """
    def __init__(self, n_output):
        super(EEGNet_ReLU, self).__init__()
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
            nn.Flatten(),
            nn.Linear(736, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
        self.classify = nn.Sequential(
            nn.Linear(256, n_output, bias=True)
        )

    def forward(self, x):
        out = self.firstConv(x)
        out = self.depthwiseConv(out)
        features = self.separableConv(out)
        out = self.classify(features)

        out2 = self.firstConv(x)
        out2 = self.depthwiseConv(out2)
        features = self.separableConv(out2)
        #out = self.classify(features)

        return out, features


def test(data, label, model):
    model.eval()
    with torch.no_grad():
        data, label = data.to(DEVICE), label.to(DEVICE)
        pred, _ = model(data)

        correct_cnt = (torch.max(pred, 1)[1] == label).sum().item()
        sample_cnt = data.shape[0]
        accuracy = correct_cnt / sample_cnt

    return accuracy


def train(model, source_data, source_label, val_data, val_label, target_data, target_label, batch_size=1080, epochs=500, lr=1e-3, path='test', format='jpg'):
    train_dataset = TensorDataset(source_data, source_label)
    val_dataset = TensorDataset(val_data, val_label)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr = lr)
    # lr_scheduler = LambdaLR(optimizer, lr_lambda)


    model.to(DEVICE)

    clf_loss_history = []
    mmd_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    test_accuracy_history = []

    best_acc = 0

    for epoch in tqdm(range(epochs)):
        model.train()

        optimizer.zero_grad()

        source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
        target_data = target_data.to(DEVICE)
        val_data = val_data.to(DEVICE)

        train_target_data = val_data
        train_target_data = train_target_data.to(DEVICE)

        preds, source = model(source_data)

        # compute loss
        clf_loss = loss_fn(preds, source_label)
        loss = clf_loss

        clf_loss_history.append(loss.item())

        correct_cnt = (torch.max(preds, 1)[1]== source_label).sum().item()
        accuracy = correct_cnt / source_data.shape[0]
        train_accuracy_history.append(accuracy)

        loss.backward()
        optimizer.step()

        val_accuracy = test(val_data, val_label, model)
        _, val = model(val_data)
        val_accuracy_history.append(val_accuracy)
        if val_accuracy > best_acc:
            best_acc = val_accuracy

        # visualize the data distribution
        # if epoch == epochs - 1:
        #     print(f'accuracy: {val_accuracy}')
        #     tsne_visulization(source.cpu().detach().numpy(), val.cpu().detach().numpy(), path='origin', format='eps')

    # # visualize the accuracy curve
    # plt.figure(figsize=(16, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_accuracy_history, label='Train Accuracy')
    # plt.plot(val_accuracy_history, label='Validation Accuracy')

    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title('Accuracy Curve')
    # plt.grid(True)
    # plt.show()


    return best_acc, val_accuracy_history


switch = False
switch = True 

model = EEGNet_ReLU(n_output=2)
source_data, source_label, val_data, val_label, target_data, target_label = read_bci_data()
print(source_data.min(), source_data.max())

if switch:
    target_data, target_label = source_data, source_label
    source_data, source_label = val_data, val_label
    val_data, val_label = target_data, target_label

best_acc, val_accuracy_history = train(model, source_data, source_label, val_data, val_label, target_data, target_label, batch_size=1080, epochs=100, lr=1e-2, path='demo')
print(best_acc)
#test_accuracy = test(target_data, target_label, model)
#print(test_accuracy)


plt.plot(range(len(val_accuracy_history)), val_accuracy_history)

plt.savefig('./acc_curve.png')
plt.close()

