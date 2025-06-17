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
from losses import *
# MMD, JMMD, JPMMD, DJP-MMD
# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA = 1

def mmd_loss(x_src, y_src, x_tar, y_pseudo, mmd_type):
    if mmd_type == 'mmd':
        return rbf_mmd(x_src, x_tar, SIGMA)
    elif mmd_type == 'jmmd':
        return rbf_jmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
    elif mmd_type == 'jpmmd':
        return rbf_jpmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
    elif mmd_type == 'djpmmd':
        return rbf_djpmmd(x_src, y_src, x_tar, y_pseudo, SIGMA)
    elif mmd_type == 'CS':
        return CS(x_src, x_tar, sigma=SIGMA, median_sigma=False)
    elif mmd_type == 'CCS':
        return CS(x_src, x_tar, sigma=SIGMA)  + CCS(x_src, x_tar, y_src.float(), y_pseudo.float(), sigma=SIGMA)

def test(data, label, model):
    model.eval()
    with torch.no_grad():
        data, label = data.to(DEVICE), label.to(DEVICE)
        pred, _,_, _ = model(data, data)

        correct_cnt = (torch.max(pred, 1)[1] == label).sum().item()
        sample_cnt = data.shape[0]
        accuracy = correct_cnt / sample_cnt

    return accuracy

# def lr_lambda(epoch):
#     if epoch < 10:
#         return 0.1
#     elif 10 <= epoch < 100:
#         return 0.01
#     else:
#         lr = 0.01 ** ((epoch // 100) + 1)
#         return 1e-4 if lr < 1e-4 else lr


def train(model, source_data, source_label, val_data, val_label, target_data, target_label, mmd_type='djpmmd', lambda_factor=0.5, batch_size=1080, epochs=500, lr=1e-1, path='test', format='jpg'):
    train_dataset = TensorDataset(source_data, source_label)
    val_dataset = TensorDataset(val_data, val_label)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr = lr)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5, weight_decay=5e-4)
    #lr_scheduler = LambdaLR(optimizer, lr_lambda)

    model.to(DEVICE)

    y_pse = torch.zeros(1080).long().cuda()

    clf_loss_history = []
    mmd_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []
    test_accuracy_history = []

    best_acc = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        #lr_scheduler.step()
        optimizer.zero_grad()

        source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
        target_data = target_data.to(DEVICE)
        val_data, val_label = val_data.to(DEVICE), val_label.to(DEVICE)

        y_src, y_tgt, features_src, features_tar = model(source_data, val_data)

        # compute loss
        clf_loss = loss_fn(y_src, source_label)

        y_pse = y_tgt.detach().max(1)[1]
        #y_src, y_tgt = F.normalize(y_src, p=2.0, dim=-1), F.normalize(y_tgt, p=2.0, dim=-1)
        y_tgt = F.softmax(y_tgt,dim=1)
        y1 = torch.zeros(source_label.shape[0], 2).cuda().scatter(1, source_label.unsqueeze(1),1).detach()
        #y1 = y_src
        loss_mmd = mmd_loss(features_src, y1, features_tar, y_tgt, mmd_type) # use true lable in the source domain to compute loss

        if epoch > 10:
            loss = clf_loss + lambda_factor * loss_mmd
        else:
            loss = clf_loss

        loss.backward()
        optimizer.step()
        

        #model.eval()
        #y_pse, _,_, _ = model(val_data, val_data)
        # y_pse = model(val_data, val_data)
        # y_pse = y_pse.detach().max(1)[1]

        clf_loss_history.append(loss.item())
        mmd_loss_history.append(loss_mmd.item())

        correct_cnt = (torch.max(y_src, 1)[1] == source_label).sum().item()
        accuracy = correct_cnt / source_data.shape[0]
        train_accuracy_history.append(accuracy)




        val_accuracy = test(val_data, val_label, model)
        val_accuracy_history.append(val_accuracy)
        if val_accuracy > best_acc:
            best_acc = val_accuracy


    discrepency = np.nanmean(mmd_loss_history)
    print(f"lambda={lambda_factor}, accuracy={best_acc}, mean discrepency={discrepency}")

    return best_acc, discrepency, val_accuracy_history
