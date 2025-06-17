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

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGMA = 1


# init weights
def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)


def lr_lambda(epoch):
    if epoch < 100:
        return 0.1
    else:
        lr = 0.1 ** ((epoch // 100) + 1)
        return 1e-4 if lr < 1e-4 else lr


def test(data, label, model):
    model.eval()
    with torch.no_grad():
        data, label = data.to(DEVICE), label.to(DEVICE)
        pred, _, _, _ = model(data, data)

        correct_cnt = (torch.max(pred, 1)[1] == label).sum().item()
        sample_cnt = data.shape[0]
        accuracy = correct_cnt / sample_cnt

    return accuracy


def train(model, source_data, source_label, val_data, val_label, target_data, target_label, lambda_factor=0.5, batch_size=1080, epochs=500, lr=1e-3, path='test', format='jpg'):
    train_dataset = TensorDataset(source_data, source_label)
    val_dataset = TensorDataset(val_data, val_label)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr = lr)
    lr_scheduler = LambdaLR(optimizer, lr_lambda)


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

        preds, mmd_loss, source, val = model(source_data, train_target_data)

        # compute loss
        clf_loss = loss_fn(preds, source_label)
        if epoch > 10:
            loss = clf_loss + lambda_factor * mmd_loss
        else:
            loss = clf_loss

        clf_loss_history.append(loss.item())
        mmd_loss_history.append(mmd_loss.item())

        correct_cnt = (torch.max(preds, 1)[1]== source_label).sum().item()
        accuracy = correct_cnt / source_data.shape[0]
        train_accuracy_history.append(accuracy)

        loss.backward()
        optimizer.step()
        #lr_scheduler.step()

        val_accuracy = test(val_data, val_label, model)
        val_accuracy_history.append(val_accuracy)
        if val_accuracy > best_acc:
            best_acc = val_accuracy

        # if epoch == epochs - 1:
            # tsne_visulization(source.cpu().detach().numpy(), val.cpu().detach().numpy(), path='original.jpg')

    discrepency = np.mean(mmd_loss_history)

    # visualize the accuracy curve
    # plt.figure(figsize=(16, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(train_accuracy_history, label='Train Accuracy')
    # plt.plot(val_accuracy_history, label='Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.title('Accuracy Curve')
    # plt.grid(True)

    print(f"lambda={lambda_factor}, accuracy={best_acc}, mean discrepency={discrepency}")

    return best_acc, discrepency, val_accuracy_history