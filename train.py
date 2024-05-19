from vivit import *

from torch import optim, save
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, IterableDataset
import matplotlib.pyplot as plt
from os import path, mkdir

import torch
import numpy as np


def collate(batch):
    
    raw = torch.stack([x[0] for x in batch])
    labels = torch.tensor([x[1] for x in batch]).reshape(-1)
    return raw, labels

def train(model, device, train_ds, val_ds, train_opts, exp_dir=None):
    """
    Fits a categorization model on the provided data

    Arguments
    ---------
    model: (A pytorch module), the categorization model to train
    train_ds: (TensorDataset), the examples (images and labels) in the training set
    val_ds: (TensorDataset), the examples (images and labels) in the validation set
    train_opts: (dict), the training schedule. Read the assignment handout
                for the keys and values expected in train_opts
    exp_dir: (string), a directory where the model checkpoints should be saved (optional)

    """

    num_epochs = train_opts["num_epochs"]
    train_dl = DataLoader(train_ds, batch_size=train_opts['batch_size'], shuffle=True, num_workers=8, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=train_opts['batch_size'], shuffle=True, num_workers=8, collate_fn=collate)

    # we will use stochastic gradient descent
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_opts["lr"],
        momentum=train_opts['momentum'],

    )

    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        train_opts['num_epochs'],
        # train_opts['step_size'],
        # train_opts['gamma']
    )

    #loss function
    criterion = CrossEntropyLoss()
    
    num_tr = len(train_ds.paths)
    num_val = len(val_ds.paths)


    # track the training metrics
    epoch_loss_tr = []
    epoch_loss_val = []
    epoch_acc_val = []
    epoch_acc_tr = []
    for epoch in range(num_epochs):

        # training phase
        model.train()
        tr_loss, train_acc = fit(model, device, train_dl, criterion, optimizer)
        lr_scheduler.step()
        train_acc = train_acc / num_tr
        epoch_loss_tr.append(tr_loss)
        epoch_acc_tr.append(train_acc)

        val_loss = None
        val_acc = None
        with torch.no_grad():
            model.eval()
            val_loss, val_acc = fit(model, device, val_dl, criterion)
            val_acc = val_acc / num_val
            epoch_loss_val.append(val_loss)
            epoch_acc_val.append(val_acc)

        # it is always good to report the training metrics at the end of every epoch
        print(f"[{epoch + 1}/{num_epochs}: tr_loss {tr_loss:.4} val_loss {val_loss:.4}"
              f"t_acc {train_acc:.2%} val_acc {val_acc:.2%}]")

        # save model checkpoint if exp_dir is specified
        if exp_dir:
            if path.exists(exp_dir):
                save(model.state_dict(), path.join(exp_dir, f"checkpoint_{epoch + 1}.pt"))
            else:
                try:
                    mkdir(exp_dir)
                    save(model.state_dict(), path.join(exp_dir, f"checkpoint_{epoch + 1}.pt"))
                except FileNotFoundError:
                    pass

    # plot the training metrics at the end of training
    plot(epoch_loss_tr)


def fit(model, device, data_loader, criterion, optimizer=None):
    """
    Executes a training (or validation) epoch
    epoch: (int), the training epoch. This parameter is used by the learning rate scheduler
    model: (a pytorch module), the categorization model begin trained
    data_loader: (DataLoader), the training or validation set
    criterion: (CrossEntropy) for this task. The objective function
    optimizer: (SGD) for this task. The optimization function (optional)
    scheduler: (StepLR) for this schedule. The learning rate scheduler (optional)

    Return
    ------
    epoch_loss: (float), the average loss on the given set for the epoch
    epoch_acc: (float), the categorization accuracy on the given set for the epoch

    """
    step = 0
    epoch_loss = epoch_acc = 0
    for i, (mini_x, mini_y) in enumerate(data_loader):
        mini_x = mini_x.to(device)
        mini_y = mini_y.to(device)

        #print(mini_x.shape, mini_y.shape)

        pred = model(mini_x)

        loss = criterion(pred, mini_y)

        epoch_loss += loss.item()
        epoch_acc += mini_y.eq(pred.argmax(dim=1)).sum().item()
        
        if i % 100 == 0:
            print("Mini loss {:.4f} acc {:.4f}".format(loss, mini_y.eq(pred.argmax(dim=1)).sum().item()))

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        step += 1

    epoch_loss = epoch_loss / len(data_loader)

    return epoch_loss, epoch_acc


def plot(loss_tr):
    """
    plots the training metrics

    Arguments
    ---------
    loss_tr: (list), the average epoch loss on the training set for each epoch
    """
    figure, ax = plt.subplots(1, 1, figsize=(12, 6))
    n = [i + 1 for i in range(len(loss_tr))]
    loss_tr = [x * 100 for x in loss_tr]

    ax.plot(n, loss_tr, 'bs-', markersize=3, label="train")
    ax.legend(loc="upper right")
    ax.set_title("Losses")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")