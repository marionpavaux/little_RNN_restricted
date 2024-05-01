from typing import List

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
import numpy as np

from rnn import BATCH_SIZE, GPU, LR, WORTH_MP
import data

from .interpret import display_net_state, plot, plot_loss_trajectory
from .save import load_data, load_checkpoint, dump_checkpoint, dump_loss


def regul_L1(net: nn.Module) -> float:
    """
    Encourage sparsity of weights
    """
    rl1 = (
        torch.sum(torch.abs(net.get_weights()["J"]))
        + torch.sum(torch.abs(net.get_weights()["W"]))
        + torch.sum(torch.abs(net.get_weights()["B"]))
    )
    return rl1


def regul_L2(net: nn.Module) -> float:
    """
    Homogeanize weights
    """
    rl2 = (
        torch.sum(torch.square(net.get_weights()["B"]))
        + torch.sum(torch.square(net.get_weights()["J"]))
        + torch.sum(torch.square(net.get_weights()["W"]))
    )
    return rl2


def regul_FR(activity: torch.Tensor) -> float:
    """
    Encourage sparsity of firing rates
    """
    rfr = torch.mean(torch.square(activity))
    return rfr


def train_epoch(
    net: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    criterion,
    optimizer,
    beta1: float = None,
    beta2: float = None,
    beta_FR: float = None,
) -> float:
    net.train()
    batch_losses = []
    for input_batch, label_batch in dataloader:
        input_batch = torch.fliplr(torch.rot90(input_batch, k=-1)).to(device, non_blocking=True)
        label_batch = torch.fliplr(torch.rot90(label_batch, k=-1)).to(device, non_blocking=True).float()
        optimizer.zero_grad()
        output, activity = net(input_batch)
        loss = criterion(output, label_batch)

        if beta1 != None:
            loss += beta1 * regul_L1(net)
        if beta2 != None:
            loss += beta2 * regul_L2(net)
        if beta_FR != None:
            loss += beta_FR * regul_FR(activity)
        loss.backward()
        optimizer.step()  # Does the update

        batch_losses.append(loss.item())

    running_loss = np.mean(batch_losses)
    return running_loss


def test_epoch(
    net: nn.Module,
    device: torch.device,
    dataloader: DataLoader,
    criterion,
    beta1: float = None,
    beta2: float = None,
    beta_FR: float = None,
) -> float:
    net.eval()
    batch_losses = []
    for input_batch, label_batch in dataloader:
        input_batch = torch.fliplr(torch.rot90(input_batch, k=-1)).to(device, non_blocking=True)
        label_batch = torch.fliplr(torch.rot90(label_batch, k=-1)).to(device, non_blocking=True).float()
        output, activity = net(input_batch)
        loss = criterion(output, label_batch)

        if beta1 != None:
            loss += beta1 * regul_L1(net)
        if beta2 != None:
            loss += beta2 * regul_L2(net)
        if beta_FR != None:
            loss += beta_FR * regul_FR(activity)

        batch_losses.append(loss.item())

    running_loss = np.mean(batch_losses)
    return running_loss


def train() -> None:
    """
    RESTRICTED
    """


def train_per_configs() -> None:
    """
    RESTRICTED
    """
