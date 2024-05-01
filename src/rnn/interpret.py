import os
from typing import Union
import time
import psutil

import _pickle as cPickle
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler

from sklearn.decomposition import PCA
import torch
import torch.nn as nn

from data import MUSCLES, FS, PATH
from .predict import predict
from utils import plot_electrode_activation


matplotlib.use("Agg")
plt.style.use("dark_background")


def display_net_state(optimizer: torch.optim, training_loss: float, epoch: int, testing_loss: float = None) -> None:
    """
    display net state on terminal

    :param optimizer
    :param running_loss: actual loss
    :param epoch: actual epoch
    """
    optim_param = optimizer.param_groups[0]

    to_print = "{}   proc {}   epoch {}   train_loss {:0.2f}".format(time.ctime(), os.getpid(), epoch, training_loss)
    if testing_loss is not None:
        to_print += "   test_loss {:0.2f}".format(testing_loss)

    if (
        "exp_avg" in optimizer.state[optim_param["params"][1]].keys()
        and "exp_avg" in optimizer.state[optim_param["params"][3]].keys()
        and "exp_avg" in optimizer.state[optim_param["params"][5]].keys()
    ):
        # get averaged B step size
        state_B = optimizer.state[optim_param["params"][1]]
        unbiased_exp_avg = state_B["exp_avg"] / (1 - optim_param["betas"][0] ** state_B["step"])
        unbiased_exp_avg_sq = state_B["exp_avg_sq"] / (1 - optim_param["betas"][1] ** state_B["step"])
        lr_B = np.format_float_scientific(
            torch.mean(
                optim_param["lr"]
                / (1 - optim_param["betas"][0] ** state_B["step"])
                * unbiased_exp_avg
                / (torch.sqrt(unbiased_exp_avg_sq) + optim_param["eps"])
            ).item(),
            precision=3,
        )

        # get averaged J step size
        state_J = optimizer.state[optim_param["params"][3]]
        unbiased_exp_avg = state_J["exp_avg"] / (1 - optim_param["betas"][0] ** state_J["step"])
        unbiased_exp_avg_sq = state_J["exp_avg_sq"] / (1 - optim_param["betas"][1] ** state_J["step"])
        lr_J = np.format_float_scientific(
            torch.mean(
                optim_param["lr"]
                / (1 - optim_param["betas"][0] ** state_B["step"])
                * unbiased_exp_avg
                / (torch.sqrt(unbiased_exp_avg_sq) + optim_param["eps"])
            ).item(),
            precision=3,
        )

        # get averaged W step size
        state_W = optimizer.state[optim_param["params"][5]]
        unbiased_exp_avg = state_W["exp_avg"] / (1 - optim_param["betas"][0] ** state_W["step"])
        unbiased_exp_avg_sq = state_W["exp_avg_sq"] / (1 - optim_param["betas"][1] ** state_W["step"])
        lr_W = np.format_float_scientific(
            torch.mean(
                optim_param["lr"]
                / (1 - optim_param["betas"][0] ** state_B["step"])
                * unbiased_exp_avg
                / (torch.sqrt(unbiased_exp_avg_sq) + optim_param["eps"])
            ).item(),
            precision=3,
        )

        to_print += "   lrB {}   lrJ {}   lrW {}".format(lr_B, lr_J, lr_W)

    to_print += "   RAM {}%".format(psutil.virtual_memory()[2])
    print(to_print)


"""
RESTRICTED FUNCTIONS WERE THERE
"""


def plot_loss_trajectory(training_config: Union[str, int], ID) -> None:
    """
    plot loss trajectory of the network

    :param training_config: int defining configuration or String 'main'
    """

    training_losses = cPickle.load(
        open(
            f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{(' ' + str(training_config)) if training_config!='main' else ''}/training_loss.pkl",
            "rb",
        )
    )
    fig = plt.figure()
    plt.plot(np.arange(len(training_losses)), training_losses, "o-", label="train")
    if training_config == "main":
        testing_losses = cPickle.load(
            open(
                f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{(' ' + str(training_config)) if training_config!='main' else ''}/testing_loss.pkl",
                "rb",
            )
        )
        plt.plot(np.arange(len(testing_losses)), testing_losses, "x-", color="#fa525b", label="test")
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.legend(framealpha=0)
    plt.box(False)
    plt.savefig(
        f"{PATH}/{ID}/{'Sub' if training_config!='main' else 'Main'}-training{(' ' + str(training_config)) if training_config!='main' else ''}/Loss.png",
        transparent=True,
    )
    plt.close(fig)
