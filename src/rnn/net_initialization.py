import math
import numpy as np
import torch
from torch.nn import init


def mean_initialization() -> None:
    """
    initialize weights for main training with the average of the params of the other training
    RESTRICTED
    """


def max_initialization() -> None:
    """
    initialize weights for main training with the absolute maximum of the params of the other training
    RESTRICTED
    """


def rank_initialization() -> None:
    """
    initialize weights for main training with the absolute maximum ranking of the params of the other training
    RESTRICTED
    """
