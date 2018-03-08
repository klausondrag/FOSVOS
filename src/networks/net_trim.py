from typing import List, Tuple, Union, Callable

import torch
import torch.nn as nn

from util.logger import get_logger

log = get_logger(__file__)


class NetTrim(nn.Module):
    """
        Original paper by A. Aghasi, A. Abdi, N. Nguyen, and J. Romberg, "Net-Trim: Convex Pruning of Deep Neural
        Networks with Performance Guarantee," NIPSs 2017
        PyTorch implementation of the paper above 18f
        TensorFlow implementation provided by the authors at https://github.com/DNNToolBox/Net-Trim-v1
    """

    def __init__(self, rho: float, alpha: float, lmbda: float):
        pass

    def forward(self, x):
        pass
