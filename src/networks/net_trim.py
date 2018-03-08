from typing import Optional

import torch.nn as nn
from torch.autograd import Variable

from util.logger import get_logger

log = get_logger(__file__)


class NetTrim(nn.Module):
    """
        Original paper by A. Aghasi, A. Abdi, N. Nguyen, and J. Romberg, "Net-Trim: Convex Pruning of Deep Neural
        Networks with Performance Guarantee," NIPSs 2017
        TensorFlow implementation provided by the authors at https://github.com/DNNToolBox/Net-Trim-v1
    """

    def __init__(self, rho: Optional[float] = 5, alpha: Optional[float] = 1.8) -> None:
        super(NetTrim, self).__init__()

        self.rho = rho
        self.alpha = alpha

    def forward(self, L: Variable, U: Variable, A: Variable, q: Variable, c: Variable, x: Variable):
        Ax = A.matmul(x)
        c_AX = c - Ax
