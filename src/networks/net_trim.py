from typing import Optional

import torch.nn as nn
from torch.autograd import Variable

from util.logger import get_logger

log = get_logger(__file__)


class NetTrim(nn.Module):
    """
        Original paper by A. Aghasi, A. Abdi, N. Nguyen, and J. Romberg, "Net-Trim: Convex Pruning of Deep Neural
        Networks with Performance Guarantee," NIPSs 2017
        PyTorch implementation of the paper above 18f
        TensorFlow implementation provided by the authors at https://github.com/DNNToolBox/Net-Trim-v1
    """

    def __init__(self, rho: Optional[float] = 5, alpha: Optional[float] = 1.8, lmbda: Optional[float] = 4,
                 epsilon: Optional[float] = 1e-6) -> None:
        super(NetTrim, self).__init__()

        self.rho = rho
        self.alpha = alpha
        self.lmbda = 1 / lmbda
        self.epsilon = epsilon

    def forward(self, X: Variable, y: Variable):
        y = y.view((1, -1))
        n_samples = X.shape[0]

        if y.shape[1] != X.shape[1]:
            raise ValueError("Dimensions of input data, X & y, are not consistent.")

        # pytorch way of doing it?
        Omega = np.where(y > self.epsilon)[1]
        Omega_c = np.where(y <= self.epsilon)[1]
