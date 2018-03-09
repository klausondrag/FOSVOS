from typing import Optional, Tuple

import torch
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

    def __init__(self, L: torch.FloatTensor, U: torch.FloatTensor, A: torch.FloatTensor, q: torch.FloatTensor,
                 c: torch.FloatTensor, rho: Optional[float] = 5, alpha: Optional[float] = 1.8,
                 n_iterations: Optional[int] = 10) -> None:
        super(NetTrim, self).__init__()

        self._L = L
        self._U = U
        self._A = A
        self._q = q
        self._c = c
        self._rho = rho
        self._alpha = alpha
        self._n_iterations = n_iterations

        self._relu = nn.ReLU(inplace=True)

    def forward(self, z: torch.FloatTensor, u: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                           torch.FloatTensor, torch.FloatTensor]:
        _1 = self._c - u + z  # c-u-z
        _2 = self._rho * torch.t(self._A) @ _1  # rho*A'*(c-u-z)
        _3 = _2 - self._q  # rho*A'*(c-u-z)-q
        _4 = torch.tril(self._L, _3)
        x = torch.triu(self.U, _4)
        x_prev = x

        # Ax = A @ x
        # c_AX = c - Ax
        #
        # # update z
        # _1 = self.alpha * c_AX  # alpha * (c - A * x)
        # _2 = (1 - self.alpha) * z  # (1 - alpha) * z_prev
        # tmp = _1 + _2 - u  # alpha*(c-A*x) + (1-alpha)*z_prev - u
        # z = self.relu(tmp)  # z = max(alpha*(c-A*x) + (1-alpha)*z_prev, 0)
        #
        # # update u
        # u = z - tmp  # z - (alpha*(c-A*x) + (1-alpha)*z_prev - u)
        #
        # # update x
        # x_prev = x
        # _1 = c - u - z  # c-u-z
        # _2 = self.rho * torch.t(A) @ _1  # rho*A'*(c-u-z)
        # _3 = _2 - q  # rho*A'*(c-u-z)-q
        # _4 = torch.tril(L, _3)
        # x = torch.triu(U, _4)

        # return dx, x, z, u
