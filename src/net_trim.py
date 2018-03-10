import numpy as np
import torch

from networks.trimmer import Trimmer


def _trim_layer(X, y, rho, alpha, lmbda, num_iterations=100):
    lmbda = 1 / lmbda
    y = np.reshape(y, newshape=(1, -1))  # make sure y is a row vector
    N = X.shape[0]

    if y.shape[1] != X.shape[1]:
        raise ValueError("Dimensions of input data, X & y, are not consistent.")

    Omega = np.where(y > 1e-6)[1]
    Omega_c = np.where(y <= 1e-6)[1]

    Q = lmbda * np.matmul(X[:, Omega], np.transpose(X[:, Omega]))
    q = -lmbda * np.matmul(X, np.transpose(y))
    P = X[:, Omega_c]
    P = P.transpose()
    c = np.zeros((len(Omega_c), 1))

    Q = np.kron([[1, -1], [-1, 1]], Q)
    q = 1 / 2 + np.append(q, -q, axis=0)
    P = np.append(P, -P, axis=1)

    A = np.append(P, -np.eye(2 * N, 2 * N), axis=0)
    c = np.append(c, np.zeros((2 * N, 1)), axis=0)

    # The ADMM part of the code
    L = np.linalg.cholesky(Q + rho * np.matmul(A.transpose(), A))
    U = L.transpose()

    z = np.zeros((len(c), 1))
    x = np.zeros((2 * N, 1))
    u = np.zeros((len(c), 1))

    L = torch.from_numpy(L)
    U = torch.from_numpy(U)
    A = torch.from_numpy(A)
    q = torch.from_numpy(q)
    c = torch.from_numpy(c)

    trimmer = Trimmer(L=L, U=U, A=A, q=q, c=c, rho=rho, alpha=alpha)
    cnt = 0
    for cnt in range(num_iterations):
        dx, x, z, u = trimmer.forward(z=z, u=u)
        if np.linalg.norm(dx) < 1e-3:
            break

    w = x[0:N] - x[N:]
    w[np.abs(w) < 1e-3] = 0
    w = w.squeeze()

    return w, cnt
