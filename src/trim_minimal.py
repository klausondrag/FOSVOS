from timeit import default_timer

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix


def load(x: str) -> np.ndarray:
    x = np.load('{}.npy'.format(x))
    x = x[0, 0, :, :]
    print(x.shape)
    return x


x = load('x')
y = load('y')

x = np.pad(x, 1, 'constant')
x = x.reshape((-1))

x = x.reshape((-1, 1))
y = y.reshape((-1, 1))

y = y[0, :]


time_start = default_timer()
rho = 5
alpha = 1.8
lmbda = 4

lmbda = 1 / lmbda
y = np.reshape(y, newshape=(1, -1))  # make sure y is a row vector
N = x.shape[0]

if y.shape[1] != x.shape[1]:
    raise ValueError("Dimensions of input data, X & y, are not consistent.")

print(default_timer() - time_start)


time_start = default_timer()
Q = csr_matrix(x)
Q = Q @ Q.T
Q *= lmbda
print(default_timer() - time_start)


time_start = default_timer()
q = -lmbda * x @ y.T
q = csr_matrix(1 / 2 + np.vstack([q, -q]))

P = csr_matrix((x.shape[0], 0))
P = P.T
P = sparse.hstack([P, -P])

c = csr_matrix((0, 1))
print(default_timer() - time_start)


time_start = default_timer()
tmp = csr_matrix([[1, -1], [-1, 1]])
Q = sparse.kron(tmp, Q)
print(default_timer() - time_start)


time_start = default_timer()
A = sparse.vstack([P, -sparse.eye(2 * N, 2 * N)])
c = sparse.vstack([c, csr_matrix((2 * N, 1))])
print(default_timer() - time_start)


time_start = default_timer()
L = Q + rho * A.T @ A
L = csr_matrix(np.linalg.cholesky(L.toarray()))
U = L.T
print(default_timer() - time_start)


time_start = default_timer()
z = csr_matrix((c.shape[0], 1))
x = csr_matrix((2 * N, 1))
u = csr_matrix((c.shape[0], 1))
print(default_timer() - time_start)


