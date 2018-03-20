import timeit
from pathlib import Path

import numpy as np
from scipy.linalg import circulant
import torch
from torch.autograd import Variable

from networks.trimmer import Trimmer
from util import io_helper, gpu_handler
from util.logger import get_logger
from util.network_provider import ResNetOnlineProvider
from util.settings import OnlineSettings
from config.mypath import Path as P

log = get_logger(__file__)


def _trim_layer(X, y, rho, alpha, lmbda, n_iterations=5, n_inner_iterations=100):
    lmbda = 1 / lmbda
    y = np.reshape(y, newshape=(1, -1))  # make sure y is a row vector
    N = X.shape[0]

    if y.shape[1] != X.shape[1]:
        raise ValueError("Dimensions of input data, X & y, are not consistent.")

    # Omega = np.where(y > 1e-6)[1]
    Omega = list(range(y.shape[1]))
    # Omega_c = np.where(y <= 1e-6)[1]
    Omega_c = []

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

    z = torch.from_numpy(z)
    x = torch.from_numpy(x)
    u = torch.from_numpy(u)

    trimmer = Trimmer(L=L, U=U, A=A, q=q, c=c, rho=rho, alpha=alpha, n_iterations=n_inner_iterations)
    cnt = 0
    for cnt in range(n_iterations):
        dx, x, z, u = trimmer.forward(z=z, u=u)
        if np.linalg.norm(dx) < 1e-3:
            break

    w = x[0:N] - x[N:]
    w[np.abs(w) < 1e-3] = 0
    w = w.squeeze()

    return w, cnt


def trim_network(X, Y, W):
    # X = X0
    # Y = X1
    # X = X.transpose()
    # Y = Y.transpose()

    # append 1 to the last row of X for model y = ReLU(W'x+b)
    # X = np.append(X, np.ones(shape=(1, X.shape[1])), axis=0)

    # original_W = layers['layer_stages.0.0.conv1.weight']
    original_W = W
    # original_b = layers['layer_stages.0.0.conv1.bias']
    refined_W = original_W.copy()
    # refined_b = original_b.copy()
    total_time = 0
    for i in range(Y.shape[0]):
        start = timeit.default_timer()
        w_tf, num_iter_tf = _trim_layer(X=X, y=Y[i, :], rho=5, alpha=1.8, lmbda=4)
        elapsed_time = timeit.default_timer() - start
        print('execution time:', elapsed_time)
        # refined_W[:, i] = w_tf[:-1]
        refined_W[:, i] = w_tf
        # refined_b[0, i] = w_tf[-1]

        total_time += elapsed_time

    print('number of non-zero values in the original weight matrix = ', np.count_nonzero(original_W == 0))
    print('number of non-zero values in the refined weight matrix = ', np.count_nonzero(refined_W == 0))
    print('total elapsed time = ', total_time)


def transform(x, y, w):
    x = np.pad(x, 1, 'constant')
    n_rows_in_x, n_columns_in_x = x.shape
    x = x.reshape((-1))
    n_elements_in_x = len(x)
    n_rows_in_w, n_columns_in_w = w.shape

    f = np.zeros((n_rows_in_x, n_columns_in_x))
    f[:n_rows_in_w, :n_columns_in_w] = w
    f = f.reshape(-1)

    w_ = circulant(f).T

    stride = 1
    n_rows_in_output = int(np.floor((n_rows_in_x - n_rows_in_w) / stride + 1))
    n_columns_in_output = int(np.floor((n_columns_in_x - n_columns_in_w) / stride + 1))

    indices = np.zeros(w_.shape[0], dtype=bool)
    for index_row, index_start in enumerate(range(0, indices.shape[0], n_columns_in_x)):
        if index_row >= n_rows_in_output:
            break
        index_end = index_start + n_columns_in_output
        indices[index_start:index_end] = [True] * n_columns_in_output

    if n_rows_in_output * n_columns_in_output != np.sum(indices):
        raise Exception('Sum of indices should match the values in output')

    w_result = w_[indices, :]

    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    return x, y, w_result


def main():
    sequence_name = 'blackswan'
    variant_offline = 11
    variant_online = 11
    resnet_version = 18

    gpu_handler.select_gpu(1)

    db_root_dir = P.db_root_dir()

    save_dir_models = Path('models')
    save_dir_models.mkdir(parents=True, exist_ok=True)
    save_dir_results = Path('results')
    save_dir_results.mkdir(parents=True, exist_ok=True)

    settings = OnlineSettings(is_training=False, is_testing=True, start_epoch=0, n_epochs=10000,
                              avg_grad_every_n=5, snapshot_every_n=10000, is_testing_while_training=False,
                              test_every_n=5, batch_size_train=1, batch_size_test=1, is_visualizing_network=False,
                              is_visualizing_results=False, offline_epoch=240,
                              variant_offline=variant_offline, variant_online=variant_online, eval_speeds=False)

    net_provider = ResNetOnlineProvider(name='resnet{0}'.format(resnet_version), save_dir=save_dir_models,
                                        settings=settings, variant_offline=variant_offline,
                                        variant_online=variant_online, version=resnet_version)
    net_provider.load_network_test(sequence=sequence_name)
    net = net_provider.network

    data_loader = io_helper.get_data_loader_test(db_root_dir, settings.batch_size_train, sequence_name)
    for minibatch_index, minibatch in enumerate(data_loader):
        inputs, gts = minibatch['image'], minibatch['gt']
        inputs, gts = Variable(inputs), Variable(gts)
        inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])
        X, Y = net.forward(inputs)
        break

    W = net.state_dict()['side_prep.0.weight']

    def to_np(x):
        return x.cpu().data.numpy()

    X = to_np(X)
    Y = to_np(Y)
    W = W.cpu().numpy()

    def get_first(x):
        return x[0, 0, :, :]

    X = get_first(X)
    Y = get_first(Y)
    W = get_first(W)

    X, Y, W = transform(X, Y, W)

    trim_network(X, Y, W)

    log.info('done')


if __name__ == '__main__':
    main()
