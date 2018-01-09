import sys
import timeit
from pathlib import Path
from collections import namedtuple

import scipy.misc as sm
from tensorboardX import SummaryWriter
import numpy as np

from torch.autograd import Variable
import torch.optim as optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from networks.osvos_vgg import OSVOS_VGG
from networks.osvos_resnet import OSVOS_RESNET
from layers.osvos_layers import class_balanced_cross_entropy_loss

from util import gpu_handler, io_helper
from util.logger import get_logger
from config.mypath import Path as P
from util.network_provider import NetworkProvider

if P.is_custom_pytorch():
    sys.path.append(P.custom_pytorch())
if P.is_custom_opencv():
    sys.path.insert(0, P.custom_opencv())
gpu_handler.select_gpu_by_hostname()

log = get_logger(__file__)

Settings = namedtuple('Settings', ['start_epoch', 'n_epochs', 'avg_grad_every_n', 'snapshot_every_n',
                                   'is_testing_while_training', 'test_every_n', 'batch_size_train', 'batch_size_test',
                                   'is_loading_vgg_caffe', 'is_visualizing_network'])

settings = Settings(start_epoch=0, n_epochs=240, avg_grad_every_n=10, snapshot_every_n=40,
                    is_testing_while_training=False, test_every_n=5, batch_size_train=1, batch_size_test=1,
                    is_loading_vgg_caffe=False, is_visualizing_network=False)


def train_and_test(net_provider: NetworkProvider, settings: Settings, is_training: bool = True,
                   is_testing: bool = True) -> None:
    if is_training:
        net_provider.load_network_train()
        data_loader_train = io_helper.get_data_loader_train(db_root_dir, settings.batch_size_train)
        data_loader_test = io_helper.get_data_loader_test(db_root_dir, settings.batch_size_test)
        optimizer = net_provider.get_optimizer()
        summary_writer = _get_summary_writer()

        io_helper.write_settings(save_dir_models, net_provider.name, settings._asdict())
        _train(net_provider, data_loader_train, data_loader_test, optimizer, summary_writer, settings.start_epoch,
               settings.n_epochs, settings.avg_grad_every_n, settings.snapshot_every_n,
               settings.is_testing_while_training, settings.test_every_n)

    if is_testing:
        net_provider.load_network_test()
        data_loader = io_helper.get_data_loader_test(db_root_dir, settings.batch_size_test)

        _test(net_provider, data_loader, save_dir_results)

    if settings.is_visualizing_network:
        io_helper.visualize_network(net_provider.network)


def _get_summary_writer() -> SummaryWriter:
    return io_helper.get_summary_writer(save_dir_models, comment='-parent')


def _train(net_provider: NetworkProvider, data_loader_train: DataLoader, data_loader_test: DataLoader,
           optimizer: Optimizer, summary_writer: SummaryWriter, start_epoch: int, n_epochs: int, avg_grad_every_n: int,
           snapshot_every_n: int, is_testing_while_training: bool, test_every_n: int) -> None:
    log.info('Start of Parent Training')

    net = net_provider.network

    n_samples_train = len(data_loader_train)
    n_samples_test = len(data_loader_test)
    running_loss_train = [0] * 5
    running_loss_test = [0] * 5
    loss_train = []
    loss_test = []
    counter_gradient = 0

    log.info('Training Network')
    for epoch in range(start_epoch, n_epochs):
        log.info(str(epoch))
        start_time = timeit.default_timer()
        for index, minibatch in enumerate(data_loader_train):
            inputs, gts = minibatch['image'], minibatch['gt']
            inputs, gts = Variable(inputs), Variable(gts)
            inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

            outputs = net.forward(inputs)

            losses = [0] * len(outputs)
            for i in range(0, len(outputs)):
                losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                running_loss_train[i] += losses[i].data[0]
            loss = (1 - epoch / n_epochs) * sum(losses[:-1]) + losses[-1]

            if index % n_samples_train == n_samples_train - 1:
                running_loss_train = [x / n_samples_train for x in running_loss_train]
                loss_train.append(running_loss_train[-1])
                summary_writer.add_scalar('data/total_loss_epoch', running_loss_train[-1], epoch)
                log.info('[Epoch: %d, numImages: %5d]' % (epoch, index + 1))
                for l in range(0, len(running_loss_train)):
                    log.info('Loss %d: %f' % (l, running_loss_train[l]))
                    running_loss_train[l] = 0

                stop_time = timeit.default_timer()
                log.info('Execution time: ' + str(stop_time - start_time))

            loss /= avg_grad_every_n
            loss.backward()
            counter_gradient += 1

            if counter_gradient % avg_grad_every_n == 0:
                summary_writer.add_scalar('data/total_loss_iter', loss.data[0], index + n_samples_train * epoch)
                optimizer.step()
                optimizer.zero_grad()
                counter_gradient = 0

        if (epoch % snapshot_every_n) == snapshot_every_n - 1 and epoch != 0:
            net_provider.save(epoch)

        if is_testing_while_training and epoch % test_every_n == (test_every_n - 1):
            for index, minibatch in enumerate(data_loader_test):
                inputs, gts = minibatch['image'], minibatch['gt']
                inputs, gts = Variable(inputs, volatile=True), Variable(gts, volatile=True)
                inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

                outputs = net.forward(inputs)

                losses = [0] * len(outputs)
                for i in range(0, len(outputs)):
                    losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                    running_loss_test[i] += losses[i].data[0]

                if index % n_samples_test == n_samples_test - 1:
                    running_loss_test = [x / n_samples_test for x in running_loss_test]
                    loss_test.append(running_loss_test[-1])

                    log.info('[Epoch: %d, numImages: %5d]' % (epoch, index + 1))
                    summary_writer.add_scalar('data/test_loss_epoch', running_loss_test[-1], epoch)
                    for l in range(0, len(running_loss_test)):
                        log.info('***Testing *** Loss %d: %f' % (l, running_loss_test[l]))
                        running_loss_test[l] = 0

    summary_writer.close()


def _test(net_provider: NetworkProvider, data_loader: DataLoader, save_dir: Path) -> None:
    log.info('Testing Network')

    net = net_provider.network

    for minibatch in data_loader:
        img, gt, seq_name, fname = minibatch['image'], minibatch['gt'], \
                                   minibatch['seq_name'], minibatch['fname']

        inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)
        inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

        outputs = net.forward(inputs)

        for index in range(inputs.size()[0]):
            pred = np.transpose(outputs[-1].cpu().data.numpy()[index, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            img_ = np.transpose(img.numpy()[index, :, :, :], (1, 2, 0))
            gt_ = np.transpose(gt.numpy()[index, :, :, :], (1, 2, 0))
            gt_ = np.squeeze(gt)

            save_dir_seq = save_dir / seq_name[index]
            save_dir_seq.mkdir(parents=True, exist_ok=True)

            file_name = save_dir_seq / '{0}.png'.format(fname[index])
            sm.imsave(str(file_name), pred)


def _load_network_train_vgg(net_provider: NetworkProvider) -> None:
    if settings.start_epoch == 0:
        if settings.is_loading_vgg_caffe:
            net_provider.init_network(pretrained=2)
        else:
            net_provider.init_network(pretrained=1)
    else:
        net_provider.init_network(pretrained=0)
        net_provider.load(settings.start_epoch)


def _load_network_test_vgg(net_provider: NetworkProvider) -> None:
    net_provider.init_network(pretrained=0)
    net_provider.load(settings.n_epochs)


def _get_optimizer_vgg(net: Module, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                       momentum: float = 0.9) -> Optimizer:
    optimizer = optim.SGD([
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': weight_decay,
         'initial_lr': learning_rate},
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': 2 * learning_rate,
         'initial_lr': 2 * learning_rate},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]],
         'weight_decay': weight_decay,
         'initial_lr': learning_rate},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': 2 * learning_rate,
         'initial_lr': 2 * learning_rate},
        {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'weight' in pr[0]], 'lr': learning_rate / 10,
         'weight_decay': weight_decay, 'initial_lr': learning_rate / 10},
        {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'bias' in pr[0]], 'lr': 2 * learning_rate / 10,
         'initial_lr': 2 * learning_rate / 10},
        {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0, 'initial_lr': 0},
        {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0, 'initial_lr': 0},
        {'params': net.fuse.weight, 'lr': learning_rate / 100, 'initial_lr': learning_rate / 100,
         'weight_decay': weight_decay},
        {'params': net.fuse.bias, 'lr': 2 * learning_rate / 100, 'initial_lr': 2 * learning_rate / 100},
    ], lr=learning_rate, momentum=momentum)
    return optimizer


def _load_network_train_resnet(net_provider: NetworkProvider) -> None:
    if settings.start_epoch == 0:
        net_provider.init_network(pretrained=True)
    else:
        net_provider.init_network(pretrained=False)
        net_provider.load(settings.start_epoch)


def _load_network_test_resnet(net_provider: NetworkProvider) -> None:
    net_provider.init_network(pretrained=False)
    net_provider.load(settings.n_epochs)


def _get_optimizer_resnet(net: Module, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                          momentum: float = 0.9) -> Optimizer:
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    return optimizer


if __name__ == '__main__':
    db_root_dir = P.db_root_dir()
    save_dir_root = P.save_root_dir()

    save_dir_models = Path('models')
    save_dir_models.mkdir(parents=True, exist_ok=True)
    save_dir_results = Path('results')
    save_dir_results.mkdir(parents=True, exist_ok=True)

    net_provider = NetworkProvider('vgg16', OSVOS_VGG, save_dir_models,
                                   load_network_train=_load_network_train_vgg,
                                   load_network_test=_load_network_test_vgg,
                                   get_optimizer=_get_optimizer_vgg)

    # net_provider = NetworkProvider('resnet18', OSVOS_RESNET, save_dir_models,
    #                                load_network_train=_load_network_train_resnet,
    #                                load_network_test=_load_network_test_resnet,
    #                                get_optimizer=_get_optimizer_resnet)

    train_and_test(net_provider, settings, is_training=False)
