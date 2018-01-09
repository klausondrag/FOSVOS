import socket
import sys
import timeit
from datetime import datetime, timezone
from pathlib import Path
from collections import namedtuple

import scipy.misc as sm
from tensorboardX import SummaryWriter
import numpy as np
import yaml

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import Optimizer
from torchvision import transforms
from torch.utils.data import DataLoader

import visualize as viz
from dataloaders import davis_2016 as db
from dataloaders import custom_transforms as tr
import networks.osvos_vgg as vo
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
settings = Settings(0, 240, 10, 40, False, 5, 1, 1, False, False)


def train_and_test(net_provider: NetworkProvider, settings: Settings, is_training: bool = True,
                   is_testing: bool = True) -> None:
    if is_training:
        _load_network_train(net_provider, settings.start_epoch, settings.is_loading_vgg_caffe)
        data_loader_train = _get_data_loader_train(settings.batch_size_train)
        data_loader_test = _get_data_loader_test(settings.batch_size_test)
        optimizer = _get_optimizer(net_provider.network)
        summary_writer = _get_summary_writer()

        io_helper.write_settings(save_dir, net_provider.name, settings._asdict())
        _train(net_provider, data_loader_train, data_loader_test, optimizer, summary_writer, settings.start_epoch,
               settings.n_epochs, settings.avg_grad_every_n, settings.snapshot_every_n,
               settings.is_testing_while_training, settings.test_every_n)

    if is_testing:
        _load_network_test(net_provider, settings.n_epochs)
        data_loader = _get_data_loader_test(settings.batch_size_test)
        save_dir_images = Path('results') / net_provider.name
        save_dir_images.mkdir(parents=True, exist_ok=True)

        _test(net_provider, data_loader, save_dir_images)

    if settings.is_visualizing_network:
        io_helper.visualize_network(net_provider.network)


def _load_network_train(net_provider: NetworkProvider, start_epoch: int, is_loading_vgg_caffe: bool) -> None:
    if start_epoch == 0:
        if is_loading_vgg_caffe:
            net_provider.init_network(pretrained=2)
        else:
            net_provider.init_network(pretrained=1)
    else:
        net_provider.init_network(pretrained=0)
        net_provider.load(start_epoch)


def _load_network_test(net_provider: NetworkProvider, n_epochs: int) -> None:
    net_provider.init_network(pretrained=0)
    net_provider.load(n_epochs)


def _get_data_loader_train(batch_size: int) -> DataLoader:
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.Resize(),
                                              # tr.ScaleNRotate(rots=(-30,30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    db_train = db.DAVIS2016(mode='train', inputRes=None, db_root_dir=db_root_dir, transform=composed_transforms)
    data_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=2)
    return data_loader


def _get_data_loader_test(batch_size: int) -> DataLoader:
    db_test = db.DAVIS2016(mode='test', db_root_dir=db_root_dir, transform=tr.ToTensor())
    data_loader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=2)
    return data_loader


def _get_optimizer(net, learning_rate: float = 1e-8, weight_decay: float = 0.0002) -> Optimizer:
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
    ], lr=learning_rate, momentum=0.9)
    return optimizer


def _get_summary_writer() -> SummaryWriter:
    return io_helper.get_summary_writer(save_dir, comment='-parent')


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

            save_dir_seq = save_dir / net_provider.name / seq_name[index]
            save_dir_seq.mkdir(parents=True, exist_ok=True)

            file_name = save_dir_seq / '{0}.png'.format(fname[index])
            sm.imsave(str(file_name), pred)


if __name__ == '__main__':
    db_root_dir = P.db_root_dir()
    save_dir_root = P.save_root_dir()

    save_dir = Path('models')
    save_dir.mkdir(parents=True, exist_ok=True)
    net_provider = NetworkProvider('vgg16', vo.OSVOS_VGG, save_dir)

    train_and_test(net_provider, settings, is_training=True)
