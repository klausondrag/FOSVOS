import sys
import timeit
from pathlib import Path

import scipy.misc as sm
from tensorboardX import SummaryWriter
import numpy as np

from torch.autograd import Variable
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from layers.osvos_layers import class_balanced_cross_entropy_loss

from util import gpu_handler, io_helper
from util.logger import get_logger
from config.mypath import Path as P
from util.network_provider import NetworkProvider, VGGOfflineProvider, ResNetOfflineProvider
from util.settings import OfflineSettings

if P.is_custom_pytorch():
    sys.path.append(P.custom_pytorch())
if P.is_custom_opencv():
    sys.path.insert(0, P.custom_opencv())
gpu_handler.select_gpu_by_hostname()

log = get_logger(__file__)


def train_and_test(net_provider: NetworkProvider, settings: OfflineSettings, is_training: bool = True,
                   is_testing: bool = True) -> None:
    if is_training:
        net_provider.load_network_train()
        data_loader_train = io_helper.get_data_loader_train(db_root_dir, settings.batch_size_train)
        data_loader_test = io_helper.get_data_loader_test(db_root_dir, settings.batch_size_test)
        optimizer = net_provider.get_optimizer()
        summary_writer = _get_summary_writer()

        io_helper.write_settings(save_dir_models, net_provider.name, settings)
        _train(net_provider, data_loader_train, data_loader_test, optimizer, summary_writer, settings.start_epoch,
               settings.n_epochs, settings.avg_grad_every_n, settings.snapshot_every_n,
               settings.is_testing_while_training, settings.test_every_n)

    if is_testing:
        net_provider.load_network_test()
        data_loader = io_helper.get_data_loader_test(db_root_dir, settings.batch_size_test)
        save_dir = save_dir_results / net_provider.name / 'offline'

        _test(net_provider, data_loader, save_dir)

    if settings.is_visualizing_network:
        io_helper.visualize_network(net_provider.network)


def _get_summary_writer() -> SummaryWriter:
    return io_helper.get_summary_writer(save_dir_models, comment='-offline')


def _train(net_provider: NetworkProvider, data_loader_train: DataLoader, data_loader_test: DataLoader,
           optimizer: Optimizer, summary_writer: SummaryWriter, start_epoch: int, n_epochs: int, avg_grad_every_n: int,
           snapshot_every_n: int, is_testing_while_training: bool, test_every_n: int) -> None:
    log.info('Start of offline training')

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
            loss = (1 - epoch / n_epochs) * sum(losses[:-1]) + losses[-1]  # type: Variable

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
            net_provider.save_model(epoch)

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


if __name__ == '__main__':
    db_root_dir = P.db_root_dir()
    save_dir_root = P.save_root_dir()

    save_dir_models = Path('models')
    save_dir_models.mkdir(parents=True, exist_ok=True)
    save_dir_results = Path('results')
    save_dir_results.mkdir(parents=True, exist_ok=True)

    is_training = True
    is_training = False

    settings = OfflineSettings(is_training=is_training, start_epoch=0, n_epochs=240, avg_grad_every_n=10,
                               snapshot_every_n=40, is_testing_while_training=False, test_every_n=5,
                               batch_size_train=1, batch_size_test=1, is_visualizing_network=False,
                               is_visualizing_results=False, is_loading_vgg_caffe=False)

    net_provider = VGGOfflineProvider('vgg16', save_dir_models, settings)
    net_provider = ResNetOfflineProvider('resnet18', save_dir_models, settings)

    train_and_test(net_provider, settings, is_training=is_training)