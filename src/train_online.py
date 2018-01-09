import sys
import timeit
from pathlib import Path
from collections import namedtuple

from tensorboardX import SummaryWriter
import scipy.misc as sm

from torch.autograd import Variable
import torch.optim as optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from networks.osvos_vgg import OSVOS_VGG
from networks.osvos_resnet import OSVOS_RESNET
from layers.osvos_layers import class_balanced_cross_entropy_loss
from dataloaders.helpers import *
from util import gpu_handler, io_helper
from util.logger import get_logger
from config.mypath import Path as P
from util.network_provider import NetworkProvider

if P.is_custom_pytorch():
    sys.path.append(P.custom_pytorch())  # Custom PyTorch

gpu_handler.select_gpu_by_hostname()
log = get_logger(__file__)

Settings = namedtuple('Settings', ['start_epoch', 'n_epochs', 'avg_grad_every_n', 'snapshot_every_n',
                                   'batch_size_train', 'batch_size_test', 'parent_name', 'parent_epoch',
                                   'is_visualizing_network', 'is_visualizing_results'])

settings = Settings(start_epoch=0, n_epochs=2000, avg_grad_every_n=5, snapshot_every_n=2000,
                    batch_size_train=1, batch_size_test=1, parent_name='vgg16', parent_epoch=240,
                    is_visualizing_network=False, is_visualizing_results=False)

settings = Settings(start_epoch=0, n_epochs=2000, avg_grad_every_n=5, snapshot_every_n=2000,
                    batch_size_train=1, batch_size_test=1, parent_name='resnet18', parent_epoch=240,
                    is_visualizing_network=False, is_visualizing_results=False)


def train_and_test(net_provider: NetworkProvider, seq_name: str, settings: Settings,
                   is_training: bool = True, is_testing: bool = True) -> None:
    _set_network_name(net_provider, settings.parent_name, seq_name)

    if is_training:
        net_provider.load_network_train()
        data_loader = io_helper.get_data_loader_train(db_root_dir, settings.batch_size_train, seq_name)
        optimizer = net_provider.get_optimizer()
        summary_writer = _get_summary_writer(seq_name)

        io_helper.write_settings(save_dir, net_provider.name, settings._asdict())
        _train(net_provider, data_loader, optimizer, summary_writer, seq_name, settings.start_epoch, settings.n_epochs,
               settings.avg_grad_every_n, settings.snapshot_every_n)

    if is_testing:
        net_provider.load_network_test()
        data_loader = io_helper.get_data_loader_test(db_root_dir, settings.batch_size_test, seq_name)
        save_dir_images = Path('results') / seq_name
        save_dir_images.mkdir(parents=True, exist_ok=True)

        _test(net_provider, data_loader, seq_name, save_dir_images, settings.is_visualizing_results)

    if settings.is_visualizing_network:
        io_helper.visualize_network(net_provider.network)


def _set_network_name(net_provider: NetworkProvider, parent_name: str, seq_name: str) -> None:
    net_provider.name = parent_name + '_' + seq_name


def _get_summary_writer(seq_name: str) -> SummaryWriter:
    return io_helper.get_summary_writer(save_dir, postfix=seq_name)


def _train(net_provider: NetworkProvider, data_loader: DataLoader, optimizer: Optimizer, summary_writer: SummaryWriter,
           seq_name: str, start_epoch: int, n_epochs: int, avg_grad_every_n: int, snapshot_every_n: int) -> None:
    log.info('Start of Online Training, sequence: ' + seq_name)

    net = net_provider.network

    speeds_training = []
    n_samples = len(data_loader)
    loss_tr = []
    counter_gradient = 0

    start_time = timeit.default_timer()
    for epoch in range(start_epoch, n_epochs):
        epoch_start_time = timeit.default_timer()

        running_loss_tr = 0
        for minibatch_index, minibatch in enumerate(data_loader):
            inputs, gts = minibatch['image'], minibatch['gt']
            inputs, gts = Variable(inputs), Variable(gts)
            inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

            outputs = net.forward(inputs)

            loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
            running_loss_tr += loss.data[0]

            if epoch % (n_epochs // 20) == (n_epochs // 20 - 1):
                running_loss_tr /= n_samples
                loss_tr.append(running_loss_tr)

                log.info('[Epoch {0}: {1}, numImages: {2}]'.format(seq_name, epoch + 1, minibatch_index + 1))
                log.info('Loss {0}: {1}'.format(seq_name, running_loss_tr))
                summary_writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

            loss /= avg_grad_every_n
            loss.backward()
            counter_gradient += 1

            if counter_gradient % avg_grad_every_n == 0:
                summary_writer.add_scalar('data/total_loss_iter', loss.data[0], minibatch_index + n_samples * epoch)
                optimizer.step()
                optimizer.zero_grad()
                counter_gradient = 0

        if (epoch % snapshot_every_n) == snapshot_every_n - 1:  # and epoch != 0:
            net_provider.save(epoch)

        epoch_stop_time = timeit.default_timer()
        t = epoch_stop_time - epoch_start_time
        log.info('epoch {0} {1}: {2} sec'.format(seq_name, str(epoch), str(t)))
        speeds_training.append(t)

    stop_time = timeit.default_timer()
    log.info('Train {0}: total training time {1} sec'.format(seq_name, str(stop_time - start_time)))
    log.info('Train {0}: time per sample {1} sec'.format(seq_name, np.asarray(t).mean()))


def _test(net_provider: NetworkProvider, data_loader: DataLoader, seq_name: str, save_dir: Path,
          is_visualizing_results: bool) -> None:
    log.info('Testing Network')

    net = net_provider.network

    if is_visualizing_results:
        ax_arr = _init_plot()

    test_start_time = timeit.default_timer()
    for minibatch in data_loader:

        img, gt, fname = minibatch['image'], minibatch['gt'], minibatch['fname']

        inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)
        inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

        outputs = net.forward(inputs)

        for index in range(inputs.size()[0]):
            pred = np.transpose(outputs[-1].cpu().data.numpy()[index, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)

            file_name = save_dir / '{0}.png'.format(fname[index])
            sm.imsave(file_name, pred)

            if is_visualizing_results:
                _visualize_results(ax_arr, gt, img, index, pred)

    test_stop_time = timeit.default_timer()
    log.info('Test {0}: total training time {1} sec'.format(seq_name, str(test_stop_time - test_start_time)))
    log.info('Test {0}: {1} images'.format(seq_name, str((len(data_loader)))))
    log.info(
        'Test {0}: time per sample {1} sec'.format(seq_name,
                                                   str((test_stop_time - test_start_time) / len(data_loader))))


def _init_plot():
    plt.close('all')
    plt.ion()
    f, ax_arr = plt.subplots(1, 3)
    return ax_arr


def _visualize_results(ax_arr, gt, img, jj, pred):
    img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
    gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
    gt_ = np.squeeze(gt)
    # Plot the particular example
    ax_arr[0].cla()
    ax_arr[1].cla()
    ax_arr[2].cla()
    ax_arr[0].set_title('Input Image')
    ax_arr[1].set_title('Ground Truth')
    ax_arr[2].set_title('Detection')
    ax_arr[0].imshow(im_normalize(img_))
    ax_arr[1].imshow(gt_)
    ax_arr[2].imshow(im_normalize(pred))
    plt.pause(0.001)


def _load_network_train_vgg(net_provider: NetworkProvider) -> None:
    net_provider.init_network(pretrained=0)
    net_provider.load(settings.parent_epoch, name=settings.parent_name)


def _load_network_test_vgg(net_provider: NetworkProvider) -> None:
    net_provider.init_network(pretrained=0)
    if is_training:
        net_provider.load(settings.n_epochs)
    else:
        net_provider.load(settings.parent_epoch, name=settings.parent_name)


def _get_optimizer_vgg(net: Module, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                       momentum: float = 0.9) -> Optimizer:
    optimizer = optim.SGD([
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]], 'weight_decay': weight_decay},
        {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': learning_rate * 2},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]],
         'weight_decay': weight_decay},
        {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': learning_rate * 2},
        {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0},
        {'params': net.fuse.weight, 'lr': learning_rate / 100, 'weight_decay': weight_decay},
        {'params': net.fuse.bias, 'lr': 2 * learning_rate / 100},
    ], lr=learning_rate, momentum=momentum)
    return optimizer


def _load_network_train_resnet(net_provider: NetworkProvider) -> None:
    net_provider.init_network(pretrained=False)
    net_provider.load(settings.parent_epoch, name=settings.parent_name)


def _load_network_test_resnet(net_provider: NetworkProvider) -> None:
    net_provider.init_network(pretrained=False)
    if is_training:
        net_provider.load(settings.n_epochs)
    else:
        net_provider.load(settings.parent_epoch, name=settings.parent_name)


def _get_optimizer_resnet(net: Module, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                          momentum: float = 0.9) -> Optimizer:
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    return optimizer


if __name__ == '__main__':
    db_root_dir = P.db_root_dir()
    exp_dir = P.exp_dir()
    save_dir = Path('models')
    save_dir.mkdir(parents=True, exist_ok=True)

    is_training = True

    net_provider = NetworkProvider('', OSVOS_VGG, save_dir,
                                   load_network_train=_load_network_train_vgg,
                                   load_network_test=_load_network_test_vgg,
                                   get_optimizer=_get_optimizer_vgg)

    net_provider = NetworkProvider('', OSVOS_RESNET, save_dir,
                                   load_network_train=_load_network_train_resnet,
                                   load_network_test=_load_network_test_resnet,
                                   get_optimizer=_get_optimizer_resnet)

    if settings.is_visualizing_results:
        import matplotlib.pyplot as plt

    sequences = ['bear', 'blackswan', 'bmx-bumps', 'bmx-trees', 'boat', 'breakdance', 'breakdance-flare', 'bus',
                 'camel', 'car-roundabout', 'car-shadow', 'car-turn', 'cows', 'dance-jump', 'dance-twirl', 'dog',
                 'dog-agility', 'drift-chicane', 'drift-straight', 'drift-turn', 'elephant', 'flamingo', 'goat', 'hike',
                 'hockey', 'horsejump-high', 'horsejump-low', 'kite-surf', 'kite-walk', 'libby', 'lucia', 'mallard-fly',
                 'mallard-water', 'motocross-bumps', 'motocross-jump', 'motorbike', 'paragliding', 'paragliding-launch',
                 'parkour', 'rhino', 'rollerblade', 'scooter-black', 'scooter-gray', 'soapbox', 'soccerball',
                 'stroller', 'surf', 'swing', 'tennis', 'train']

    already_done = []
    sequences = [s for s in sequences if s not in already_done]

    # [train_and_test(net_provider, s, settings, is_training=is_training) for s in sequences]
    train_and_test(net_provider, 'bear', settings, is_training=is_training)
