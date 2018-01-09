import socket
import sys
import timeit
from datetime import datetime
from pathlib import Path

import scipy.misc as sm
from tensorboardX import SummaryWriter
import numpy as np

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

from util import gpu_handler
from util.logger import get_logger
from config.mypath import Path as P
from util.network_provider import NetworkProvider

if P.is_custom_pytorch():
    sys.path.append(P.custom_pytorch())
if P.is_custom_opencv():
    sys.path.insert(0, P.custom_opencv())
gpu_handler.select_gpu_by_hostname()

log = get_logger(__file__)

p = {
    'trainBatch': 1,
}

nEpochs = 240
useTest = 1
testBatch = 1
nTestInterval = 5
db_root_dir = P.db_root_dir()
save_dir_root = P.save_root_dir()

is_visualizing_network = False
snapshot = 40
nAveGrad = 10

is_loading_vgg_caffe = False
start_epoch = 0

save_dir = Path('models')
save_dir.mkdir(parents=True, exist_ok=True)

net_provider = NetworkProvider('vgg16', vo.OSVOS_VGG, save_dir)


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


def _get_summary_writer() -> SummaryWriter:
    log_dir = save_dir / 'runs' / (datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    summary_writer = SummaryWriter(log_dir=str(log_dir), comment='-parent')
    return summary_writer


def _visualize_network(net):
    x = torch.randn(1, 3, 480, 854)
    x = Variable(x)
    y = net.forward(x)
    g = viz.make_dot(y, net.state_dict())
    g.view()


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


def _get_data_loader_train() -> DataLoader:
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.Resize(),
                                              # tr.ScaleNRotate(rots=(-30,30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    db_train = db.DAVIS2016(mode='train', inputRes=None, db_root_dir=db_root_dir, transform=composed_transforms)
    data_loader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)
    return data_loader


def _get_data_loader_test() -> DataLoader:
    db_test = db.DAVIS2016(mode='test', db_root_dir=db_root_dir, transform=tr.ToTensor())
    data_loader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=2)
    return data_loader


def _train():
    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = [0] * 5
    running_loss_ts = [0] * 5
    loss_tr = []
    loss_ts = []
    aveGrad = 0

    log.info("Training Network")
    for epoch in range(start_epoch, nEpochs):
        start_time = timeit.default_timer()
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['image'], sample_batched['gt']

            inputs, gts = Variable(inputs), Variable(gts)
            inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

            outputs = net.forward(inputs)

            losses = [0] * len(outputs)
            for i in range(0, len(outputs)):
                losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                running_loss_tr[i] += losses[i].data[0]
            loss = (1 - epoch / nEpochs) * sum(losses[:-1]) + losses[-1]

            if ii % num_img_tr == num_img_tr - 1:
                running_loss_tr = [x / num_img_tr for x in running_loss_tr]
                loss_tr.append(running_loss_tr[-1])
                summary_writer.add_scalar('data/total_loss_epoch', running_loss_tr[-1], epoch)
                log.info('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
                for l in range(0, len(running_loss_tr)):
                    log.info('Loss %d: %f' % (l, running_loss_tr[l]))
                    running_loss_tr[l] = 0

                stop_time = timeit.default_timer()
                log.info("Execution time: " + str(stop_time - start_time))

            loss /= nAveGrad
            loss.backward()
            aveGrad += 1

            if aveGrad % nAveGrad == 0:
                summary_writer.add_scalar('data/total_loss_iter', loss.data[0], ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            net_provider.save(epoch)

        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            for ii, sample_batched in enumerate(testloader):
                inputs, gts = sample_batched['image'], sample_batched['gt']

                inputs, gts = Variable(inputs, volatile=True), Variable(gts, volatile=True)
                inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

                outputs = net.forward(inputs)

                losses = [0] * len(outputs)
                for i in range(0, len(outputs)):
                    losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                    running_loss_ts[i] += losses[i].data[0]

                if ii % num_img_ts == num_img_ts - 1:
                    running_loss_ts = [x / num_img_ts for x in running_loss_ts]
                    loss_ts.append(running_loss_ts[-1])

                    log.info('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
                    summary_writer.add_scalar('data/test_loss_epoch', running_loss_ts[-1], epoch)
                    for l in range(0, len(running_loss_ts)):
                        log.info('***Testing *** Loss %d: %f' % (l, running_loss_ts[l]))
                        running_loss_ts[l] = 0

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


n_epochs = 400


def train_and_test(net_provider: NetworkProvider, is_training: bool = True, is_testing: bool = True) -> None:
    if is_training:
        _load_network_train(net_provider, start_epoch, is_loading_vgg_caffe)
        data_loader = _get_data_loader_train()
        optimizer = _get_optimizer(net_provider.network)
        summary_writer = _get_summary_writer()

    if is_testing:
        _load_network_test(net_provider, n_epochs)
        data_loader = _get_data_loader_test()
        save_dir_images = Path('results') / net_provider.name
        save_dir_images.mkdir(parents=True, exist_ok=True)

        _test(net_provider, data_loader, save_dir_images)

    if is_visualizing_network:
        _visualize_network(net_provider.network)


if __name__ == '__main__':
    settings = None
    train_and_test(net_provider, 'bear', settings, is_training=True)
