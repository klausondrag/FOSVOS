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

# Setting of parameters
# Parameters in p are used for the name of the model
p = {
    'trainBatch': 1,  # Number of Images in each mini-batch
}

# # Setting other parameters
nEpochs = 240  # 240  # Number of epochs for training (500.000/2079)
useTest = 1  # See evolution of the test set when training?
testBatch = 1  # Testing Batch
nTestInterval = 5  # 5  # Run on test set every nTestInterval epochs
db_root_dir = P.db_root_dir()
save_dir_root = P.save_root_dir()

should_visualize_network = False
snapshot = 40  # 40  # Store a model every snapshot epochs
nAveGrad = 10

should_load_vgg_caffe = False
start_epoch = 0  # Default is 0, change if want to resume

save_dir = Path('models')
save_dir.mkdir(exist_ok=True)

net_provider = NetworkProvider('vgg16', vo.OSVOS_VGG, save_dir)

if start_epoch == 0:
    if should_load_vgg_caffe:
        net = net_provider.init_network(pretrained=2)
    else:
        net = net_provider.init_network(pretrained=1)
else:
    net = net_provider.init_network(pretrained=0)
    net_provider.load(start_epoch)

# Logging into Tensorboard
log_dir = save_dir / 'runs' / (datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
writer = SummaryWriter(log_dir=str(log_dir), comment='-parent')
y = net.forward(Variable(torch.randn(1, 3, 480, 854)))
writer.add_graph(net, y[-1])

# Visualize the network
if should_visualize_network:
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


optimizer = _get_optimizer(net)


def _get_data_loader_train() -> DataLoader:
    # Define augmentation transformations as a composition
    composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                              tr.Resize(),
                                              # tr.ScaleNRotate(rots=(-30,30), scales=(.75, 1.25)),
                                              tr.ToTensor()])
    db_train = db.DAVIS2016(mode='train', inputRes=None, db_root_dir=db_root_dir, transform=composed_transforms)
    data_loader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=2)
    return data_loader


trainloader = _get_data_loader_train()


def _get_data_loader_test() -> DataLoader:
    db_test = db.DAVIS2016(mode='test', db_root_dir=db_root_dir, transform=tr.ToTensor())
    data_loader = DataLoader(db_test, batch_size=testBatch, shuffle=False, num_workers=2)
    return data_loader


testloader = _get_data_loader_test()

num_img_tr = len(trainloader)
num_img_ts = len(testloader)
running_loss_tr = [0] * 5
running_loss_ts = [0] * 5
loss_tr = []
loss_ts = []
aveGrad = 0

log.info("Training Network")
# Main Training and Testing Loop
for epoch in range(start_epoch, nEpochs):
    start_time = timeit.default_timer()
    # One training epoch
    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs, gts = Variable(inputs), Variable(gts)
        inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

        outputs = net.forward(inputs)

        # Compute the losses, side outputs and fuse
        losses = [0] * len(outputs)
        for i in range(0, len(outputs)):
            losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
            running_loss_tr[i] += losses[i].data[0]
        loss = (1 - epoch / nEpochs) * sum(losses[:-1]) + losses[-1]

        # Print stuff
        if ii % num_img_tr == num_img_tr - 1:
            running_loss_tr = [x / num_img_tr for x in running_loss_tr]
            loss_tr.append(running_loss_tr[-1])
            writer.add_scalar('data/total_loss_epoch', running_loss_tr[-1], epoch)
            log.info('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
            for l in range(0, len(running_loss_tr)):
                log.info('Loss %d: %f' % (l, running_loss_tr[l]))
                running_loss_tr[l] = 0

            stop_time = timeit.default_timer()
            log.info("Execution time: " + str(stop_time - start_time))

        # Backward the averaged gradient
        loss /= nAveGrad
        loss.backward()
        aveGrad += 1

        # Update the weights once in nAveGrad forward passes
        if aveGrad % nAveGrad == 0:
            writer.add_scalar('data/total_loss_iter', loss.data[0], ii + num_img_tr * epoch)
            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

    # Save the model
    if (epoch % snapshot) == snapshot - 1 and epoch != 0:
        net_provider.save(epoch)

    # One testing epoch
    if useTest and epoch % nTestInterval == (nTestInterval - 1):
        for ii, sample_batched in enumerate(testloader):
            inputs, gts = sample_batched['image'], sample_batched['gt']

            # Forward pass of the mini-batch
            inputs, gts = Variable(inputs, volatile=True), Variable(gts, volatile=True)
            inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

            outputs = net.forward(inputs)

            # Compute the losses, side outputs and fuse
            losses = [0] * len(outputs)
            for i in range(0, len(outputs)):
                losses[i] = class_balanced_cross_entropy_loss(outputs[i], gts, size_average=False)
                running_loss_ts[i] += losses[i].data[0]
            loss = (1 - epoch / nEpochs) * sum(losses[:-1]) + losses[-1]

            # Print stuff
            if ii % num_img_ts == num_img_ts - 1:
                running_loss_ts = [x / num_img_ts for x in running_loss_ts]
                loss_ts.append(running_loss_ts[-1])

                log.info('[Epoch: %d, numImages: %5d]' % (epoch, ii + 1))
                writer.add_scalar('data/test_loss_epoch', running_loss_ts[-1], epoch)
                for l in range(0, len(running_loss_ts)):
                    log.info('***Testing *** Loss %d: %f' % (l, running_loss_ts[l]))
                    running_loss_ts[l] = 0

writer.close()

# Test parent network
log.info('Testing Network')
net = net_provider.init_network(pretrained=0)
net_provider.load(nEpochs)

db_test = db.DAVIS2016(mode='test', db_root_dir=db_root_dir, transform=tr.ToTensor())
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=2)
for ii, sample_batched in enumerate(testloader):

    img, gt, seq_name, fname = sample_batched['image'], sample_batched['gt'], \
                               sample_batched['seq_name'], sample_batched['fname']

    # Forward of the mini-batch
    inputs, gts = Variable(img, volatile=True), Variable(gt, volatile=True)
    inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

    outputs = net.forward(inputs)

    for jj in range(int(inputs.size()[0])):
        pred = np.transpose(outputs[-1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
        pred = 1 / (1 + np.exp(-pred))
        pred = np.squeeze(pred)
        img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
        gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
        gt_ = np.squeeze(gt)

        save_dir_seq = save_dir / net_provider.name / seq_name[jj]
        save_dir_seq.mkdir(exist_ok=True)

        # Save the result, attention to the index jj
        file_name = save_dir_seq / '{0}.png'.format(fname[jj])
        sm.imsave(str(file_name), pred)


def train_and_test():
    pass


if __name__ == '__main__':
    settings = None
    train_and_test(net_provider, 'bear', settings, should_train=True)
