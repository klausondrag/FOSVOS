import timeit

import torch
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from layers.osvos_layers import class_balanced_cross_entropy_loss
from util.logger import get_logger
import util.gpu_handler as gpu_handler
from config.mypath import PathConfig

log = get_logger(__file__)


def train(n_epochs: int, data_loader: DataLoader, net: torch.nn.Module,
          optimizer: Optimizer, writer: SummaryWriter, sequence_name: str,
          snapshot_every_n: int, n_avg_gradient: int) -> None:
    log.info('Start of training of sequence {}'.format(sequence_name))

    log_every_n = n_epochs // 20
    num_img_tr = len(data_loader)
    loss_tr = []
    avg_gradient = 0

    time_start_total = timeit.default_timer()

    for epoch in range(1, n_epochs + 1):
        time_start_epoch = timeit.default_timer()

        running_loss_tr = 0
        for ii, sample_batched in enumerate(data_loader):

            inputs, gts = sample_batched['image'], sample_batched['gt']

            # Forward-Backward of the mini-batch
            inputs, gts = Variable(inputs), Variable(gts)
            inputs, gts = gpu_handler.cast_cuda_if_possible([inputs, gts])

            outputs = net.forward(inputs)

            # Compute the fuse loss
            loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
            running_loss_tr += loss.data[0]

            # Print stuff
            if epoch % log_every_n == 0:
                running_loss_tr /= num_img_tr
                loss_tr.append(running_loss_tr)

                log.info('[Epoch: %d, numImages: %5d]' % (epoch + 1, ii + 1))
                log.info('Loss: %f' % running_loss_tr)
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)

            # Backward the averaged gradient
            loss /= n_avg_gradient
            loss.backward()
            avg_gradient += 1

            # Update the weights once in nAveGrad forward passes
            if avg_gradient % n_avg_gradient == 0:
                writer.add_scalar('data/total_loss_iter', loss.data[0], ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                avg_gradient = 0

        time_stop_epoch = timeit.default_timer()
        log.info('Epoch {0} training time: {1:.03f}'.format(epoch, time_stop_epoch - time_start_epoch))

        # Save the model
        if (epoch % snapshot_every_n) == 0:
            f = PathConfig.get_online_file(sequence_name, epoch)
            torch.save(net.state_dict(), str(f))

    time_stop_total = timeit.default_timer()
    log.info('Total training time: {0:.03f}'.format(time_stop_total - time_start_total))

    return loss_tr
