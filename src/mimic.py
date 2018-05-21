import argparse
from pathlib import Path
from typing import Optional
import shutil

import torch
from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from layers.osvos_layers import class_balanced_cross_entropy_loss
from networks.osvos_resnet import OSVOS_RESNET
from util import gpu_handler, experiment_helper, io_helper
from util.logger import get_logger

log = get_logger(__file__)


def get_net(sequence_name: Optional[str], is_offline_mode: bool) -> nn.Module:
    net = OSVOS_RESNET(pretrained=False)
    if is_offline_mode:
        path_model = './models/resnet18_11_epoch-239.pth'
    else:
        path_model = './models/resnet18_11_11_' + sequence_name + '_epoch-9999.pth'
    path_model = Path(path_model)
    log.info('Loading model from %s', str(path_model))
    parameters = torch.load(str(path_model), map_location=lambda storage, loc: storage)
    net.load_state_dict(parameters)
    net = gpu_handler.cast_cuda_if_possible(net)
    return net


class DummyProvider:
    def __init__(self, net):
        self.network = net


def get_experiment_id(scale_down_exponent: int, sequence_name: Optional[str], learning_rate: float, criterion: str,
                      criterion_from: str, learn_from: str) -> str:
    s = 'sequence={1},sde={0},lr={2:0.1e},criterion={3},criterion_from={4},learn_from={5}'
    return s.format(str(scale_down_exponent), 'offline' if sequence_name is None else sequence_name,
                    learning_rate, criterion, criterion_from, learn_from)


def main(n_epochs: int, sequence_name: Optional[str], is_offline_mode: bool, scale_down_exponent: int,
         learning_rate: float, no_training: bool, criterion: str, criterion_from: str, learn_from: str) -> None:
    experiment_id = get_experiment_id(scale_down_exponent, sequence_name, learning_rate, criterion, criterion_from,
                                      learn_from)
    log.info('Experiment ID: %s', experiment_id)
    path_stem = 'resnet18/11'
    path_stem += '/' + 'mimic'
    path_stem += '/' + experiment_id
    path_stem += '/' + ('offline' if is_offline_mode else 'online')
    log.info('Path stem: %s', str(path_stem))

    path_output_model_base = Path('models') / path_stem
    path_output_model_base.mkdir(parents=True, exist_ok=True)

    dataloader_val = io_helper.get_data_loader_test(Path('/usr/stud/ondrag/DAVIS'), batch_size=1,
                                                    seq_name=sequence_name)

    if not no_training:
        dataloader_train = io_helper.get_data_loader_train(Path('/usr/stud/ondrag/DAVIS'), batch_size=5,
                                                           seq_name=sequence_name)
        net_teacher = None
        if learn_from == 'teacher':
            net_teacher = get_net(sequence_name, is_offline_mode)
            net_teacher.train()
            net_teacher.is_mode_mimic = True
            net_teacher = gpu_handler.cast_cuda_if_possible(net_teacher)

        net_student = OSVOS_RESNET(pretrained=False, scale_down_exponent=scale_down_exponent, is_mode_mimic=True)
        net_student.train()
        net_student = gpu_handler.cast_cuda_if_possible(net_student)

        optimizer = optim.Adam(net_student.parameters(), lr=learning_rate, weight_decay=0.0002)

        if criterion == 'MSE':
            criterion = nn.MSELoss(size_average=False)
            criterion = gpu_handler.cast_cuda_if_possible(criterion)
        elif criterion == 'L1':
            criterion = nn.L1Loss(size_average=False)
            criterion = gpu_handler.cast_cuda_if_possible(criterion)
        elif criterion == 'CBCEL':
            criterion = class_balanced_cross_entropy_loss
        else:
            raise Exception('Unknown loss function')

        path_tensorboard = Path('tensorboard') / path_stem
        if path_tensorboard.exists():
            log.warn('Deleting existing tensorboard directory: %s', str(path_tensorboard))
            shutil.rmtree(str(path_tensorboard))

        path_tensorboard = str(path_tensorboard)
        log.info('Logging for tensorboard in directory: %s', path_tensorboard)
        writer = SummaryWriter(path_tensorboard)

        log.info('Starting Training')
        for epoch in range(1, n_epochs + 1):
            calculate_loss(criterion, epoch, n_epochs, learn_from, net_student, net_teacher, dataloader_train,
                           optimizer, 'train', writer)

            if epoch % 10 == 0:
                log.info('Validating...')
                calculate_loss(criterion, epoch, n_epochs, learn_from, net_student, net_teacher, dataloader_val,
                               optimizer, 'val', writer)

        writer.close()
        log.info('Finished Training')

        path_output_model = path_output_model_base / (str(n_epochs) + '.pth')
        log.info('Saving model to %s', str(path_output_model))
        torch.save(net_student.state_dict(), str(path_output_model))

    net_student = OSVOS_RESNET(pretrained=False, scale_down_exponent=scale_down_exponent, is_mode_mimic=True)
    log.info('Loading model from %s', str(path_output_model))
    net_student.load_state_dict(torch.load(str(path_output_model), map_location=lambda storage, loc: storage))
    net_student = gpu_handler.cast_cuda_if_possible(net_student)
    net_student.eval()

    net_provider = DummyProvider(net_student)

    path_output_images = Path('results') / path_stem / str(n_epochs)
    log.info('Saving images to %s', str(path_output_images))

    # first time to measure the speed
    experiment_helper.test(net_provider, dataloader_val, path_output_images, is_visualizing_results=False,
                           eval_speeds=True, seq_name=sequence_name)

    # second time for image output
    experiment_helper.test(net_provider, dataloader_val, path_output_images, is_visualizing_results=False,
                           eval_speeds=False, seq_name=sequence_name)


def calculate_loss(criterion, epoch, n_epochs, learn_from, net_student, net_teacher, dataloader, optimizer,
                   mode, writer):
    if mode == 'train':
        net_student.train()
        net_teacher.train()
    else:
        net_student.eval()
        net_teacher.eval()

    loss_epoch = 0.0
    for minibatch in dataloader:
        if mode == 'train':
            net_student.zero_grad()
            optimizer.zero_grad()

        loss = _get_loss_minibatch(criterion, epoch, n_epochs, learn_from, minibatch, net_student, net_teacher)
        loss_epoch += loss.item()

        if mode == 'train':
            loss.backward()
            optimizer.step()

        loss_epoch /= len(dataloader.dataset)
    writer.add_scalar('data/{mode}/loss'.format(mode=mode), loss_epoch, epoch)


def _get_loss_minibatch(criterion, epoch, n_epochs, learn_from, minibatch, net_student, net_teacher):
    inputs_image = minibatch['image']
    inputs_image = gpu_handler.cast_cuda_if_possible(inputs_image)
    outputs_student = net_student.forward(inputs_image)

    if learn_from == 'teacher':
        outputs_teacher = net_teacher.forward(inputs_image)
    else:
        ground_truth = minibatch['gt']
        ground_truth = gpu_handler.cast_cuda_if_possible(ground_truth)

    losses = [0] * len(outputs_student)
    for i in range(0, len(outputs_student)):
        o_student = outputs_student[i]
        o_student = gpu_handler.cast_cuda_if_possible(o_student)

        if learn_from == 'teacher':
            o_teacher = outputs_teacher[i]
            o_teacher = o_teacher.detach()
            o_teacher = gpu_handler.cast_cuda_if_possible(o_teacher)
            losses[i] = criterion(o_student, o_teacher)
        else:
            losses[i] = criterion(o_student, ground_truth)

    loss = (1 - epoch / n_epochs) * sum(losses[:-1]) + losses[-1]  # type: Variable
    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--gpu-id', default=None, type=int, help='The gpu id to use')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('-s', '--sequence-name', default=None, type=Optional[str])
    parser.add_argument('-sg', '--sequence-group', default=None, type=Optional[int])
    parser.add_argument('-sgs', '--sequence-group-size', default=None, type=Optional[int])

    parser.add_argument('--n-epochs', default=300, type=int, help='')
    parser.add_argument('--scale-down-exponent', default=0, type=int, help='')
    parser.add_argument('--learning-rate', default=1e-2, type=float, help='')
    parser.add_argument('--no-training', action='store_true',
                        help='True if the program should train the model, else False')
    parser.add_argument('--criterion', default='MSE', type=str, help='The loss to use',
                        choices=['MSE', 'L1', 'CBCEL'])

    args = parser.parse_args()

    gpu_handler.select_gpu(args.gpu_id)

    if args.offline:
        args.sequence_name = None

    if not args.offline and args.sequence_name is None:
        sequences_val = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows',
                         'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf',
                         'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']

        sequences_train = ['bear', 'bmx-bumps', 'boat', 'breakdance-flare', 'bus', 'car-turn', 'dance-jump',
                           'dog-agility', 'drift-turn', 'elephant', 'flamingo', 'hike', 'hockey', 'horsejump-low',
                           'kite-walk', 'lucia', 'mallard-fly', 'mallard-water', 'motocross-bumps', 'motorbike',
                           'paragliding', 'rhino', 'rollerblade', 'scooter-gray', 'soccerball', 'stroller', 'surf',
                           'swing', 'tennis', 'train']

        sequences_all = list(set(sequences_train + sequences_val))

        if args.sequence_group is None:
            already_done = []
            sequences = [s
                         for s in sequences_val
                         if s not in already_done]
        else:
            sequences = [s
                         for i, s in enumerate(sequences_val)
                         if i % args.sequence_group_size == args.sequence_group]

        [main(args.n_epochs, s, args.offline, args.scale_down_exponent, args.learning_rate,
              args.no_training, args.criterion, criterion_from='all', learn_from='ground_truth')
         for s in sequences]

    else:
        main(args.n_epochs, args.sequence_name, args.offline, args.scale_down_exponent, args.learning_rate,
             args.no_training, args.criterion, criterion_from='all', learn_from='ground_truth')
