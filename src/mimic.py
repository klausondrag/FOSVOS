import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

import torch
from torch import nn, optim
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from networks.osvos_resnet import OSVOS_RESNET
from util import gpu_handler, experiment_helper, io_helper
from util.logger import get_logger

log = get_logger(__file__)


def get_net(sequence_name: Optional[str], mimic_offline: bool) -> nn.Module:
    net = OSVOS_RESNET(pretrained=False)
    if mimic_offline:
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


def get_suffix(scale_down_exponential, sequence_name, learning_rate, loss) -> str:
    return ',sde={0},sequence={1},lr={2},loss={3}'.format(str(scale_down_exponential),
                                                          'offline' if sequence_name is None else sequence_name,
                                                          str(learning_rate),
                                                          loss)


def main(n_epochs: int, sequence_name: Optional[str], mimic_offline: bool, scale_down_exponential: int,
         learning_rate: float, no_training: bool) -> None:
    if mimic_offline:
        sequence_name = None

    loss = 'L1'
    loss = 'MSE'

    suffix = get_suffix(scale_down_exponential, sequence_name, learning_rate, loss)
    log.info('Suffix: %s', suffix)

    current_time = datetime.now().isoformat()
    tensorboard_dir = Path('tensorboard') / 'mimic' / suffix[1:] / str(current_time)
    tensorboard_dir = str(tensorboard_dir)
    log.info('Logging for tensorboard in directory: %s', tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)

    if mimic_offline:
        path_output_model = Path('./models/resnet18_11' + suffix + '.pth')
    else:
        path_output_model = Path('./models/resnet18_11_11_' + sequence_name + '_epoch-9999' + suffix + '.pth')

    data_loader_train = io_helper.get_data_loader_train(Path('/usr/stud/ondrag/DAVIS'), batch_size=5,
                                                        seq_name=sequence_name)
    data_loader_test = io_helper.get_data_loader_test(Path('/usr/stud/ondrag/DAVIS'), batch_size=1,
                                                      seq_name=sequence_name)

    if not no_training:

        net_teacher = get_net(sequence_name, mimic_offline)
        net_teacher.train()
        net_teacher.is_mode_mimic = True
        net_teacher = gpu_handler.cast_cuda_if_possible(net_teacher)

        net_student = OSVOS_RESNET(pretrained=False, scale_down_exponential=scale_down_exponential, is_mode_mimic=True)
        net_student.train()
        net_student = gpu_handler.cast_cuda_if_possible(net_student)

        optimizer = optim.Adam(net_student.parameters(), lr=learning_rate, weight_decay=0.0002)

        if loss == 'MSE':
            criterion = nn.MSELoss()
        elif loss == 'L1':
            criterion = nn.L1Loss()
        else:
            raise Exception('Unknown loss function')
        criterion = gpu_handler.cast_cuda_if_possible(criterion)

        log.info('Starting Training')
        for epoch in range(n_epochs):
            loss_training = 0.0
            for minibatch in data_loader_train:
                net_student.zero_grad()
                optimizer.zero_grad()

                inputs_teacher, inputs_student = minibatch['image'], minibatch['image']
                inputs_teacher, inputs_student = (Variable(inputs_teacher, requires_grad=False),
                                                  Variable(inputs_student, requires_grad=True))
                inputs_teacher, inputs_student = gpu_handler.cast_cuda_if_possible([inputs_teacher, inputs_student])

                outputs_teacher = net_teacher.forward(inputs_teacher)
                outputs_teacher = outputs_teacher[-1]
                outputs_teacher = outputs_teacher.detach()
                outputs_teacher = gpu_handler.cast_cuda_if_possible(outputs_teacher)

                outputs_student = net_student.forward(inputs_student)
                outputs_student = outputs_student[-1]
                outputs_student = gpu_handler.cast_cuda_if_possible(outputs_student)

                loss = criterion(outputs_student, outputs_teacher)
                loss_training += loss.data[0]
                loss.backward()
                optimizer.step()

            loss_training /= len(data_loader_train)
            writer.add_scalar('data/training/loss', loss_training, epoch)

            if epoch % 200 == 199:
                log.info('Training: epoch {0}, loss == {1}'.format(epoch, loss.data[0]))

            if epoch % 100 == 99:
                log.info('Validating...')
                loss_validation = 0.0
                for minibatch in data_loader_train:
                    inputs_image, inputs_ground_truth = minibatch['image'], minibatch['gt']
                    inputs_image, inputs_ground_truth = (Variable(inputs_image, requires_grad=False,
                                                                  volatile=True),
                                                         Variable(inputs_ground_truth, requires_grad=False,
                                                                  volatile=True))
                    inputs_image, inputs_ground_truth = gpu_handler.cast_cuda_if_possible(
                        [inputs_image, inputs_ground_truth])

                    outputs_student = net_student.forward(inputs_image)
                    outputs_student = outputs_student[-1]
                    outputs_student = gpu_handler.cast_cuda_if_possible(outputs_student)

                    loss = criterion(outputs_student, inputs_ground_truth)
                    loss_validation += loss.data[0]

                loss_validation /= len(data_loader_test)
                writer.add_scalar('data/validation/loss', loss_validation, epoch)
                log.info('Validation: epoch {0}, loss == {1}'.format(epoch, loss_validation))

        log.info('Finished Training')

        log.info('Saving model to %s', str(path_output_model))
        torch.save(net_student.state_dict(), str(path_output_model))

    net_student = OSVOS_RESNET(pretrained=False, scale_down_exponential=scale_down_exponential, is_mode_mimic=True)
    log.info('Loading model from %s', str(path_output_model))
    net_student.load_state_dict(torch.load(str(path_output_model), map_location=lambda storage, loc: storage))
    net_student = gpu_handler.cast_cuda_if_possible(net_student)
    net_student.eval()

    net_provider = DummyProvider(net_student)

    path_output_images = Path('./results/resnet18/11/11' + suffix)
    log.info('Saving images to %s', str(path_output_images))

    # first time to measure the speed
    experiment_helper.test(net_provider, data_loader_test, path_output_images, is_visualizing_results=False,
                           eval_speeds=True, seq_name=sequence_name)

    # second time for image output
    experiment_helper.test(net_provider, data_loader_test, path_output_images, is_visualizing_results=False,
                           eval_speeds=False, seq_name=sequence_name)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--n-epochs', default=20, type=int, help='')
    parser.add_argument('--gpu-id', default=1, type=int, help='The gpu id to use')
    parser.add_argument('-o', '--object', default='blackswan', type=str, help='The object to train on')
    parser.add_argument('--mimic-offline', action='store_true', help='')
    parser.add_argument('--scale-down-exponential', default=0, type=int, help='')
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='')
    parser.add_argument('--no-training', action='store_true',
                        help='True if the program should train the model, else False')
    args = parser.parse_args()

    gpu_handler.select_gpu(args.gpu_id)

    # args.no_training = True
    # args.n_epoch = 1000
    # args.object = 'libby'
    # args.scale_down_exponential = 2

    main(args.n_epochs, args.object, args.mimic_offline, args.scale_down_exponential, args.learning_rate,
         args.no_training)
