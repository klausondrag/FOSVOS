import argparse
from pathlib import Path
from typing import Optional

import torch
from torch import nn, optim
from torch.autograd import Variable

from networks.osvos_resnet import OSVOS_RESNET
from util import gpu_handler, experiment_helper, io_helper
from util.logger import get_logger

log = get_logger(__file__)


def get_net(sequence_name: Optional[str], mimic_offline: bool) -> nn.Module:
    net = OSVOS_RESNET(pretrained=True)
    if mimic_offline:
        path_model = './models/resnet18_11_epoch-239.pth'
    else:
        path_model = './models/resnet18_11_11_' + sequence_name + '_epoch-9999.pth'
    path_model = Path(path_model)
    parameters = torch.load(str(path_model), map_location=lambda storage, loc: storage)
    net.load_state_dict(parameters)
    net = gpu_handler.cast_cuda_if_possible(net)
    return net


class DummyProvider:
    def __init__(self, net):
        self.network = net


def get_suffix(scale_down_exponential, sequence_name, learning_rate) -> str:
    return ',sde={0},mimic={1},lr={2}'.format(str(scale_down_exponential),
                                              'offline' if sequence_name is None else sequence_name,
                                              str(learning_rate))


def main(n_epochs: int, sequence_name: Optional[str], mimic_offline: bool, scale_down_exponential: int,
         learning_rate: float) -> None:
    if mimic_offline:
        sequence_name = None

    suffix = get_suffix(scale_down_exponential, sequence_name, learning_rate)
    log.info('Suffix: %s', suffix)

    data_loader_train = io_helper.get_data_loader_train(Path('/usr/stud/ondrag/DAVIS'), batch_size=5,
                                                        seq_name=sequence_name)
    data_loader_test = io_helper.get_data_loader_test(Path('/usr/stud/ondrag/DAVIS'), batch_size=1,
                                                      seq_name=sequence_name)

    net_teacher = get_net(sequence_name, mimic_offline)
    net_student = OSVOS_RESNET(pretrained=False, scale_down_exponential=scale_down_exponential)

    optimizer = optim.Adam(net_student.parameters(), lr=learning_rate, weight_decay=0.0002)
    criterion = nn.MSELoss()
    criterion = gpu_handler.cast_cuda_if_possible(criterion)
    log.info('Starting Training')
    for epoch in range(n_epochs):
        for minibatch in data_loader_train:
            net_student.zero_grad()
            optimizer.zero_grad()

            inputs_teacher, inputs_student = minibatch['image'], minibatch['image']
            inputs_teacher, inputs_student = (Variable(inputs_teacher, requires_grad=False),
                                              Variable(inputs_student, requires_grad=True))
            inputs_teacher, inputs_student = gpu_handler.cast_cuda_if_possible([inputs_teacher, inputs_student])

            outputs_teacher = net_teacher.forward(inputs_teacher)
            outputs_teacher = outputs_teacher.detach()
            outputs_student = net_teacher.forward(inputs_student)

            loss = criterion(outputs_student, outputs_teacher)
            loss.backward()
            optimizer.step()

            if epoch & 200 == 199:
                log.info('Epoch {0}: Loss == {1}'.format(epoch, loss.data[0]))
    log.info('Finished Training')

    if mimic_offline:
        path_output_model = Path('./models/resnet18_11' + suffix + '.pth')
    else:
        path_output_model = Path('./models/resnet18_11_11_' + sequence_name + '_epoch-9999' + suffix + '.pth')

    log.info('Saving model to %s', str(path_output_model))
    torch.save(net_student.state_dict(), str(path_output_model))

    net_provider = DummyProvider(net_student)

    path_output_images = Path('../results/resnet18/11' + suffix)

    # first time to measure the speed
    experiment_helper.test(net_provider, data_loader_test, path_output_images, is_visualizing_results=False,
                           eval_speeds=True, seq_name=sequence_name)

    # second time for image output
    experiment_helper.test(net_provider, data_loader_test, path_output_images, is_visualizing_results=False,
                           eval_speeds=False, seq_name=sequence_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--n-epochs', default=20, type=int, help='')
    parser.add_argument('--gpu-id', default=1, type=int, help='The gpu id to use')
    parser.add_argument('-o', '--object', default='blackswan', type=str, help='The object to train on')
    parser.add_argument('--mimic-offline', action='store_true', help='')
    parser.add_argument('--scale-down-exponential', default=0, type=int, help='')
    parser.add_argument('--learning-rate', default=1e-4, type=float, help='')
    args = parser.parse_args()

    gpu_handler.select_gpu(args.gpu_id)

    main(args.n_epochs, args.object, args.mimic_offline, args.scale_down_exponential, args.learning_rate)
