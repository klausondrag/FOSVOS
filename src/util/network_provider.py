from pathlib import Path
from typing import Optional, Callable

import torch
from torch.nn import Module
from torch.optim import Optimizer

from networks.osvos_vgg import OSVOS_VGG
from util import gpu_handler
from util.logger import get_logger

log = get_logger(__file__)


class NetworkProvider:

    def __init__(self, name: str, network_type: type, save_dir: Path,
                 load_network_train: Callable[[object], None], load_network_test: Callable[[object], None],
                 get_optimizer: Callable[[Module], Optimizer]) -> None:
        # the functions receive NetworkProvider, but this is not allowed as a type hint
        self.name = name
        self.network_type = network_type
        self.save_dir = save_dir
        self.network = None
        self._load_network_train = load_network_train
        self._load_network_test = load_network_test
        self._get_optimizer = get_optimizer

    def init_network(self, **kwargs) -> object:
        net = self.network_type(**kwargs)
        net = gpu_handler.cast_cuda_if_possible(net, verbose=True)
        self.network = net
        return net

    def _get_file_path(self, epoch: int, name: Optional[str] = None) -> str:
        model_name = self.name if name is None else name
        return str(self.save_dir / '{0}_epoch-{1}.pth'.format(model_name, str(epoch)))

    def load(self, epoch: int, name: Optional[str] = None) -> None:
        file_path = self._get_file_path(epoch - 1, name)
        log.info("Loading weights from: {0}".format(file_path))
        self.network.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))

    def save(self, epoch: int, name: Optional[str] = None) -> None:
        file_path = self._get_file_path(epoch, name)
        log.info("Saving weights to: {0}".format(file_path))
        torch.save(self.network.state_dict(), file_path)

    def load_network_train(self) -> None:
        self._load_network_train(self)

    def load_network_test(self) -> None:
        self._load_network_test(self)

    def get_optimizer(self) -> Optimizer:
        return self._get_optimizer(self.network)


if __name__ == '__main__':
    # code below is simply for showing the use case
    if False:
        # parent
        save_dir = Path('models')
        net_provider = NetworkProvider('vgg16', OSVOS_VGG, save_dir)

        resume_epoch = 0
        load_caffe_vgg = False
        nEpochs = 240

        # parent train
        if resume_epoch == 0:
            if load_caffe_vgg:
                net = net_provider.init_network(pretrained=2)
            else:
                net = net_provider.init_network(pretrained=1)
        else:
            net = net_provider.init_network(pretrained=0)
            net_provider.load(resume_epoch)

        epoch = 1
        net_provider.save(epoch)

        # parent test
        net = net_provider.init_network(pretrained=0)
        net_provider.load(nEpochs)

        # online
        save_dir = Path('models')
        net_provider = NetworkProvider('vgg16_blackswan', OSVOS_VGG, save_dir)

        # online train
        net = net_provider.init_network(pretrained=0)
        net_provider.load(nEpochs, name='vgg16')
        epoch = 1
        net_provider.save(epoch)

        # online test
        net = net_provider.init_network(pretrained=0)
        net_provider.load(nEpochs)
