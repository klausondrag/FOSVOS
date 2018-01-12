from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

import torch
from torch import optim
from torch.optim import Optimizer

from networks.osvos_resnet import OSVOS_RESNET
from networks.osvos_vgg import OSVOS_VGG
from util import gpu_handler
from util.logger import get_logger
from .settings import Settings, ParentSettings, OnlineSettings

log = get_logger(__file__)


class NetworkProvider(ABC):

    def __init__(self, name: str, save_dir: Path, network_type: type, settings: Settings) -> None:
        self.name = name
        self.save_dir = save_dir
        self.network_type = network_type
        self._settings = settings
        self.network = None

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

    @abstractmethod
    def load_network_train(self) -> None:
        pass

    @abstractmethod
    def load_network_test(self) -> None:
        pass

    @abstractmethod
    def get_optimizer(self) -> Optimizer:
        pass


class VGG16ParentProvider(NetworkProvider):

    def __init__(self, name: str, save_dir: Path, settings: ParentSettings):
        super(VGG16ParentProvider, self).__init__(name=name, save_dir=save_dir, settings=settings,
                                                  network_type=OSVOS_VGG)

    def load_network_train(self) -> None:
        if self._settings.start_epoch == 0:
            if self._settings.is_loading_vgg_caffe:
                self.init_network(pretrained=2)
            else:
                self.init_network(pretrained=1)
        else:
            self.init_network(pretrained=0)
            self.load(self._settings.start_epoch)

    def load_network_test(self) -> None:
        self.init_network(pretrained=0)
        self.load(self._settings.n_epochs)

    def get_optimizer(self, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                      momentum: float = 0.9) -> Optimizer:
        net = self.network
        optimizer = optim.SGD([
            {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]],
             'weight_decay': weight_decay,
             'initial_lr': learning_rate},
            {'params': [pr[1] for pr in net.stages.named_parameters() if 'bias' in pr[0]], 'lr': 2 * learning_rate,
             'initial_lr': 2 * learning_rate},
            {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]],
             'weight_decay': weight_decay,
             'initial_lr': learning_rate},
            {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]], 'lr': 2 * learning_rate,
             'initial_lr': 2 * learning_rate},
            {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'weight' in pr[0]],
             'lr': learning_rate / 10,
             'weight_decay': weight_decay, 'initial_lr': learning_rate / 10},
            {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'bias' in pr[0]],
             'lr': 2 * learning_rate / 10,
             'initial_lr': 2 * learning_rate / 10},
            {'params': [pr[1] for pr in net.upscale.named_parameters() if 'weight' in pr[0]], 'lr': 0, 'initial_lr': 0},
            {'params': [pr[1] for pr in net.upscale_.named_parameters() if 'weight' in pr[0]], 'lr': 0,
             'initial_lr': 0},
            {'params': net.fuse.weight, 'lr': learning_rate / 100, 'initial_lr': learning_rate / 100,
             'weight_decay': weight_decay},
            {'params': net.fuse.bias, 'lr': 2 * learning_rate / 100, 'initial_lr': 2 * learning_rate / 100},
        ], lr=learning_rate, momentum=momentum)
        return optimizer


class VGG16OnlineProvider(NetworkProvider):

    def __init__(self, name: str, save_dir: Path, settings: OnlineSettings):
        super(VGG16OnlineProvider, self).__init__(name=name, save_dir=save_dir, settings=settings,
                                                  network_type=OSVOS_VGG)

    def load_network_train(self) -> None:
        self.init_network(pretrained=0)
        self.load(self._settings.parent_epoch, name=self._settings.parent_name)

    def load_network_test(self) -> None:
        self.init_network(pretrained=0)
        if self._settings.is_training:
            self.load(self._settings.n_epochs)
        else:
            self.load(self._settings.parent_epoch, name=self._settings.parent_name)

    def get_optimizer(self, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                      momentum: float = 0.9) -> Optimizer:
        net = self.network
        optimizer = optim.SGD([
            {'params': [pr[1] for pr in net.stages.named_parameters() if 'weight' in pr[0]],
             'weight_decay': weight_decay},
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


class ResNet18ParentProvider(NetworkProvider):

    def __init__(self, name: str, save_dir: Path, settings: ParentSettings):
        super(ResNet18ParentProvider, self).__init__(name=name, save_dir=save_dir, settings=settings,
                                                     network_type=OSVOS_RESNET)

    def load_network_train(self) -> None:
        if self._settings.start_epoch == 0:
            self.init_network(pretrained=True)
        else:
            self.init_network(pretrained=False)
            self.load(self._settings.start_epoch)

    def load_network_test(self) -> None:
        self.init_network(pretrained=False)
        self.load(self._settings.n_epochs)

    def get_optimizer(self, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                      momentum: float = 0.9) -> Optimizer:
        optimizer = optim.SGD(self.network.parameters(), lr=learning_rate, momentum=momentum)
        return optimizer


class ResNet18OnlineProvider(NetworkProvider):

    def __init__(self, name: str, save_dir: Path, settings: OnlineSettings):
        super(ResNet18OnlineProvider, self).__init__(name=name, save_dir=save_dir, settings=settings,
                                                     network_type=OSVOS_RESNET)

    def load_network_train(self) -> None:
        self.init_network(pretrained=False)
        self.load(self._settings.parent_epoch, name=self._settings.parent_name)

    def load_network_test(self) -> None:
        self.init_network(pretrained=False)
        if self._settings.is_training:
            self.load(self._settings.n_epochs)
        else:
            self.load(self._settings.parent_epoch, name=self._settings.parent_name)

    def get_optimizer(self, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                      momentum: float = 0.9) -> Optimizer:
        optimizer = optim.SGD(self.network.parameters(), lr=learning_rate, momentum=momentum)
        return optimizer
