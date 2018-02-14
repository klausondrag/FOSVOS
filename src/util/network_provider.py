from pathlib import Path
from typing import Optional, Dict
from abc import ABC, abstractmethod

import torch
from torch import optim
from torch.optim import Optimizer

from networks.osvos_resnet import OSVOS_RESNET
from networks.osvos_vgg import OSVOS_VGG
from util import gpu_handler
from util.logger import get_logger
from .settings import Settings, OfflineSettings, OnlineSettings

log = get_logger(__file__)


class NetworkProvider(ABC):

    def __init__(self, name: str, save_dir: Path, network_type: type, settings: Settings,
                 variant_offline: Optional[int] = None) -> None:
        self.name = name
        self.save_dir = save_dir
        self.network_type = network_type
        self._settings = settings
        self.variant_offline = variant_offline
        self.network = None

    def init_network(self, **kwargs) -> object:
        net = self.network_type(**kwargs)
        net = gpu_handler.cast_cuda_if_possible(net, verbose=True)
        self.network = net
        return net

    def _get_file_path(self, epoch: int, sequence: Optional[str] = None) -> Path:
        model_name = self.name
        if self.variant_offline is not None:
            model_name += '_' + str(self.variant_offline)
        if sequence is not None:
            model_name += '_' + sequence

        file_path = self.save_dir / '{0}_epoch-{1}.pth'.format(model_name, str(epoch))
        return file_path

    def load_model(self, epoch: int, sequence: Optional[str] = None) -> None:
        file_path = self._get_file_path(epoch - 1, sequence)
        log.info("Loading weights from: {0}".format(str(file_path)))
        if not file_path.exists():
            log.error('Model {0} does not exist!'.format(str(file_path)))
        self.network.load_state_dict(torch.load(str(file_path), map_location=lambda storage, loc: storage))

    def save_model(self, epoch: int, sequence: Optional[str] = None) -> None:
        file_path = str(self._get_file_path(epoch, sequence))
        log.info("Saving weights to: {0}".format(file_path))
        torch.save(self.network.state_dict(), file_path)

    @abstractmethod
    def load_network_train(self) -> None:
        pass

    @abstractmethod
    def load_network_test(self, sequence: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def get_optimizer(self) -> optim.SGD:
        pass


class VGGOfflineProvider(NetworkProvider):

    def __init__(self, name: str, save_dir: Path, settings: OfflineSettings, variant_offline: Optional[int] = None):
        super(VGGOfflineProvider, self).__init__(name=name, save_dir=save_dir, settings=settings,
                                                 network_type=OSVOS_VGG, variant_offline=variant_offline)

    def load_network_train(self) -> None:
        if self._settings.start_epoch == 0:
            if self._settings.is_loading_vgg_caffe:
                self.init_network(pretrained=2)
            else:
                self.init_network(pretrained=1)
        else:
            self.init_network(pretrained=0)
            self.load_model(self._settings.start_epoch)

    def load_network_test(self, sequence: Optional[str] = None) -> None:
        self.init_network(pretrained=0)
        self.load_model(self._settings.n_epochs, sequence=sequence)

    def get_optimizer(self, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                      momentum: float = 0.9) -> optim.SGD:
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


class VGGOnlineProvider(NetworkProvider):

    def __init__(self, name: str, save_dir: Path, settings: OnlineSettings, variant_offline: Optional[int] = None):
        super(VGGOnlineProvider, self).__init__(name=name, save_dir=save_dir, settings=settings,
                                                network_type=OSVOS_VGG, variant_offline=variant_offline)

    def load_network_train(self) -> None:
        self.init_network(pretrained=0)
        self.load_model(self._settings.offline_epoch)

    def load_network_test(self, sequence: Optional[str] = None) -> None:
        self.init_network(pretrained=0)
        self.load_model(self._settings.n_epochs, sequence=sequence)

    def get_optimizer(self, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                      momentum: float = 0.9) -> optim.SGD:
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


class ResNetOfflineProvider(NetworkProvider):

    def __init__(self, name: str, save_dir: Path, settings: OfflineSettings, variant: Optional[int] = None,
                 version: int = 18):
        super(ResNetOfflineProvider, self).__init__(name=name, save_dir=save_dir, settings=settings,
                                                    network_type=OSVOS_RESNET, variant=variant)
        self.version = version

    def load_network_train(self) -> None:
        if self._settings.start_epoch == 0:
            self.init_network(pretrained=True, version=self.version)
        else:
            self.init_network(pretrained=False, version=self.version)
            self.load_model(self._settings.start_epoch)

    def load_network_test(self, sequence: Optional[str] = None) -> None:
        self.init_network(pretrained=False, version=self.version)
        self.load_model(self._settings.n_epochs, sequence=sequence)

    def get_optimizer(self, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                      momentum: float = 0.9) -> Optimizer:
        net = self.network
        default_var = optim.SGD([
            {'params': [pr[1] for pr in net.layer_stages.named_parameters() if 'weight' in pr[0]],
             'weight_decay': weight_decay, 'initial_lr': learning_rate},
            {'params': [pr[1] for pr in net.layer_stages.named_parameters() if 'bias' in pr[0]],
             'lr': 2 * learning_rate, 'initial_lr': 2 * learning_rate},
            {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]],
             'weight_decay': weight_decay, 'initial_lr': learning_rate},
            {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]],
             'lr': 2 * learning_rate, 'initial_lr': 2 * learning_rate},
            {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'weight' in pr[0]],
             'lr': learning_rate / 10, 'weight_decay': weight_decay, 'initial_lr': learning_rate / 10},
            {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'bias' in pr[0]],
             'lr': 2 * learning_rate / 10, 'initial_lr': 2 * learning_rate / 10},
            {'params': [pr[1] for pr in net.upscale_side_prep.named_parameters() if 'weight' in pr[0]],
             'lr': 0, 'initial_lr': 0},
            {'params': [pr[1] for pr in net.upscale_score_dsn.named_parameters() if 'weight' in pr[0]],
             'lr': 0, 'initial_lr': 0},
            {'params': net.layer_fuse.weight, 'weight_decay': weight_decay,
             'lr': learning_rate / 100, 'initial_lr': learning_rate / 100},
            {'params': net.layer_fuse.bias, 'lr': 2 * learning_rate / 100, 'initial_lr': 2 * learning_rate / 100},
        ], lr=learning_rate, momentum=momentum)

        if self.variant is None:
            optimizer = default_var
        else:
            from .variants import variants
            v = variants[self.variant][1]
            params = [net.layer_stages.parameters, net.side_prep.parameters, net.score_dsn.parameters,
                      net.upscale_side_prep.parameters, net.upscale_score_dsn.parameters, net.layer_fuse.parameters]
            if v == 0:
                optimizer = default_var
            elif v == 1:
                optimizer = optim.SGD(params)
            elif v == 2:
                optimizer = optim.Adam(params)
        return optimizer


class ResNetOnlineProvider(NetworkProvider):

    def __init__(self, name: str, save_dir: Path, settings: OnlineSettings, variant: Optional[int] = None,
                 version: int = 18):
        super(ResNetOnlineProvider, self).__init__(name=name, save_dir=save_dir, settings=settings,
                                                   network_type=OSVOS_RESNET, variant=variant)
        self.version = version

    def load_network_train(self) -> None:
        self.init_network(pretrained=False, version=self.version)
        self.load_model(self._settings.offline_epoch)

    def load_network_test(self, sequence: Optional[str] = None) -> None:
        self.init_network(pretrained=False, version=self.version)
        self.load_model(self._settings.n_epochs, sequence=sequence)

    def get_optimizer(self, learning_rate: float = 1e-8, weight_decay: float = 0.0002,
                      momentum: float = 0.9) -> Optimizer:
        net = self.network
        default_var = optim.SGD([
            {'params': [pr[1] for pr in net.layer_stages.named_parameters() if 'weight' in pr[0]],
             'weight_decay': weight_decay, 'initial_lr': learning_rate},
            {'params': [pr[1] for pr in net.layer_stages.named_parameters() if 'bias' in pr[0]],
             'lr': 2 * learning_rate, 'initial_lr': 2 * learning_rate},
            {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]],
             'weight_decay': weight_decay, 'initial_lr': learning_rate},
            {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]],
             'lr': 2 * learning_rate, 'initial_lr': 2 * learning_rate},
            {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'weight' in pr[0]],
             'lr': learning_rate / 10, 'weight_decay': weight_decay, 'initial_lr': learning_rate / 10},
            {'params': [pr[1] for pr in net.score_dsn.named_parameters() if 'bias' in pr[0]],
             'lr': 2 * learning_rate / 10, 'initial_lr': 2 * learning_rate / 10},
            {'params': [pr[1] for pr in net.upscale_side_prep.named_parameters() if 'weight' in pr[0]],
             'lr': 0, 'initial_lr': 0},
            {'params': [pr[1] for pr in net.upscale_score_dsn.named_parameters() if 'weight' in pr[0]],
             'lr': 0, 'initial_lr': 0},
            {'params': net.layer_fuse.weight, 'weight_decay': weight_decay,
             'lr': learning_rate / 100, 'initial_lr': learning_rate / 100},
            {'params': net.layer_fuse.bias, 'lr': 2 * learning_rate / 100, 'initial_lr': 2 * learning_rate / 100},
        ], lr=learning_rate, momentum=momentum)

        if self.variant is None:
            optimizer = default_var
        else:
            from .variants import variants
            v = variants[self.variant][1]
            params = [net.layer_stages.parameters, net.side_prep.parameters, net.score_dsn.parameters,
                      net.upscale_side_prep.parameters, net.upscale_score_dsn.parameters, net.layer_fuse.parameters]
            if v == 0:
                optimizer = default_var
            elif v == 1:
                optimizer = optim.SGD([
                    {'params': [pr[1] for pr in net.layer_stages.named_parameters() if 'weight' in pr[0]],
                     'weight_decay': weight_decay},
                    {'params': [pr[1] for pr in net.layer_stages.named_parameters() if 'bias' in pr[0]],
                     'lr': learning_rate * 2},
                    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'weight' in pr[0]],
                     'weight_decay': weight_decay},
                    {'params': [pr[1] for pr in net.side_prep.named_parameters() if 'bias' in pr[0]],
                     'lr': learning_rate * 2},
                    {'params': [pr[1] for pr in net.upscale_side_prep.named_parameters() if 'weight' in pr[0]],
                     'lr': 0},
                    {'params': [pr[1] for pr in net.upscale_score_dsn.named_parameters() if 'weight' in pr[0]],
                     'lr': 0},
                    {'params': net.layer_fuse.weight, 'lr': learning_rate / 100, 'weight_decay': weight_decay},
                    {'params': net.layer_fuse.bias, 'lr': 2 * learning_rate / 100},
                ], lr=learning_rate, momentum=momentum)
            elif v == 2:
                optimizer = optim.SGD(params)
            elif v == 3:
                optimizer = optim.Adam(params)
        return optimizer


provider_mapping = {
    ('offline', 'vgg16'): VGGOfflineProvider,
    ('online', 'vgg16'): VGGOnlineProvider,
    ('offline', 'resnet18'): ResNetOfflineProvider,
    ('online', 'resnet18'): ResNetOnlineProvider,
    ('offline', 'resnet34'): ResNetOfflineProvider,
    ('online', 'resnet34'): ResNetOnlineProvider
}  # type: Dict[(str, str), NetworkProvider]
