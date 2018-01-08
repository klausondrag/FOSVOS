from pathlib import Path

import torch

from networks.osvos_vgg import OSVOS_VGG
from util import gpu_handler
from util.logger import get_logger

log = get_logger(__file__)


class NetworkProvider:

    def __init__(self, name: str, network_type: type, save_dir: Path, name_parent: str = None) -> None:
        self.name = name
        self.network_type = network_type
        self.save_dir = save_dir
        self.network = None
        self.name_parent = name_parent

    def init_network(self, **kwargs) -> object:
        net = self.network_type(**kwargs)
        net = gpu_handler.cast_cuda_if_possible(net, verbose=True)
        self.network = net
        return net

    def _get_file_path(self, epoch: int, use_parent: bool) -> str:
        model_name = self.name_parent if use_parent else self.name
        return str(self.save_dir / '{0}_epoch-{1}.pth'.format(model_name, str(epoch)))

    def load(self, epoch: int, use_parent: bool = False) -> None:
        file_path = self._get_file_path(epoch - 1, use_parent)
        log.info("Loading weights from: {0}".format(file_path))
        self.network.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))

    def save(self, epoch: int, use_parent: bool = False) -> None:
        file_path = self._get_file_path(epoch, use_parent)
        log.info("Saving weights to: {0}".format(file_path))
        torch.save(self.network.state_dict(), file_path)



# parent
save_dir = Path('models')
np = NetworkProvider('vgg16', OSVOS_VGG, save_dir)

resume_epoch = 0
load_caffe_vgg = False
nEpochs = 240

# parent train
if resume_epoch == 0:
    if load_caffe_vgg:
        net = np.init_network(pretrained=2)
    else:
        net = np.init_network(pretrained=1)
else:
    net = np.init_network(pretrained=0)
    np.load(resume_epoch)

epoch = 1
np.save(epoch)

# parent test
net = net_provider.init_network(pretrained=0)
net_provider.load(nEpochs)

# online
save_dir = Path('models')
net_provider = NetworkProvider('vgg16_blackswan', OSVOS_VGG, save_dir, name_parent='vgg16')

# online train
net = net_provider.init_network(pretrained=0)
net_provider.load(nEpochs, use_parent=True)
epoch = 1
net_provider.save(epoch)

# online test
net = net_provider.init_network(pretrained=0)
net_provider.load(nEpochs, use_parent=True)
