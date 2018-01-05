from pathlib import Path

import torch

from networks.osvos_vgg import OSVOS_VGG
from util import gpu_handler
from util.logger import get_logger

log = get_logger(__file__)


class NetworkProvider:

    def __init__(self, name: str, network_type: type, save_dir: Path) -> None:
        self.name = name
        self.network_type = network_type
        self.save_dir = save_dir
        self.network = None

    def init_network(self, **kwargs) -> object:
        net = self.network_type(**kwargs)
        self.network = net
        return net

    def _get_file_path(self, epoch: int) -> str:
        return str(self.save_dir / '{0}_epoch-{1}.pth'.format(self.name, str(epoch)))

    def load(self, epoch: int) -> None:
        file_path = self._get_file_path(epoch - 1)
        log.info("Loading weights from: {0}".format(file_path))
        self.network.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))

    def save(self, epoch: int) -> None:
        file_path = self._get_file_path(epoch)
        log.info("Saving weights to: {0}".format(file_path))
        torch.save(self.network.state_dict(), file_path)


save_dir = Path('models')
np = NetworkProvider('vgg16', OSVOS_VGG, save_dir)

# parent train
resume_epoch = 0
load_caffe_vgg = False
if resume_epoch == 0:
    if load_caffe_vgg:
        net = np.init_network(pretrained=2)
    else:
        net = np.init_network(pretrained=1)
else:
    net = np.init_network(pretrained=0)
    np.load(resume_epoch)
net = gpu_handler.cast_cuda_if_possible(net, verbose=True)

# parent test
nEpochs = 240
net = np.init_network(pretrained=0)
net = gpu_handler.cast_cuda_if_possible(net, verbose=True)
np.load(nEpochs)
