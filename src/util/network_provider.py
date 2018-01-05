from pathlib import Path

import torch

from networks.osvos_vgg import OSVOS_VGG
from util.logger import get_logger

log = get_logger(__file__)


class NetworkProvider:

    def __init__(self, name: str, network_type: type, save_dir: Path, **kwargs) -> None:
        self.name = name
        self.network_type = network_type
        self.save_dir = save_dir
        self.arguments = kwargs
        self.network = None

    def _get_file_path(self, epoch: int) -> str:
        return str(self.save_dir / '{0}_epoch-{1}.pth'.format(self.name, str(epoch - 1)))

    def load(self, epoch: int) -> None:
        file_path = self._get_file_path(epoch)
        log.info("Updating weights from: {0}".format(file_path))
        net = self.network_type(**self.arguments)
        net.load_state_dict(torch.load(file_path, map_location=lambda storage, loc: storage))
        self.network = net

# save_dir = Path('models')
# np = NetworkProvider('vgg16', OSVOS_VGG, save_dir)
# modelName = str(exp_name)
# if resume_epoch == 0:
#     if load_caffe_vgg:
#         net = vo.OSVOS_VGG(pretrained=2)
#     else:
#         net = vo.OSVOS_VGG(pretrained=1)
# else:
