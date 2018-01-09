import torch.nn as nn
from torchvision.models import resnet18

from util.logger import get_logger

log = get_logger(__file__)


class OSVOS_RESNET(nn.Module):
    def __init__(self, pretrained: bool):
        super(OSVOS_RESNET, self).__init__()
        log.info("Constructing OSVOS resnet architecture...")
        self.resnet = resnet18(pretrained=pretrained)

    def forward(self, x):
        return self.resnet.forward(x)
