import torch.nn as nn
from copy import deepcopy
from torch.utils import model_zoo
from torchvision.models import resnet18, ResNet
from torchvision.models.resnet import BasicBlock, model_urls

from layers.osvos_layers import interp_surgery
from util.logger import get_logger

log = get_logger(__file__)


class OSVOS_RESNET(nn.Module):
    def __init__(self, pretrained: bool):
        self.inplanes = 64
        super(OSVOS_RESNET, self).__init__()
        log.info("Constructing OSVOS resnet architecture...")
        block = BasicBlock
        layers = [2, 2, 2, 2]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #
        # self._initialize_weights()
        # if pretrained:
        #     self._load_from_pytorch()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _make_layers_osvos(cfg, in_channels):
        layers = []
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True))
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.extend([conv2d, nn.ReLU(inplace=True)])
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.001)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.zero_()
                m.weight.data = interp_surgery(m)

    def _load_from_pytorch(self) -> None:
        log.info('Loading weights from PyTorch Resnet')
        _resnet = resnet18(pretrained=True)

        inds = self._find_conv_layers(_resnet)
        k = 0
        for i in range(len(self.stages)):
            for j in range(len(self.stages[i])):
                if isinstance(self.stages[i][j], nn.Conv2d):
                    self.stages[i][j].weight = deepcopy(_resnet.features[inds[k]].weight)
                    self.stages[i][j].bias = deepcopy(_resnet.features[inds[k]].bias)
                    k += 1
                elif isinstance(self.stages[i][j], nn.BatchNorm2d):
                    self.stages[i][j].weight = deepcopy(_resnet.features[inds[k]].weight)
                    self.stages[i][j].bias = deepcopy(_resnet.features[inds[k]].bias)
                    k += 1

    @staticmethod
    def _find_conv_layers(_resnet):
        inds = []
        for i in range(len(_resnet.features)):
            if isinstance(_resnet.features[i], nn.Conv2d):
                inds.append(i)
            elif isinstance(_resnet.features[i], nn.BatchNorm2d):
                inds.append(i)
        return inds
