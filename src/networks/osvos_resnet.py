from copy import deepcopy
from typing import List, Tuple, Union, Callable

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.resnet import BasicBlock, Bottleneck

from layers.osvos_layers import interp_surgery, center_crop
from util.logger import get_logger

log = get_logger(__file__)


class OSVOS_RESNET(nn.Module):
    def __init__(self, pretrained: bool, version=18):
        self.inplanes = 64
        super(OSVOS_RESNET, self).__init__()
        log.info("Constructing OSVOS resnet architecture...")

        block, layers, model_creation = self._match_version(version)

        self.layer_base = self._make_layer_base()
        layer0 = self._make_layer(block, 64, layers[0])
        layer1 = self._make_layer(block, 128, layers[1], stride=2)
        layer2 = self._make_layer(block, 256, layers[2], stride=2)
        layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer_stages = nn.ModuleList([layer0, layer1, layer2, layer3])

        side_input_channels = [64, 128, 256, 512]
        (self.side_prep, self.upscale_side_prep, self.score_dsn,
         self.upscale_score_dsn) = self._make_osvos_layers(side_input_channels)

        self.layer_fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self._initialize_weights()
        if pretrained:
            self._load_from_pytorch(model_creation)

    @staticmethod
    def _match_version(version: int) -> Tuple[Union[BasicBlock, Bottleneck], List[int], Callable]:
        if version == 18:
            block, layers, model_creation = BasicBlock, [2, 2, 2, 2], resnet18
        elif version == 34:
            block, layers, model_creation = BasicBlock, [3, 4, 6, 3], resnet34
        elif version == 50:
            block, layers, model_creation = Bottleneck, [3, 4, 6, 3], resnet50
        elif version == 101:
            block, layers, model_creation = Bottleneck, [3, 4, 23, 3], resnet101
        elif version == 152:
            block, layers, model_creation = Bottleneck, [3, 8, 36, 3], resnet152
        else:
            raise Exception('Invalid version for resnet. Must be one of [18, 34, 50, 101, 152].')
        return block, layers, model_creation

    @staticmethod
    def _make_layer_base() -> nn.Sequential:
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        return nn.Sequential(conv1, bn1, relu, maxpool)

    @staticmethod
    def _make_osvos_layers(side_input_channels: List[int]) -> Tuple[nn.ModuleList, nn.ModuleList,
                                                                    nn.ModuleList, nn.ModuleList]:
        side_prep = nn.ModuleList()
        upscale_side_prep = nn.ModuleList()
        score_dsn = nn.ModuleList()
        upscale_score_dsn = nn.ModuleList()

        for index, channels in enumerate(side_input_channels):
            side_prep.append(nn.Conv2d(channels, 16, kernel_size=3, padding=1))

            upscale_side_prep.append(nn.ConvTranspose2d(16, 16, kernel_size=2 ** (3 + index), stride=2 ** (2 + index),
                                                        bias=False))

            score_dsn.append(nn.Conv2d(16, 1, kernel_size=1, padding=0))
            upscale_score_dsn.append(nn.ConvTranspose2d(1, 1, kernel_size=2 ** (3 + index), stride=2 ** (2 + index),
                                                        bias=False))
        return side_prep, upscale_side_prep, score_dsn, upscale_score_dsn

    def forward(self, x):
        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        x = self.layer_base(x)

        side = []
        side_out = []
        for (layer_stage, layer_side_prep, layer_upscale_side_prep,
             layer_score_dsn, layer_upscale_score_dsn) in zip(self.layer_stages, self.side_prep,
                                                              self.upscale_side_prep,
                                                              self.score_dsn, self.upscale_score_dsn):
            x = layer_stage(x)
            temp_side_prep = layer_side_prep(x)

            temp_upscale = layer_upscale_side_prep(temp_side_prep)
            temp_cropped = center_crop(temp_upscale, crop_h, crop_w)
            side.append(temp_cropped)

            temp_score_dsn = layer_score_dsn(temp_side_prep)
            temp_upscale_ = layer_upscale_score_dsn(temp_score_dsn)
            temp_cropped_ = center_crop(temp_upscale_, crop_h, crop_w)
            side_out.append(temp_cropped_)

        out = torch.cat(side[:], dim=1)
        out = self.layer_fuse(out)
        side_out.append(out)
        return side_out

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # compare osvos_vgg with vgg
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

    def _load_from_pytorch(self, model_creation) -> None:  # model_creation: Callable[[bool], nn.Module]
        log.info('Loading weights from PyTorch Resnet')
        basic_resnet = model_creation(pretrained=True)
        indices_copy_from = self._find_conv_layers(basic_resnet)
        counter = 0
        counter = self._copy_layer(basic_resnet, indices_copy_from, counter, self.layer_base)
        counter = self._copy_layer(basic_resnet, indices_copy_from, counter, self.layer_stages)

    @staticmethod
    def _copy_layer(basic_resnet: nn.Module, indices_copy_from: List[int], counter: int,
                    block: Union[nn.ModuleList, nn.Sequential]):
        for layer in block:
            for module in layer:
                if isinstance(module, nn.Conv2d):
                    module.weight = deepcopy(basic_resnet.features[indices_copy_from[counter]].weight)
                    module.bias = deepcopy(basic_resnet.features[indices_copy_from[counter]].bias)
                    counter += 1
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight = deepcopy(basic_resnet.features[indices_copy_from[counter]].weight)
                    module.bias = deepcopy(basic_resnet.features[indices_copy_from[counter]].bias)
                    counter += 1
        return counter

    @staticmethod
    def _find_conv_layers(basic_resnet: nn.Module) -> List[int]:
        indices = []
        for index in range(len(basic_resnet.features)):
            if isinstance(basic_resnet.features[index], nn.Conv2d):
                indices.append(index)
            elif isinstance(basic_resnet.features[index], nn.BatchNorm2d):
                indices.append(index)
        return indices
