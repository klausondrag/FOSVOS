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
    def __init__(self, pretrained: bool, version: int = 18, n_channels_input: int = 3, n_channels_output: int = 1,
                 scale_down_exponential: int = 0, is_mode_mimic: bool = False):
        self.is_mode_mimic = is_mode_mimic
        self.scale_down_exponential = scale_down_exponential
        self.inplanes = 64 // (2 ** scale_down_exponential)
        super(OSVOS_RESNET, self).__init__()
        log.info("Constructing OSVOS resnet architecture...")

        block, layers, model_creation = self._match_version(version)
        n_channels_side_inputs = [64, 128, 256, 512]
        n_channels_side_inputs = [i // (2 ** scale_down_exponential)
                                  for i in n_channels_side_inputs]

        self.layer_base = self._make_layer_base(n_channels_input=n_channels_input,
                                                n_channels_output=n_channels_side_inputs[0])

        self.layer_stages = self._make_layer_stages(block, layers, n_channels_side_inputs)

        (self.side_prep, self.upscale_side_prep, self.score_dsn,
         self.upscale_score_dsn, self.layer_fuse) = self._make_osvos_layers(channels_side_input=n_channels_side_inputs,
                                                                            n_channels_output=n_channels_output)

        self._initialize_weights()
        if pretrained:
            self._load_from_pytorch(model_creation)

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
        if self.training and self.is_mode_mimic:
            return torch.cat(side_out)
        else:
            return side_out

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
    def _make_layer_base(n_channels_input: int, n_channels_output: int) -> nn.Sequential:
        conv1 = nn.Conv2d(n_channels_input, n_channels_output, kernel_size=7, stride=2, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(n_channels_output)
        relu = nn.ReLU(inplace=True)
        maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        return nn.Sequential(conv1, bn1, relu, maxpool)

    def _make_layer_stages(self, block: Union[BasicBlock, Bottleneck], layers: List[int],
                           n_channels_side_inputs: List[int]) -> nn.ModuleList:
        layer0 = self._make_layer(block, n_channels_side_inputs[0], layers[0])
        layer1 = self._make_layer(block, n_channels_side_inputs[1], layers[1], stride=2)
        layer2 = self._make_layer(block, n_channels_side_inputs[2], layers[2], stride=2)
        layer3 = self._make_layer(block, n_channels_side_inputs[3], layers[3], stride=2)
        layer_stages = nn.ModuleList([layer0, layer1, layer2, layer3])
        return layer_stages

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

    @staticmethod
    def _make_osvos_layers(channels_side_input: List[int], n_channels_output: int,
                           n_channels_output_side_prep: int = 16,
                           n_channels_output_upscale_side_prep: int = 16) -> Tuple[nn.ModuleList, nn.ModuleList,
                                                                                   nn.ModuleList, nn.ModuleList,
                                                                                   nn.Module]:
        side_prep = nn.ModuleList()
        upscale_side_prep = nn.ModuleList()
        score_dsn = nn.ModuleList()
        upscale_score_dsn = nn.ModuleList()

        for index, n_channels in enumerate(channels_side_input):
            side_prep.append(nn.Conv2d(n_channels, n_channels_output_side_prep, kernel_size=3, padding=1))

            upscale_side_prep.append(nn.ConvTranspose2d(n_channels_output_side_prep,
                                                        n_channels_output_upscale_side_prep,
                                                        kernel_size=2 ** (3 + index), stride=2 ** (2 + index),
                                                        bias=False))

            score_dsn.append(nn.Conv2d(n_channels_output_side_prep, n_channels_output, kernel_size=1, padding=0))
            upscale_score_dsn.append(nn.ConvTranspose2d(n_channels_output, n_channels_output,
                                                        kernel_size=2 ** (3 + index), stride=2 ** (2 + index),
                                                        bias=False))

        n_channels_fuse_input = n_channels_output_upscale_side_prep * len(channels_side_input)
        layer_fuse = nn.Conv2d(n_channels_fuse_input, n_channels_output, kernel_size=1, padding=0)

        return side_prep, upscale_side_prep, score_dsn, upscale_score_dsn, layer_fuse

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
        resnet = model_creation(pretrained=True)
        resnet_base = [resnet.conv1, resnet.bn1]
        resnet_stages = [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4]

        self._copy_weights(resnet_base, self.layer_base)
        for block_src, block_dest in zip(resnet_stages, self.layer_stages):
            self._copy_weights(block_src, block_dest)

    @staticmethod
    def _copy_weights(block_resnet: Union[List[nn.Module], nn.Sequential],
                      block_osvos: Union[List[nn.Module], nn.Sequential]) -> None:
        for module_src, module_dest in zip(block_resnet, block_osvos):
            if isinstance(module_src, nn.Conv2d) or isinstance(module_src, nn.BatchNorm2d):
                module_dest.weight = deepcopy(module_src.weight)
                module_dest.bias = deepcopy(module_src.bias)


class BasicBlockDummy(nn.Module):
    expansion = 1

    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample, stride):
        super(BasicBlockDummy, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
