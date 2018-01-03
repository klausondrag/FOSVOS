import math
import os
from copy import deepcopy
from pathlib import Path as P

import scipy.io
import torch
import torch.nn as nn
import torch.nn.modules as modules
from torchvision.models import vgg16

from config.mypath import Path
from layers.osvos_layers import center_crop, interp_surgery
from util.logger import get_logger

log = get_logger(__file__)


class OSVOS(nn.Module):
    def __init__(self, pretrained=1):
        super(OSVOS, self).__init__()
        lay_list = [[64, 64],
                    ['M', 128, 128],
                    ['M', 256, 256, 256],
                    ['M', 512, 512, 512],
                    ['M', 512, 512, 512]]
        in_channels = [3, 64, 128, 256, 512]

        log.info("Constructing OSVOS architecture...")
        stages = modules.ModuleList()
        side_prep = modules.ModuleList()
        score_dsn = modules.ModuleList()
        upscale = modules.ModuleList()
        upscale_ = modules.ModuleList()

        # Construct the network
        for i in range(0, len(lay_list)):
            # Make the layers of the stages
            stages.append(self._make_layers_osvos(lay_list[i], in_channels[i]))

            # Attention, side_prep and score_dsn start from layer 2
            if i > 0:
                # Make the layers of the preparation step
                side_prep.append(nn.Conv2d(lay_list[i][-1], 16, kernel_size=3, padding=1))

                # Make the layers of the score_dsn step
                score_dsn.append(nn.Conv2d(16, 1, kernel_size=1, padding=0))
                # upscale.append(nn.Upsample(scale_factor=2 ** i, mode='bilinear'))
                upscale_.append(nn.ConvTranspose2d(1, 1, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))
                upscale.append(nn.ConvTranspose2d(16, 16, kernel_size=2 ** (1 + i), stride=2 ** i, bias=False))

        self.upscale = upscale
        self.upscale_ = upscale_
        self.stages = stages
        self.side_prep = side_prep
        self.score_dsn = score_dsn

        self.fuse = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        log.info("Initializing weights")
        self._initialize_weights(pretrained)

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

    def forward(self, x):
        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])
        x = self.stages[0](x)

        side = []
        side_out = []
        for i in range(1, len(self.stages)):
            x = self.stages[i](x)
            side_temp = self.side_prep[i - 1](x)
            side.append(center_crop(self.upscale[i - 1](side_temp), crop_h, crop_w))
            side_out.append(center_crop(self.upscale_[i - 1](self.score_dsn[i - 1](side_temp)), crop_h, crop_w))

        out = torch.cat(side[:], dim=1)
        out = self.fuse(out)
        side_out.append(out)
        return side_out

    def _initialize_weights(self, pretrained):
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

        if pretrained == 1:
            self._load_from_pytorch()
        elif pretrained == 2:
            self._load_from_caffe()

    def _load_from_pytorch(self) -> None:
        log.info('Loading weights from PyTorch VGG')
        _vgg = vgg16(pretrained=True)

        inds = self._find_conv_layers(_vgg)
        k = 0
        for i in range(len(self.stages)):
            for j in range(len(self.stages[i])):
                if isinstance(self.stages[i][j], nn.Conv2d):
                    self.stages[i][j].weight = deepcopy(_vgg.features[inds[k]].weight)
                    self.stages[i][j].bias = deepcopy(_vgg.features[inds[k]].bias)
                    k += 1

    @staticmethod
    def _find_conv_layers(_vgg):
        inds = []
        for i in range(len(_vgg.features)):
            if isinstance(_vgg.features[i], nn.Conv2d):
                inds.append(i)
        return inds

    def _load_from_caffe(self) -> None:
        log.info('Loading weights from Caffe VGG')
        caffe_weights = scipy.io.loadmat(os.path.join(Path.models_dir(), 'vgg_hed_caffe.mat'))

        caffe_ind = 0
        for ind, layer in enumerate(self.stages.parameters()):
            if ind % 2 == 0:
                c_w = torch.from_numpy(caffe_weights['weights'][0][caffe_ind].transpose())
                assert (layer.data.shape == c_w.shape)
                layer.data = c_w
            else:
                c_b = torch.from_numpy(caffe_weights['biases'][0][caffe_ind][:, 0])
                assert (layer.data.shape == c_b.shape)
                layer.data = c_b
                caffe_ind += 1
