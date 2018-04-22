# code for paper "Pruning Convolutional Neural Networks for Resource Efficient Inference"
# code adopted from https://github.com/eeric/channel_prune
# which itself is adopted from https://github.com/jacobgil/pytorch-pruning

from pathlib import Path
import sys
from typing import Optional
import operator
import heapq

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils import data
from torch import optim

if sys.path[0] != '..':
    sys.path.insert(0, '..')

path_ros = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if path_ros in sys.path:
    del sys.path[sys.path.index(path_ros)]

from networks.osvos_resnet import OSVOS_RESNET
from util import io_helper, experiment_helper
from layers.osvos_layers import class_balanced_cross_entropy_loss, center_crop

import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--n-epochs-train', default=20, type=int, help='version to try')
parser.add_argument('--n-epochs-finetune', default=20, type=int, help='version to try')
parser.add_argument('--percentage-prune', default=50, type=int, help='version to try')
parser.add_argument('--gpu-id', default=1, type=int, help='The gpu id to use')
args = parser.parse_args()

print('GPU:', args.gpu_id)
torch.cuda.set_device(device=args.gpu_id)

p = Path('../results/resnet18/11/11_min_{0}_{1}_{2}'.format(args.n_epochs_train,
                                                            args.n_epochs_finetune,
                                                            args.percentage_prune))

print(args.n_epochs_train, args.n_epochs_finetune, args.percentage_prune, str(p))


def get_net() -> nn.Module:
    net = OSVOS_RESNET(pretrained=False)
    path_model = Path('../models/resnet18_11_11_blackswan_epoch-9999.pth')
    parameters = torch.load(str(path_model), map_location=lambda storage, loc: storage)
    net.load_state_dict(parameters)
    net = net.cuda()
    return net


net = get_net()


def total_num_filters(net: OSVOS_RESNET) -> int:
    n_filters = 0
    for m in net.layer_base.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n_filters += m.out_channels
    for m in net.layer_stages.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n_filters += m.out_channels
    return n_filters


def total_num_filters_old(net: nn.Module) -> int:
    n_filters = 0
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            n_filters += m.out_channels
        if n_filters > 0:
            return n_filters
    return n_filters


n_filters = total_num_filters(net)
n_filters_to_prune_per_iter = 256
n_filters_to_prune_per_iter = 512
n_iterations = 1 + int(n_filters / n_filters_to_prune_per_iter * args.percentage_prune / 100)

print('Filters in model:', n_filters)
print('Prune n filters per iteration:', n_filters_to_prune_per_iter)
print('Number of iterations:', n_iterations)


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


class FilterPruner:
    def __init__(self, net: nn.Module):
        self.net = net
        self.reset()

    def reset(self) -> None:
        self.filter_ranks = {}

    def forward(self, x):
        self.activations = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = {}
        activation_index = 0
        kk = 0

        crop_h, crop_w = int(x.size()[-2]), int(x.size()[-1])

        for l in self.net.layer_base:
            x = l(x)
            if isinstance(l, nn.modules.conv.Conv2d):
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = kk
                activation_index += 1
                kk += 1
        # x = self.net.layer_base(x)

        side = []
        side_out = []
        for (layer_stage, layer_side_prep, layer_upscale_side_prep,
             layer_score_dsn, layer_upscale_score_dsn) in zip(self.net.layer_stages, self.net.side_prep,
                                                              self.net.upscale_side_prep,
                                                              self.net.score_dsn, self.net.upscale_score_dsn):
            # x = layer_stage(x)
            for kt in range(2):
                residual = x
                x = layer_stage[kt].conv1(x)
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = kk
                activation_index += 1
                kk += 1
                x = layer_stage[kt].bn1(x)
                x = layer_stage[kt].relu(x)
                x = layer_stage[kt].conv2(x)
                x.register_hook(self.compute_rank)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = kk
                activation_index += 1
                kk += 1
                x = layer_stage[kt].bn2(x)

                if layer_stage[kt].downsample is not None:
                    residual = layer_stage[kt].downsample[0](residual)
                    # TODO: figure out activation to layer
                    # x.register_hook(self.compute_rank)
                    # self.activations.append(x)
                    # self.activation_to_layer[activation_index] = kk
                    # activation_index += 1
                    # kk += 1
                    residual = layer_stage[kt].downsample[1](residual)
                x += residual
                x = layer_stage[kt].relu(x)

            temp_side_prep = layer_side_prep(x)

            temp_upscale = layer_upscale_side_prep(temp_side_prep)
            temp_cropped = center_crop(temp_upscale, crop_h, crop_w)
            side.append(temp_cropped)

            temp_score_dsn = layer_score_dsn(temp_side_prep)
            temp_upscale_ = layer_upscale_score_dsn(temp_score_dsn)
            temp_cropped_ = center_crop(temp_upscale_, crop_h, crop_w)
            side_out.append(temp_cropped_)

        out = torch.cat(side[:], dim=1)
        out = self.net.layer_fuse(out)
        side_out.append(out)
        return side_out

    def compute_rank(self, grad):
        activation_index = len(self.activations) - self.grad_index - 1
        activation = self.activations[activation_index]
        values = torch.sum((activation * grad), dim=0, keepdim=True).sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)[
                 0, :, 0, 0].data

        # Normalize the rank by the filter dimensions
        values = values / (activation.size(0) * activation.size(2) * activation.size(3))

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_().cuda()
            # self.filter_ranks[activation_index] = torch.FloatTensor(activation.size(1)).zero_()

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            v = torch.abs(self.filter_ranks[i])
            divisor = np.sqrt(torch.sum(v * v))
            if divisor > 1e-5:
                v = v / divisor
                self.filter_ranks[i] = v.cpu()

    def lowest_ranking_filters(self, n_filters_to_prune_per_iter):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

        return heapq.nsmallest(n_filters_to_prune_per_iter, data, operator.itemgetter(2))

    def get_prunning_plan(self, n_filters_to_prune_per_iter):
        filters_to_prune = self.lowest_ranking_filters(n_filters_to_prune_per_iter)

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for (l, f, _) in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        for l in filters_to_prune_per_layer:
            filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
            for i in range(len(filters_to_prune_per_layer[l])):
                filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        return filters_to_prune


pruner = FilterPruner(net)

# data_loader = io_helper.get_data_loader_test(Path('/home/klaus/dev/datasets/DAVIS'), batch_size=1, seq_name='blackswan')
data_loader = io_helper.get_data_loader_test(Path('/usr/stud/ondrag/DAVIS'), batch_size=1, seq_name='blackswan')


def train(pruner: FilterPruner, data_loader: data.DataLoader, n_epochs: Optional[int] = 1) -> None:
    for epoch in range(n_epochs):
        for minibatch in data_loader:
            pruner.net.zero_grad()
            inputs, gts = minibatch['image'], minibatch['gt']
            inputs, gts = Variable(inputs), Variable(gts)
            inputs, gts = inputs.cuda(), gts.cuda()

            outputs = pruner.forward(inputs)
            loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
            loss.backward()


def fine_tune(net: nn.Module(), data_loader: data.DataLoader, n_epochs: Optional[int] = 1) -> None:
    optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.0002)
    avg_grad_every_n = 5
    counter_gradient = 0

    for epoch in range(n_epochs):
        for minibatch in data_loader:
            net.zero_grad()
            inputs, gts = minibatch['image'], minibatch['gt']
            inputs, gts = Variable(inputs), Variable(gts)
            inputs, gts = inputs.cuda(), gts.cuda()

            outputs = net.forward(inputs)
            loss = class_balanced_cross_entropy_loss(outputs[-1], gts, size_average=False)
            loss /= avg_grad_every_n
            loss.backward()
            counter_gradient += 1

            if counter_gradient % avg_grad_every_n == 0:
                counter_gradient = 0
                optimizer.step()
                optimizer.zero_grad()


def prune_resnet18_conv_layer(net: OSVOS_RESNET, layer_index: int, filter_index: int) -> OSVOS_RESNET:
    # fix missing bias

    if layer_index == 0:
        conv_old = net.layer_base[0]
        batchnorm_old = net.layer_base[1]
        conv_next_old = net.layer_stages[0][0].conv1
        downsample_1_old = net.layer_stages[0][0].downsample

        conv_new = prune_convolution(conv_old, filter_index, is_reducing_channels_out=True)
        batchnorm_new = prune_batchnorm(batchnorm_old, filter_index)
        conv_next_new = prune_convolution(conv_next_old, filter_index, is_reducing_channels_out=False)
        if downsample_1_old is None:
            n_channels_out = net.layer_stages[0][0].bn2.num_features
            downsample_1_new = nn.Sequential(nn.Conv2d(conv_next_new.in_channels, n_channels_out,
                                                       kernel_size=1, stride=1, bias=False),
                                             nn.BatchNorm2d(n_channels_out))
        else:
            conv_downsample_1_new = prune_convolution(downsample_1_old[0], filter_index, is_reducing_channels_out=False)
            downsample_1_new = nn.Sequential(conv_downsample_1_new, downsample_1_old[1])

        net.layer_base = nn.Sequential(conv_new, batchnorm_new, *list(net.layer_base.children())[2:])

        bb_old = net.layer_stages[0][0]
        bb_new = BasicBlockDummy(conv_next_new, bb_old.bn1, bb_old.relu, bb_old.conv2, bb_old.bn2, downsample_1_new,
                                 bb_old.stride)
        net.layer_stages[0] = nn.Sequential(bb_new, net.layer_stages[0][1])

    else:
        index_stage = (layer_index - 1) // 4
        index_block = (layer_index - 1) % 4
        if index_block == 0:
            # don't need to change downsample because the [0]conv1.in_channels and [1].conv2.out_channels stay the same
            conv_old = net.layer_stages[index_stage][0].conv1
            batchnorm_old = net.layer_stages[index_stage][0].bn1
            conv_next_old = net.layer_stages[index_stage][0].conv2

            conv_new = prune_convolution(conv_old, filter_index, is_reducing_channels_out=True)
            batchnorm_new = prune_batchnorm(batchnorm_old, filter_index)
            conv_next_new = prune_convolution(conv_next_old, filter_index, is_reducing_channels_out=False)

            bb_old = net.layer_stages[index_stage][0]
            bb_new = BasicBlockDummy(conv_new, batchnorm_new, bb_old.relu, conv_next_new, bb_old.bn2, bb_old.downsample,
                                     bb_old.stride)
            net.layer_stages[index_stage] = nn.Sequential(bb_new, net.layer_stages[index_stage][1])
        elif index_block == 1:
            conv_old = net.layer_stages[index_stage][0].conv2
            batchnorm_old = net.layer_stages[index_stage][0].bn2
            conv_next_old = net.layer_stages[index_stage][1].conv1
            downsample_1_old = net.layer_stages[index_stage][0].downsample

            conv_new = prune_convolution(conv_old, filter_index, is_reducing_channels_out=True)
            batchnorm_new = prune_batchnorm(batchnorm_old, filter_index)
            conv_next_new = prune_convolution(conv_next_old, filter_index, is_reducing_channels_out=False)

            if downsample_1_old is None:
                n_channels_out = batchnorm_new.num_features
                downsample_1_new = nn.Sequential(nn.Conv2d(net.layer_stages[index_stage][0].conv1.in_channels,
                                                           n_channels_out, kernel_size=1, stride=1, bias=False),
                                                 nn.BatchNorm2d(n_channels_out))
            else:
                conv_downsample_1_new = prune_convolution(downsample_1_old[0], filter_index,
                                                          is_reducing_channels_out=True)
                batchnorm_downsample_1_new = prune_batchnorm(downsample_1_old[1], filter_index)
                downsample_1_new = nn.Sequential(conv_downsample_1_new, batchnorm_downsample_1_new)

            bb_1_old = net.layer_stages[index_stage][0]
            bb_1_new = BasicBlockDummy(bb_1_old.conv1, bb_1_old.bn1, bb_1_old.relu, conv_new, batchnorm_new,
                                       downsample_1_new, bb_1_old.stride)

            downsample_2_old = net.layer_stages[index_stage][1].downsample
            if downsample_2_old is None:
                n_channels_out = net.layer_stages[index_stage][1].bn2.num_features
                downsample_2_new = nn.Sequential(nn.Conv2d(conv_next_new.in_channels, n_channels_out,
                                                           kernel_size=1, stride=1, bias=False),
                                                 nn.BatchNorm2d(n_channels_out))
            else:
                conv_downsample_2_new = prune_convolution(downsample_2_old[0], filter_index,
                                                          is_reducing_channels_out=False)
                downsample_2_new = nn.Sequential(conv_downsample_2_new, downsample_2_old[1])

            bb_2_old = net.layer_stages[index_stage][1]
            bb_2_new = BasicBlockDummy(conv_next_new, bb_2_old.bn1, bb_2_old.relu, bb_2_old.conv2, bb_2_old.bn2,
                                       downsample_2_new, bb_2_old.stride)
            net.layer_stages[index_stage] = nn.Sequential(bb_1_new, bb_2_new)

        elif index_block == 2:
            # don't need to change downsample because the [0]conv1.in_channels and [1].conv2.out_channels stay the same
            conv_old = net.layer_stages[index_stage][1].conv1
            batchnorm_old = net.layer_stages[index_stage][1].bn1
            conv_next_old = net.layer_stages[index_stage][1].conv2

            conv_new = prune_convolution(conv_old, filter_index, is_reducing_channels_out=True)
            batchnorm_new = prune_batchnorm(batchnorm_old, filter_index)
            conv_next_new = prune_convolution(conv_next_old, filter_index, is_reducing_channels_out=False)

            bb_old = net.layer_stages[index_stage][1]
            bb_new = BasicBlockDummy(conv_new, batchnorm_new, bb_old.relu, conv_next_new, bb_old.bn2, bb_old.downsample,
                                     bb_old.stride)
            net.layer_stages[index_stage] = nn.Sequential(net.layer_stages[index_stage][0], bb_new)
        else:
            conv_old = net.layer_stages[index_stage][1].conv2
            batchnorm_old = net.layer_stages[index_stage][1].bn2
            downsample_1_old = net.layer_stages[index_stage][1].downsample

            conv_new = prune_convolution(conv_old, filter_index, is_reducing_channels_out=True)
            batchnorm_new = prune_batchnorm(batchnorm_old, filter_index)

            if downsample_1_old is None:
                n_channels_out = conv_new.out_channels
                downsample_1_new = nn.Sequential(nn.Conv2d(net.layer_stages[index_stage][1].conv1.in_channels,
                                                           n_channels_out, kernel_size=1, stride=1, bias=False),
                                                 nn.BatchNorm2d(n_channels_out))
            else:
                conv_downsample_1_new = prune_convolution(downsample_1_old[0], filter_index,
                                                          is_reducing_channels_out=True)
                batchnorm_downsample_1_new = prune_batchnorm(downsample_1_old[1], filter_index)
                downsample_1_new = nn.Sequential(conv_downsample_1_new, batchnorm_downsample_1_new)

            bb_old = net.layer_stages[index_stage][1]
            bb_new = BasicBlockDummy(bb_old.conv1, bb_old.bn1, bb_old.relu, conv_new, batchnorm_new, downsample_1_new,
                                     bb_old.stride)
            net.layer_stages[index_stage] = nn.Sequential(net.layer_stages[index_stage][0], bb_new)

            net.side_prep[index_stage] = prune_convolution(net.side_prep[index_stage], filter_index,
                                                           is_reducing_channels_out=False)

            if index_stage <= 2:
                conv_next_old = net.layer_stages[index_stage + 1][0].conv1
                conv_next_new = prune_convolution(conv_next_old, filter_index, is_reducing_channels_out=False)
                downsample_2_old = net.layer_stages[index_stage + 1][0].downsample

                if downsample_2_old is None:
                    n_channels_out = net.layer_stages[index_stage + 1][0].bn2.num_features
                    downsample_2_new = nn.Sequential(nn.Conv2d(conv_next_new.in_channels, n_channels_out,
                                                               kernel_size=1, stride=1, bias=False),
                                                     nn.BatchNorm2d(n_channels_out))
                else:
                    conv_downsample_2_new = prune_convolution(downsample_2_old[0], filter_index,
                                                              is_reducing_channels_out=False)
                    downsample_2_new = nn.Sequential(conv_downsample_2_new, downsample_2_old[1])
                bb_old = net.layer_stages[index_stage + 1][0]
                bb_new = BasicBlockDummy(conv_next_new, bb_old.bn1, bb_old.relu, bb_old.conv2, bb_old.bn2,
                                         downsample_2_new, bb_old.stride)
                net.layer_stages[index_stage + 1] = nn.Sequential(bb_new, net.layer_stages[index_stage + 1][1])

    return net


def prune_convolution(conv, filter_index, is_reducing_channels_out: bool):
    reduction_channels_in = 0 if is_reducing_channels_out else 1
    reduction_channels_out = 1 if is_reducing_channels_out else 0
    new_conv = torch.nn.Conv2d(in_channels=conv.in_channels - reduction_channels_in,
                               out_channels=conv.out_channels - reduction_channels_out,
                               kernel_size=conv.kernel_size,
                               stride=conv.stride,
                               padding=conv.padding,
                               dilation=conv.dilation,
                               groups=conv.groups,
                               bias=False)
    old_weights = conv.weight.data.cpu().numpy()
    new_weights = new_conv.weight.data.cpu().numpy()

    if is_reducing_channels_out:
        new_weights[:filter_index, :, :, :] = old_weights[:filter_index, :, :, :]
        new_weights[filter_index:, :, :, :] = old_weights[filter_index + 1:, :, :, :]
    else:
        new_weights[:, :filter_index, :, :] = old_weights[:, :filter_index, :, :]
        new_weights[:, filter_index:, :, :] = old_weights[:, filter_index + 1:, :, :]

    new_conv.weight.data = torch.from_numpy(new_weights).cuda()
    # new_conv.weight.data = torch.from_numpy(new_weights)
    return new_conv


def prune_batchnorm(batchnorm_old, filter_index):
    new_batchnorm = nn.BatchNorm2d(num_features=batchnorm_old.num_features - 1,
                                   eps=batchnorm_old.eps,
                                   momentum=batchnorm_old.momentum,
                                   affine=batchnorm_old.affine)
    # net.layer_base[1].track_running_stats no attribute...
    old_weights = batchnorm_old.weight.data.cpu().numpy()
    new_weights = new_batchnorm.weight.data.cpu().numpy()
    new_weights[:filter_index] = old_weights[:filter_index]
    new_weights[filter_index:] = old_weights[filter_index + 1:]
    new_batchnorm.weight.data = torch.from_numpy(new_weights).cuda()
    return new_batchnorm


# net.layer_base[1]

def get_candidates_to_prune(pruner: FilterPruner, n_filters_to_prune: int, net: nn.Module,
                            data_loader: data.DataLoader):
    pruner.reset()
    train(pruner, data_loader, n_epochs=args.n_epochs_train)
    pruner.normalize_ranks_per_layer()
    return pruner.get_prunning_plan(n_filters_to_prune)


for _ in tqdm(range(n_iterations)):
    prune_targets = get_candidates_to_prune(pruner, n_filters_to_prune_per_iter, net, data_loader)
    layers_prunned = {}

    # net = net.cpu()
    layer_index_prev = -1
    for layer_index, filter_index in prune_targets:
        # if layer_index != layer_index_prev:
        #     print(layer_index_prev, net)
        #     layer_index_prev = layer_index
        net = prune_resnet18_conv_layer(net, layer_index, filter_index)

    # net = net.cuda()
    # print("Plan to prune...", net)

    fine_tune(net, data_loader, n_epochs=args.n_epochs_finetune)


# path_export = Path('../models/resnet18_11_11_blackswan_epoch-9999_min.pth')
# torch.save(net.state_dict(), str(path_export))


class DummyProvider:
    def __init__(self, net):
        self.network = net


net_provider = DummyProvider(net)

# first time to measure the speed
experiment_helper.test(net_provider, data_loader, p, is_visualizing_results=False, eval_speeds=True,
                       seq_name='blackswan')

# second time for image output
experiment_helper.test(net_provider, data_loader, p, is_visualizing_results=False, eval_speeds=False,
                       seq_name='blackswan')
