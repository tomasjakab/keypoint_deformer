import os
from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from . import logger


def load_network(net, path):
    """
    load network parameters whose name exists in the pth file.
    return:
        INT trained step

    From https://github.com/yifita/deep_cage
    """
    # warnings.DeprecationWarning("load_network is deprecated. Use module.load_state_dict(strict=False) instead.")
    if isinstance(path, str):
        logger.info("loading network from {}".format(path))
        if path[-3:] == "pth":
            loaded_state = torch.load(path)
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
        else:
            loaded_state = np.load(path).item()
            if "states" in loaded_state:
                loaded_state = loaded_state["states"]
    elif isinstance(path, dict):
        loaded_state = path

    network = net.module if isinstance(
        net, torch.nn.DataParallel) else net

    missingkeys, unexpectedkeys = network.load_state_dict(loaded_state, strict=False)
    if len(missingkeys)>0:
        logger.warn("load_network {} missing keys".format(len(missingkeys)), "\n".join(missingkeys))
    if len(unexpectedkeys)>0:
        logger.warn("load_network {} unexpected keys".format(len(unexpectedkeys)), "\n".join(unexpectedkeys))


def save_network(net, directory, network_label, epoch_label=None, **kwargs):
    """
    save model to directory with name {network_label}_{epoch_label}.pth
    Args:
        net: pytorch model
        directory: output directory
        network_label: str
        epoch_label: convertible to str
        kwargs: additional value to be included
    
    From https://github.com/yifita/deep_cage
    """
    save_filename = "_".join((network_label, str(epoch_label))) + ".pth"
    save_path = os.path.join(directory, save_filename)
    merge_states = OrderedDict()
    merge_states["states"] = net.cpu().state_dict()
    for k in kwargs:
        merge_states[k] = kwargs[k]
    torch.save(merge_states, save_path)
    net = net.cuda()


def weights_init(m):
    """
    initialize the weighs of the network for Convolutional layers and batchnorm layers

    From https://github.com/yifita/deep_cage
    """
    if isinstance(m, (torch.nn.modules.conv._ConvNd, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
        torch.nn.init.constant_(m.bias, 0.0)
        torch.nn.init.constant_(m.weight, 1.0)


class Conv1d(nn.Module):
    """
    1dconvolution with custom normalization and activation
    
    From https://github.com/yifita/deep_cage
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 activation=None, normalization=None, momentum=0.01, conv_params={}):
        super(Conv1d, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias, **conv_params)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            elif self.activation == "tanh":
                self.act = nn.Tanh()
            else:
                raise ValueError("only \"relu/elu/lrelu/tanh\" implemented")

    def forward(self, x, epoch=None):
        x = self.conv(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x

        

class Linear(nn.Module):
    """
    1dconvolution with custom normalization and activation
    
    From https://github.com/yifita/deep_cage
    """

    def __init__(self, in_channels, out_channels, bias=True,
                 activation=None, normalization=None, momentum=0.01):
        super(Linear, self).__init__()
        self.activation = activation
        self.normalization = normalization
        bias = not normalization and bias
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

        if normalization is not None:
            if self.normalization == 'batch':
                self.norm = nn.BatchNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            elif self.normalization == 'instance':
                self.norm = nn.InstanceNorm1d(
                    out_channels, affine=True, eps=0.001, momentum=momentum)
            else:
                raise ValueError(
                    "only \"batch/instance\" normalization permitted.")

        # activation
        if activation is not None:
            if self.activation == 'relu':
                self.act = nn.ReLU()
            elif self.activation == 'elu':
                self.act = nn.ELU(alpha=1.0)
            elif self.activation == 'lrelu':
                self.act = nn.LeakyReLU(0.1)
            elif self.activation == "tanh":
                self.act = nn.Tanh()
            else:
                raise ValueError("only \"relu/elu/lrelu/tanh\" implemented")

    def forward(self, x, epoch=None):
        x = self.linear(x)

        if self.normalization is not None:
            x = self.norm(x)

        if self.activation is not None:
            x = self.act(x)

        return x
