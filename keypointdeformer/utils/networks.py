# From https://github.com/yifita/deep_cage

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from .nn import Conv1d, Linear


class PointNetfeat(nn.Module):
    """
    From https://github.com/yifita/deep_cage
    """
    def __init__(self, dim=3, num_points=2500, global_feat=True, trans=False, bottleneck_size=512, activation="relu", normalization=None):
        super().__init__()
        self.conv1 = Conv1d(dim, 64, 1, activation=activation, normalization=normalization)
        # self.stn_embedding = STN(num_points = num_points, K=64)
        self.conv2 = Conv1d(64, 128, 1, activation=activation, normalization=normalization)
        self.conv3 = Conv1d(128, bottleneck_size, 1, activation=None, normalization=normalization)
        #self.mp1 = torch.nn.MaxPool1d(num_points)

        self.trans = trans
        #self.mp1 = torch.nn.MaxPool1d(num_points)
        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        if self.trans:
            trans = self.stn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans)
            x = x.transpose(2,1)
        x = self.conv1(x)
        pointfeat = x
        x = self.conv2(x)
        x = self.conv3(x)
        x,_ = torch.max(x, dim=2)
        if self.trans:
            if self.global_feat:
                return x, trans
            else:
                x = x.view(batchsize, -1, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans
        else:
            return x



class MLPDeformer2(nn.Module):
    """
    From https://github.com/yifita/deep_cage
    """
    def __init__(self, dim, bottleneck_size, npoint, residual=True, normalization=None):
        super().__init__()
        self.npoint = npoint
        self.dim = dim
        self.residual = residual
        self.layers = nn.Sequential(
                Linear(bottleneck_size, 512, activation="lrelu", normalization=normalization),
                Linear(512, 256, activation="lrelu", normalization=normalization),
                Linear(256, npoint*dim)
            )
    def forward(self, code):
        B, _ = code.shape
        x = self.layers(code)
        x = x.reshape(B, self.dim, self.npoint)
        return x
