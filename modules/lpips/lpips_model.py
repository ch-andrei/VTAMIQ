"""
Andrei Chubarau:
Modified and simplified LPIPS, code adapted from https://github.com/richzhang/PerceptualSimilarity
Zhang et al. "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018
"""

import torch
import torch.nn as nn
import torchvision.models as tv
from torchvision.models.utils import load_state_dict_from_url


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class NetLinLayer(nn.Module):
    def __init__(self, ch_in, init_weights=True):
        super(NetLinLayer, self).__init__()

        self.dropout = nn.Dropout(0.25)
        self.conv = nn.Conv2d(ch_in, 1, kernel_size=1, stride=1, padding=0, bias=False)

        if init_weights:
            self.init_weights()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        return x

    def init_weights(self):
        # set all to 1
        self.conv.weight.data = torch.clamp(self.conv.weight.data, min=1., max=1.)

    def clamp_weights(self):
        # difference should always have a positive effect on error, hence we want to
        # clamp model weights to prevent negative values for the relevant modules
        self.conv.weight.data = torch.clamp(self.conv.weight.data, min=1e-3, )


class LPIPSm(nn.Module):
    """
    Reimplemented version of LPIPS (m stands for modified). Original architechture using Alexnet backbone.
    """

    def __init__(self, with_luminance=True):
        super(LPIPSm, self).__init__()

        self.net = AlexnetSliced()
        self.d1 = NetLinLayer(64)
        self.d2 = NetLinLayer(192)
        self.d3 = NetLinLayer(384)
        self.d4 = NetLinLayer(256)
        self.d5 = NetLinLayer(256)
        self.diff_net = [self.d1, self.d2, self.d3, self.d4, self.d5]

        self.clamp_weights()

    def forward(self, patches):
        patches_ref = patches[0]
        patches_dist = patches[1]

        B, N, C, P, P = patches_ref.shape

        patches_ref = patches_ref.view(B * N, C, P, P)
        patches_dist = patches_dist.view(B * N, C, P, P)

        x1, x2 = self.net(patches_ref), self.net(patches_dist)

        val = None
        for i in range(len(self.diff_net)):
            diff = (normalize_tensor(x1[i]) - normalize_tensor(x2[i]))
            diff = diff * diff
            diff = self.diff_net[i](diff)
            avg = spatial_average(diff, keepdim=True)
            val = avg if val is None else val + avg
            # print('val', val)

        val = val.view(B, N, -1)
        val = torch.mean(val, dim=1)

        # print(float(torch.mean(val)))

        val = torch.clamp_min(val, 0.)  # 0. equals perfect quality

        return val.flatten()

    def clamp_weights(self):
        for net in self.diff_net:
            net.clamp_weights()


class AlexnetSliced(nn.Module):
    def __init__(self, pretrained=True, requires_grad=True):
        super(AlexnetSliced, self).__init__()

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        alexnet_pretrained = tv.alexnet(pretrained=pretrained)
        alexnet_pretrained_features = alexnet_pretrained.features
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, x):
        x = self.slice1(x)
        h_relu1 = x
        x = self.slice2(x)
        h_relu2 = x
        x = self.slice3(x)
        h_relu3 = x
        x = self.slice4(x)
        h_relu4 = x
        x = self.slice5(x)
        h_relu5 = x
        return h_relu1, h_relu2, h_relu3, h_relu4, h_relu5
