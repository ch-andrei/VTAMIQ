from functools import partial

import torch
from timm.layers import DropPath
from torch import nn


def lerp(start, end, weight):
    return start + weight * (end - start)


# adapted from https://github.com/yulunzhang/RCAN
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, num_rcabs, reduction=4, path_drop_prob=0.1, use_bn=False,
                 use_ms_cam=False, use_local=False, input1d=False):
        ConvNd = nn.Conv1d if input1d else nn.Conv2d
        super(ResidualGroup, self).__init__()
        self.body = nn.Sequential(
            *[
                RCAB(n_feat, reduction,
                     use_bn=use_bn, use_ca=True, use_ms_cam=use_ms_cam, use_local=use_local, input1d=input1d)
                for _ in range(num_rcabs)
            ],
            ConvNd(n_feat, n_feat, kernel_size=1),
        )
        self.drop = DropPath(path_drop_prob)

    def forward(self, x):
        return x + self.drop(self.body(x))


# Residual Channel Attention Block (RCAB)
# adapted from https://github.com/yulunzhang/RCAN
class RCAB(nn.Module):
    def __init__(self, n_feat, reduction, use_bn=True, use_ca=True, use_ms_cam=False, use_local=False, input1d=False):
        # NOTE: use_ms_cam=True and use_local=True may help when input is BxNxK (not BxNx1)
        super(RCAB, self).__init__()
        ConvNd = nn.Conv1d if input1d else nn.Conv2d
        BatchNorm = nn.BatchNorm1d if input1d else nn.BatchNorm2d
        ChannelAttn = partial(MS_CAM, use_local=use_local, use_bn=use_bn) if use_ms_cam else CALayer
        self.body = nn.Sequential(
            BatchNorm(n_feat) if use_bn else nn.Sequential(),
            nn.PReLU(),
            ConvNd(n_feat, n_feat, kernel_size=1),
            BatchNorm(n_feat) if use_bn else nn.Sequential(),
            ChannelAttn(n_feat, reduction, input1d=input1d) if use_ca else ConvNd(n_feat, n_feat, kernel_size=1),
        )

    def forward(self, x):
        return x + self.body(x)


def channel_attn_layers(channels, hidden, use_bn=True, input1d=True):
    ConvNd = nn.Conv1d if input1d else nn.Conv2d
    BatchNorm = nn.BatchNorm1d if input1d else nn.BatchNorm2d
    return [
        ConvNd(channels, hidden, kernel_size=1),
        BatchNorm(hidden) if use_bn else nn.Sequential(),
        nn.ReLU(inplace=True),
        ConvNd(hidden, channels, kernel_size=1),
        BatchNorm(channels) if use_bn else nn.Sequential(),
    ]


# Channel Attention (CA) Layer
# adapted from https://github.com/yulunzhang/RCAN
# Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu,
# "Image Super-Resolution Using Very Deep Residual Channel Attention Networks", ECCV 2018, [arXiv]
class CALayer(nn.Module):
    """
    CA with "global" attention only
    """
    def __init__(self, dim, reduction=8, input1d=False):
        super(CALayer, self).__init__()
        hidden = dim // reduction
        AdaptivePoolNd = nn.AdaptiveAvgPool1d if input1d else nn.AdaptiveAvgPool2d
        self.conv_du = nn.Sequential(
            AdaptivePoolNd(1),
            *channel_attn_layers(dim, hidden, use_bn=False, input1d=True)
        )

    def forward(self, x, return_weight=False):
        w = torch.sigmoid(self.conv_du(x))
        if return_weight:
            return w
        return x * w


def return_zero(*args, **kwargs):
    return 0.


# Adapted from "Attentional Feature Fusion" https://arxiv.org/abs/2009.14082
# https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
# similar to CALayer but also uses optional local attention
class AFF(nn.Module):
    def __init__(self, channels, reduction=2, use_bn=True, use_local=True, input1d=False):
        super(AFF, self).__init__()
        hidden = channels // reduction
        AdaptivePoolNd = nn.AdaptiveAvgPool1d if input1d else nn.AdaptiveAvgPool2d
        self.local_att = nn.Sequential(
            *channel_attn_layers(channels, hidden, use_bn=use_bn, input1d=input1d) if use_local else None
        )
        self.global_att = nn.Sequential(
            AdaptivePoolNd(1),
            *channel_attn_layers(channels, hidden, use_bn=use_bn, input1d=input1d)
        )

    def forward(self, x, y, return_weight=False):
        xy = x + y
        w = torch.sigmoid(self.local_att(xy) + self.global_att(xy))
        if return_weight:
            return w
        return lerp(x, y, w)


# Adapted from "Attentional Feature Fusion" https://arxiv.org/abs/2009.14082
# https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
# similar to CALayer but also uses optional local attention
# NOTE: MS_CAM is AFF but for a single input (y is not used)
class MS_CAM(AFF):
    def forward(self, x, y=0, return_weight=False):
        w = torch.sigmoid(self.local_att(x) + self.global_att(x))
        if return_weight:
            return w
        return w * x


# Adapted from "Attentional Feature Fusion" https://arxiv.org/abs/2009.14082
# https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
class iAFF(nn.Module):
    def __init__(self, channels, reduction=2, use_bn=True, use_local=True, input1d=False):
        super().__init__()
        self.aff = AFF(channels, reduction, use_bn=use_bn, use_local=use_local, input1d=input1d)
        self.ms_cam = MS_CAM(channels, reduction, use_bn=use_bn, use_local=use_local, input1d=input1d)

    def forward(self, x, y, return_weight=False):
        xi = self.aff(x, y)
        w = self.ms_cam(xi, return_weight=True)
        if return_weight:
            return w
        return lerp(x, y, w)
