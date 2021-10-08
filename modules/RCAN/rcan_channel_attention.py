"""
Andrei Chubarau:
RCAN code adapted from https://github.com/yulunzhang/RCAN
Yulun Zhang, Kunpeng Li, Kai Li, Lichen Wang, Bineng Zhong, and Yun Fu,
"Image Super-Resolution Using Very Deep Residual Channel Attention Networks", ECCV 2018, [arXiv]
"""

from torch import nn


# Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction, bias):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.PReLU(),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, n_feat, reduction=16, bias_ca=True, bn=False):

        super(RCAB, self).__init__()

        self.body = nn.Sequential(
            nn.BatchNorm2d(n_feat) if bn else nn.Sequential(),
            nn.PReLU(),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True),
            nn.BatchNorm2d(n_feat) if bn else nn.Sequential(),
            CALayer(n_feat, reduction, bias_ca)
            # nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True),  # replace CA with FC
        )

    def forward(self, x):
        return x + self.body(x)


class ResidualGroup(nn.Module):
    def __init__(self, n_feat, num_rcabs_per_group, reduction, bn=False):
        super(ResidualGroup, self).__init__()
        modules_body = [RCAB(n_feat, reduction, bn=bn) for _ in range(num_rcabs_per_group)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        return x + self.body(x)


class ResidualGroupXY(ResidualGroup):
    """
    Modified residual group to use a secondary signal for the attention stage
    """
    def forward(self, x, y):
        return x * self.body(y)
