import torch
from torch import nn


class PreferenceModule(nn.Module):
    """
        simple module used to remap from quality prediction difference (delta Q) to preference judgement.
    """
    def __init__(self, weight=1.):
        super(PreferenceModule, self).__init__()
        self.p = nn.Parameter(torch.Tensor(weight))

    def forward(self, q1, q2):
        return torch.sigmoid(self.p * (q2 - q1)).flatten()

