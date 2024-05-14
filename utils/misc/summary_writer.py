import torch
from tensorboardX import SummaryWriter

from utils.misc import accumulators


class SplitSummaryWriter(SummaryWriter):
    """
    Divides logs into sections based on Split names
    """
    def __init__(self, log_every_n_steps=1, **kwargs):
        self.log_every_n_steps = log_every_n_steps
        super(SplitSummaryWriter, self).__init__(**kwargs)
        self.tags = {}
        self.accumuls = {}

    def add_scalar(self, split_name, tag, scalar_value, step, walltime=None, force_add=False):
        if split_name not in self.tags:
            self.tags[split_name] = {}
            self.accumuls[split_name] = {}

        accumuls = self.accumuls[split_name]
        tags = self.tags[split_name]

        if tag not in tags:
            count = len(tags) + 1
            tags[tag] = "{}/{}.{}".format(split_name, count, tag)
            accumuls[tag] = accumulators.Mean()

        accumul = accumuls[tag]
        tag = tags[tag]

        # add new value to mean accumulator
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.detach().cpu().item()
        accumul.add(scalar_value)

        # log mean value if needed and reset accumulator
        if force_add or (step+1) % self.log_every_n_steps == 0:
            super(SplitSummaryWriter, self).add_scalar(tag, accumul.value(), global_step=step, walltime=walltime)
            accumul.reset()
