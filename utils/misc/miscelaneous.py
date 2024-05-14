import numpy as np
import scipy.stats as st
import os

from utils.logging import log_warn


def check_unused_kwargs(log_tag="", **kwargs):
    for kwarg in kwargs:
        log_warn(f"Unused kwarg [{kwarg}={kwargs[kwarg]}]", log_tag)


def float2str3(value):
    return float2str(value, decimals=3)


def float2str(value, decimals=6):
    if value is not float:
        value = float(value)
    format_str = "{:." + str(decimals)
    if decimals == 0:
        return f"{int(value)}"
    elif abs(value) < 10 ** -decimals and value != 0:
        format_str += "E"
    else:
        format_str += "f"
    format_str += "}"
    return format_str.format(value)


def lerp(a, b, ratio=0.5):
    ratio = min(1., max(0., ratio))
    return a + (b - a) * ratio


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    # print('gkern offset', offset, -nsig*(1-offset), nsig*(1+offset))
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    return kern1d/kern1d.sum()


# lerps over a list of values and a sliding gaussian weight
def lerp_list_gau(l, ratio=0.5, nsig=10):
    lnp = np.array(l)
    n = len(l)
    i = int((1-ratio)*(n-1))
    weights = gkern(kernlen=2*n-1, nsig=nsig)[i:i+n]
    if len(lnp.shape) > 1:
        weights = weights.reshape(-1, 1)
    weighted = lnp * weights / weights.sum()
    return weighted.sum(axis=0) if len(lnp.shape) > 1 else weighted.sum()


def split_list(list, num_splits, append_leftover_to_last=False):
    n = len(list)
    list_per_split = [[] for _ in range(num_splits)]
    num_per_split = int(len(list) / num_splits)
    leftover = n - num_per_split * num_splits
    for i in range(num_splits):
        starting_index = i * num_per_split
        for item in list[starting_index: starting_index + num_per_split]:
            list_per_split[i].append(item)
    starting_index = num_splits * num_per_split
    for i, item in enumerate(list[starting_index: starting_index + leftover]):
        if append_leftover_to_last:
            index = num_splits - 1
        else:
            index = i % num_splits
        list_per_split[index].append(item)
    return list_per_split


def recursive_dict_flatten(d_in, d_out=None, key_chain=None):
    if d_out is None:
        d_out = dict()
    if key_chain is None:
        key_chain = list()
    for key in d_in:
        if isinstance(d_in[key], dict):
            recursive_dict_flatten(d_in[key], d_out, key_chain + [key, ])
        else:
            full_key = '.'.join(key_chain) + "." + key
            d_out[full_key] = d_in[key]
    return d_out


def split_filename_and_extension(file_name):
    name_split_by_dot = file_name.split('.')
    img_filename_no_ext = '.'.join(name_split_by_dot[:-1])
    img_extension = name_split_by_dot[-1]
    return img_filename_no_ext, img_extension


def verify_image_extension(path, extensions=None):
    if extensions is None:
        extensions = ["jpg", "png", "bmp"]
    path, _ = split_filename_and_extension(path)
    for extension in extensions:
        path_ext = f"{path}.{extension}"
        if os.path.exists(path_ext):
            return path_ext
    raise FileNotFoundError(f"No such file or directory: {path}")


class MaintainRandomSeedConsistency(object):
    # on enter, set a given random state; on exit, set previous random state
    def __init__(self, temporary_seed=None):
        self.seed_temp = temporary_seed  # if None, will not modify random states
        self.seed_prev = None

    def __enter__(self):
        if self.seed_temp is not None:
            self.seed_prev = np.random.get_state()
            np.random.seed(self.seed_temp)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.seed_temp is not None:
            np.random.set_state(self.seed_prev)
