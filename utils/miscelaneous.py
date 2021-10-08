import numpy as np
import scipy.stats as st


def float2str3(value):
    return float2str(value, decimals=3)


def float2str(value, decimals=6):
    if value is not float:
        value = float(value)
    format_str = "{:." + str(decimals)
    if abs(value) < 10 ** -decimals and value != 0:
        format_str += "E"
    else:
        format_str += "f"
    format_str += "}"
    return format_str.format(value)


def lerp(a, b, ratio=0.5):
    ratio = min(1, max(0, ratio))
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


def split(list, num_splits, append_leftover_to_last=False):
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


def split_filename_and_extension(file_name):
    name_split_by_dot = file_name.split('.')
    img_filename_no_ext = '.'.join(name_split_by_dot[:-1])
    img_extension = name_split_by_dot[-1]
    return img_filename_no_ext, img_extension
