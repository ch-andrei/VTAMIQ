import torch
import tabulate
import os

import torch.nn as nn

from utils.logging import log_warn


def tcpu(t):
    return t.cpu().item()


def tinfo(tag, t):
    # torch.tensors
    print('tinfo:', tag, t.shape, tcpu(t.min()), tcpu(t.mean()), tcpu(t.max()))


def ainfo(tag, a):
    # np.arrays
    print('ainfo:', tag, a.shape, a.min(), a.mean(), a.max())


def init_weights_linear(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.normal_(layer.bias, std=1e-6)


def set_grad(layer, requires_grad):
    for param in layer.parameters():
        param.requires_grad = requires_grad


def normalize_magnitude(x, eps=1e-9):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def normalize_tensor(x, eps=1e-9):
    x = x - x.min()
    return x / (x.max() + eps)


def save_model_params(model, output_dir):
    # save model layout
    num_params, parameters = get_num_parameter(model, trainable=False)
    path = os.path.join(output_dir, "model_params.txt")
    with open(path, "w") as f:
        f.write("Number of parameters: {} ({} for float32)\n".format(
            human_format(num_params),
            sizeof_fmt(num_params * 4)))
        f.write(tabulate.tabulate(parameters))


def print_parameters(model, full=True):
    # compute number of parameters
    num_params, parameters = get_num_parameter(model, trainable=False)
    num_bytes = num_params * 4  # assume float32 for all
    print(f"Number of parameters: {human_format(num_params)} ({sizeof_fmt(num_bytes)} for float32)")
    num_trainable_params, trainable_parameters = get_num_parameter(model, trainable=True)
    print("Number of trainable parameters:", human_format(num_trainable_params))
    if full:
        print("Model parameters (trainable marked with '(t)'):")
        # Print detailed parameters
        print(tabulate.tabulate(parameters))


def print_flops(model):
    from thop import profile
    batch_size = 1
    num_patches = 500
    shape = (batch_size, num_patches, 3, 16, 16)
    pos_shape = (batch_size, num_patches, 2)
    model = model.train()
    input_data = torch.rand(*shape).cuda()
    input_pos = torch.rand(*pos_shape).cuda()
    num_flops, num_params = profile(model, inputs=((input_data, input_data), input_pos))
    print("Number of FLOPS:", human_format(num_flops))


def load_model(model_state_dict, model, info=""):
    # print("load_model() got model_state_dict with weights for the following layers:")
    # print(model_state_dict.keys())
    try:
        # try strict load
        model.load_state_dict(model_state_dict)
    except RuntimeError as re:
        print(re)
        log_warn('Continuing with partial load... {}'.format(info))
        # try relaxed load
        model.load_state_dict(model_state_dict, strict=False)


def get_num_parameter(model, trainable=False):
    if trainable:
        params = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
    else:
        params = [(("(t) " + n) if p.requires_grad else n, p) for (n, p) in model.named_parameters()]

    total_params = sum(p.numel() for (n, p) in params)
    num_param_list = [(n, p.numel()) for (n, p) in params]

    return total_params, num_param_list


def human_format(num):
    num = float("{:.3g}".format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return "{}{}".format(
        "{:f}".format(num).rstrip("0").rstrip("."), ["", "K", "M", "B", "T"][magnitude]
    )


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)
