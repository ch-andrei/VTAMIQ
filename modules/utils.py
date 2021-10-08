import torch


def cpu_float(tensor):
    return float(tensor.cpu())


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat ** 2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


def load_model(model_state_dict, model):
    try:
        model.load_state_dict(model_state_dict)
    except RuntimeError as re:
        print(re)
        print('WARNING: Continuing with partial load...')
        model.load_state_dict(model_state_dict, strict=False)


def get_num_parameter(model, trainable=False):
    if trainable:
        params = [(n, p) for (n, p) in model.named_parameters() if p.requires_grad]
    else:
        params = [(n, p) for (n, p) in model.named_parameters()]

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
