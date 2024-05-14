"""
Andrei Chubarau:
Work flow inspired by https://github.com/epfml/attention-cnn
"""

import time

import torch
from tqdm.auto import tqdm

import torch.nn.functional as functional

import utils.misc.accumulators as accumulators

from utils.misc.correlations import compute_correlations as _compute_correlations
from utils.logging import log_warn, FileLogger, Logger, log
from utils.misc.miscelaneous import float2str
from utils.misc.summary_writer import SplitSummaryWriter

from modules.utils import *
from train_config import *

import cv2


def get_optimizer_scheduler(models, train_loader):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """

    validate_configs_check()

    parameters = []
    for model in models:
        if model is not None:
            for parameter in model.parameters():
                parameters.append(parameter)

    assert 0 < len(parameters), "Optimizer must have parameters to optimize."  # check that not everything is frozen

    lr = global_config["optimizer_learning_rate"]

    if global_config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            parameters,
            lr=lr,
            momentum=global_config["optimizer_sgd_momentum"],
            weight_decay=global_config["optimizer_weight_decay"],
            nesterov=global_config["optimizer_sgd_nesterov"],
        )

    elif global_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=lr,
            weight_decay=global_config["optimizer_weight_decay"]
        )

    elif global_config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=global_config["optimizer_weight_decay"]
        )

    else:
        raise ValueError("Unexpected value for optimizer")

    scheduler_verbose = not global_config["scheduler_step_per_batch"]

    if global_config["scheduler_step_per_batch"]:
        log_warn("scheduler_step_per_batch=True; LR will be updated after every batch.")

    if global_config["scheduler_type"] == "lambda":
        num_steps = global_config["num_epochs"]  # update LR after each epoch
        if global_config["scheduler_step_per_batch"]:
            num_steps *= len(train_loader)  # update LR every batch
        lambda_goal = global_config["optimizer_learning_rate_decay_lambda_goal"]
        lambda_ratio = lambda_goal ** (1.0 / num_steps)
        log(f"Using LambdaLR scheduler with lambda_ratio={float2str(lambda_ratio, 6)} and num_steps={num_steps}. "
              f"Initial LR={lr:.3e}; final LR={lr * lambda_ratio ** num_steps:.3e} "
              f"(using lambda_goal={lambda_goal} of initial LR).")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step_num: lambda_ratio ** step_num,
            verbose=scheduler_verbose,
        )

        # log("Adam/AdamW/AdaBound optimizers ignore all learning rate schedules.")
    elif global_config["scheduler_type"] == "cosine":
        num_steps = global_config["num_epochs"]
        if global_config["scheduler_step_per_batch"]:
            num_steps *= len(train_loader)
        log(f'Using CosineAnnealingLR scheduler with num_steps={num_steps}')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_steps,
            eta_min=global_config["optimizer_learning_rate_decay_cosine"] * lr,
            verbose=scheduler_verbose,
        )

    elif global_config["scheduler_type"] == "multistep":
        log(f'Using MultiStepLR scheduler.')
        if global_config["scheduler_step_per_batch"]:
            # TODO
            raise NotImplementedError("not implement step per batch")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=global_config["optimizer_decay_after_n_epochs"],
            gamma=global_config["optimizer_learning_rate_decay_multistep"],
            verbose=scheduler_verbose,
        )

    else:
        raise ValueError("Unexpected value for scheduler")

    return optimizer, scheduler


def get_checkpoint(filename, device):
    if filename is None:
        return None

    """Load model from a checkpoint"""
    log("Loading checkpoint file '{}'".format(filename))
    with open(filename, "rb") as f:
        checkpoint = torch.load(f, map_location=device)

    return checkpoint


def get_device():
    return torch.device("cuda" if not global_config["no_cuda"] and torch.cuda.is_available() else "cpu")


def get_model(device, checkpoint_file=None, force_load_pretrained=False, **model_kwargs):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """

    validate_configs_check()

    model_name = global_config["model"]
    has_checkpoint = checkpoint_file is not None

    log(f"Initializing Model: {model_name}.")

    model_type, model_config = get_model_type_and_config(model_name)
    model = model_type(
        **model_config,
        **model_kwargs
    )

    if has_checkpoint:
        # read checkpoint file and preprocess model state dict

        log(f"Model {global_config['model']} loading pretrained weights...")
        model_state_dict = get_checkpoint(checkpoint_file, device)[MODEL_STATE_DICT]

        if "VTAMIQ" in model_name:
            def pop_layers_from_model_state_dict(layer_prefix):
                for layer_name in list(model_state_dict.keys()):
                    if layer_prefix in layer_name:
                        model_state_dict.pop(layer_name)

            if not force_load_pretrained and not pretraining_config["allow_pretrained_weights_vit"]:
                log("Will not load transformer weights from checkpoint file.")
                pop_layers_from_model_state_dict("transformer.")

            if not force_load_pretrained and not pretraining_config["allow_pretrained_weights_diffnet"]:
                log("Will not load diffnet weights from checkpoint file.")
                pop_layers_from_model_state_dict("calibration_diff.")
                pop_layers_from_model_state_dict("calibration_feat.")
                pop_layers_from_model_state_dict("q_predictor.")

        load_model(model_state_dict, model, "model")
    else:
        log_warn(f"Model {global_config['model']} used without pre-trained model weights.")

    model.to(device, dtype=torch.float32)

    if device == torch.device("cuda"):
        log(f"Model {global_config['model']} using GPU.")
    else:
        log(f"Model {global_config['model']} using CPU.")

    return model


def get_pref_module(use_pref_module, device, checkpoint_file=None):
    validate_configs_check()

    pref_module = None
    if use_pref_module:
        log("Using preference module.")

        from modules.vtamiq.common import PreferenceModule
        pref_module = PreferenceModule(**pref_module_config)

        if checkpoint_file is not None:
            try:
                pref_module_state_dict = get_checkpoint(checkpoint_file, device)[PREF_MODULE_STATE_DICT]
                load_model(pref_module_state_dict, pref_module, "pref_module")
            except (KeyError, AttributeError):
                log("pref_module parameters missing from the provided checkpoint file...")

        pref_module.to(device, dtype=torch.float32)

    return pref_module


def get_models_dict(model, pref_module=None):
    models = {MODEL_STATE_DICT: model}
    if pref_module is not None:
        models[PREF_MODULE_STATE_DICT] = pref_module
    return models


def save_checkpoint(output_dir, filename, models, optimizer, scaler, epoch, spearman):
    """Store a checkpoint file to the output directory"""
    path = os.path.join(output_dir, filename)

    # Ensure the output directory exists
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    _models = dict()
    for model_name in models:
        model = models[model_name]
        if model is None:
            continue
        _models[model_name] = OrderedDict([
            (key, value) for key, value in model.state_dict().items()
        ])

    model_state_dict = {
        "epoch": epoch,
        "SROCC": spearman,
        **_models
    }

    if global_config["save_optimizer"]:
        model_state_dict["optimizer"] = optimizer
        model_state_dict["scaler"] = scaler

    time.sleep(1)  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save(model_state_dict, path)


def get_data_tuple(batch, device) -> object:
    return tuple(data.to(device, dtype=torch.float32, non_blocking=True) for data in batch)


def split_per_image(x, has_batch_dim=True, clone=True):
    # Note 1: clone may be needed to ensure .view() compatibility issues
    # Note 2: batch size B is optional; for K input images, x may be
    # x.shape = (B, K, N, C, P, P) -> len = 6
    # x.shape = (K, N, C, P, P) -> len = 5
    x_shape = x.shape
    num_images = x_shape[1] if has_batch_dim else x_shape[0]
    x_i = lambda i: x[:, i] if has_batch_dim else x[i].unsqueeze(0)
    x_per_image = tuple((x_i(i).clone() if clone else x_i(i)) for i in range(num_images))
    return x_per_image


def model_forward(model, model_name, patches, pos, scales):
    # tinfo('patches', patches[0])
    if "vtamiq" in model_name.lower():
        return model(patches, pos, scales)
    else:
        raise ValueError(f"Unsupported model {model_name}")


def predict(model, pref_module, data, is_pairwise, output_feats, use_scales):
    q, patches, pos, scales = data[:4]

    if is_pairwise:
        pref, pdist1, pdist2 = split_per_image(patches)
        posref, posdist1, posdist2 = split_per_image(pos)
        scalesref, scalesdist1, scalesdist2 = split_per_image(scales) if use_scales else (None, None, None)

        out1 = model((pref, pdist1), (posref, posdist1), (scalesref, scalesdist1))
        out2 = model((pref, pdist2), (posref, posdist2), (scalesref, scalesdist2))

        q1 = out1[0]
        q2 = out2[0]

        if output_feats:
            feats = (out1[1], out2[1])
        else:
            feats = None

        if pref_module is not None:
            q_p = pref_module(q1, q2)

        else:
            q_p = torch.sigmoid(q1 - q2)  # preference

    else:
        patches = split_per_image(patches)
        pos = split_per_image(pos)
        scales = split_per_image(scales) if use_scales else (None, None)

        out = model(patches, pos, scales)

        q_p, feats = out if output_feats else (out[0], None)

    q_p = q_p.flatten()

    return q, q_p, feats


def optimizer_step(loss, optimizer, scaler, model):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)  # need to call this before clip grad norm
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()


def spearman_loss(x, y):
    """
    measures Spearman’s correlation coefficient between target logits and output logits:
    att: [n, m]
    grad_att: [n, m]
    """

    def _rank_correlation_(att_map, att_gd):
        n = torch.tensor(att_map.shape[1])
        upper = 6 * torch.sum((att_gd - att_map).pow(2), dim=1)
        down = n * (n.pow(2) - 1.0)
        return (1.0 - (upper / down)).mean(dim=-1)

    x = x.sort(dim=1)[1]
    y = y.sort(dim=1)[1]
    correlation = _rank_correlation_(x.float(), y.float())
    return correlation


def pears_loss(x, y, eps=1e-6):
    xm = x - x.mean()
    ym = y - y.mean()

    normxm = torch.linalg.norm(xm) + eps
    normym = torch.linalg.norm(ym) + eps

    r = torch.dot(xm / normxm, ym / normym)
    r = torch.clamp(r, 0, 1)

    return 1 - r


def rank_loss(d, y, num_images, eps=1e-6, norm_num=True):
    loss = torch.zeros(1, device=d.device, dtype=d.dtype)
    # loss = 0

    if num_images < 2:
        return loss

    dp = torch.abs(d)

    combinations = torch.combinations(torch.arange(num_images), 2)
    combinations_count = max(1, len(combinations))

    for i, j in combinations:
        rl = torch.clamp_min(-(y[i] - y[j]) * (d[i] - d[j]) / (torch.abs(y[i] - y[j]) + eps), min=0)
        loss += rl / max(dp[i], dp[j])  # normalize by maximum absolute value

    if norm_num:
        loss = loss / combinations_count  # mean

    return loss


def mae_loss(d, y):
    return functional.l1_loss(d, y)


def mse_loss(d, y):
    return functional.mse_loss(d, y)


def loss_func_iqa(d, y, batch_size, device, w_mae_loss, w_rank_loss, w_pears_loss):
    # w_sum = w_mae_loss + w_rank_loss
    mae_value = mae_loss(d, y)
    rank_loss_value = rank_loss(d, y, batch_size)  # return loss_rank.detach().item(),
    pears_loss_value = pears_loss(d, y)
    return rank_loss_value + pears_loss_value, \
        mae_value.detach().item(), \
        rank_loss_value.detach().item(), \
        pears_loss_value.detach().item()


def average_over_repeats(x, num_repeats):
    # reshape from (N*num_repeats,) to (num_repeats, N), then average over num_repeats
    return np.mean(x.reshape(num_repeats, -1), axis=0)


def compute_correlations_cat_flat(ys, yp, num_repeats=1):
    ys = np.array(torch.cat(ys, dim=0).flatten(), dtype=float)
    yp = np.array(torch.cat(yp, dim=0).flatten(), dtype=float)
    if 1 < num_repeats:
        ys = average_over_repeats(ys, num_repeats)
        yp = average_over_repeats(yp, num_repeats)
    return _compute_correlations(ys, yp)


def get_tag(tag):
    return "" if (tag is None or tag == "") else f"{tag}-"


def writer_log_losses(
        writer, split_name, loss, loss_mae, loss_rank, loss_pears, step, tag=None, force_add=False, loss_di=None
):
    tag = get_tag(tag)
    writer.add_scalar(split_name, tag + "loss", loss, step, force_add=force_add)
    writer.add_scalar(split_name, tag + "mae_loss", loss_mae, step, force_add=force_add)
    writer.add_scalar(split_name, tag + "rank_loss", loss_rank, step, force_add=force_add)
    writer.add_scalar(split_name, tag + "pears_loss", loss_pears, step, force_add=force_add)
    if loss_di is not None:
        writer.add_scalar(split_name, tag + "DI_loss", loss_di, step, force_add=force_add)


def writer_log_losses_pairwise(writer, split_name, loss, step, tag=None, force_add=False):
    tag = get_tag(tag)
    writer.add_scalar(split_name, tag + "mae_loss", loss, step, force_add=force_add)


def writer_log_correlations(writer, split_name, correlations, step, tag=None, force_add=False):
    tag = get_tag(tag)
    writer.add_scalar(split_name, tag + SROCC_FIELD, correlations[SROCC_FIELD], step, force_add=force_add)
    writer.add_scalar(split_name, tag + KROCC_FIELD, correlations[KROCC_FIELD], step, force_add=force_add)
    writer.add_scalar(split_name, tag + PLCC_FIELD, correlations[SROCC_FIELD], step, force_add=force_add)
    writer.add_scalar(split_name, tag + RMSE_FIELD, correlations[RMSE_FIELD], step, force_add=force_add)


def log_loader_indices(logger, loader, split_name_actual):
    # to avoid logging too much text, check size of split
    dataset = loader.dataset
    split_name_loader = loader.split_name
    if len(dataset.splits_dict_ref[split_name_loader].indices) < 20000:
        logger(f"Dataset for {split_name_actual}: "
               f"dataset {dataset.name} ref images:",
               dataset.splits_dict_ref[split_name_loader])


def get_feats_for_coral(feats, pairwise=False, use_feats_separate=False, use_feats_dist=True, use_feats_diff=False):
    # non-pairwise predict() returns feats=(feats_ref, feats_dist, feats_diff): 3-element tuple
    # each of these is a vector of size {batch_size x token_num x transformer_dim}
    # when pairwise data is used, predict() returns (feats1, feats2), each being a 3-element tuple.
    # select only feats_diff for CORAL loss

    if pairwise:
        # in pairwise mode, feats is a tuple with 2 feature sets
        feats_ref1, feats_dist1, feats_diff1 = feats[0]
        feats_ref2, feats_dist2, feats_diff2 = feats[1]
        # simply concatenate features
        feats_ref = torch.cat((feats_ref1, feats_ref2), dim=0)
        feats_dist = torch.cat((feats_dist1, feats_dist1), dim=0)
        feats_diff = torch.cat((feats_diff1, feats_diff1), dim=0)

    else:
        feats_ref, feats_dist, feats_diff = feats

    out = (feats_ref,)

    if use_feats_dist:
        out += (feats_dist,)

    if use_feats_diff:
        out += (feats_diff,)

    if use_feats_separate:
        return out
    else:
        return (torch.cat(out, dim=0), )


def do_training(
        model, scaler, optimizer, scheduler, device, loader, pref_module,
        w_mae_loss, w_rank_loss, w_pears_loss,
        is_pairwise, is_debug, output_dir, logger, writer, checkpoint_every_n_batches,
        epoch, step,
):
    model.train()
    if pref_module is not None:
        pref_module.train()

    use_scales = training_run_uses_scales()

    train_iter = iter(loader)
    train_total_steps = len(loader)

    q_vals = []
    qp_vals = []
    # loop over training data
    for batch_i in tqdm(range(train_total_steps)):

        try:
            data = train_iter.__next__()
        except cv2.error as e:
            # NOTE: this is a workaround for errors during loading of images
            log_warn("Encountered CV2 error while loading batch:")
            log(e)
            log("Skipping current batch...")
            continue

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(dtype=torch.float16):
            data = get_data_tuple(data, device)

            q, q_p, feats = predict(
                model, pref_module, data, is_pairwise, output_feats=False, use_scales=use_scales
            )

            batch_size = q.shape[0]

            if batch_size < 2:
                log_warn("Batch size < 2; skipping current batch.")

            if is_pairwise:
                # only optimize MAE when training with pairwise preference data
                loss = mae_loss(q_p, q)

            else:
                loss, loss_mae, loss_rank, loss_pears = loss_func_iqa(
                    q_p, q, batch_size, device, w_mae_loss, w_rank_loss, w_pears_loss
                )

        optimizer_step(loss, optimizer, scaler, model)

        q_vals.append(q.detach().cpu())
        qp_vals.append(q_p.detach().cpu())

        if not is_debug:
            if is_pairwise:
                writer_log_losses_pairwise(writer, SPLIT_NAME_TRAIN, loss, step)
            else:
                writer_log_losses(writer, SPLIT_NAME_TRAIN, loss, loss_mae, loss_rank, loss_pears, step)

            if 4 < batch_size:  # correlations cant be computed for small batch sizes
                # correlations for current batch
                cors_s_batch = compute_correlations_cat_flat([q_vals[-1]], [qp_vals[-1]])

                writer.add_scalar(SPLIT_NAME_TRAIN, "SROCC_batch_s", cors_s_batch[SROCC_FIELD], step)

        if not is_debug and (batch_i + 1) % checkpoint_every_n_batches == 0:  # +1 to skip early save
            logger(
                f"Saving latest model during training: epoch=[{epoch}], split=[{SPLIT_NAME_TRAIN}], batch_i=[{batch_i}]")
            model_path = "latest.pth".format(epoch, batch_i)
            save_checkpoint(output_dir, model_path, get_models_dict(model, pref_module), optimizer, scaler, epoch, -1)

        if global_config["scheduler_step_per_batch"]:
            if not is_debug:
                writer.add_scalar(
                    SPLIT_NAME_TRAIN, "LR", scheduler.get_last_lr()[0], train_total_steps * (epoch - 1) + batch_i
                )
            scheduler.step()

        step += 1

    correlations = compute_correlations_cat_flat(q_vals, qp_vals)

    # end of epoch logging
    if not is_debug:
        if not global_config["scheduler_step_per_batch"]:
            writer.add_scalar(SPLIT_NAME_TRAIN, "LR", scheduler.get_last_lr()[0], epoch, force_add=True)
        writer_log_correlations(writer, SPLIT_NAME_TRAIN, correlations, epoch, force_add=True)

    if global_config["scheduler_step_per_batch"]:
        log(f"Current scheduler LR=[{float2str3(scheduler.get_last_lr()[0])}]")
    else:
        scheduler.step()

    return step, correlations


def do_validation(model, pref_module, device,
                  is_pairwise, is_debug, w_mae_loss, w_rank_loss, w_pears_loss,
                  split_name, loader, step, epoch, writer,
                  num_repeats=1, log_writer=True, output_logger=None, tag=""):

    use_scales = training_run_uses_scales()

    y = []
    yp = []
    with torch.no_grad():
        model.eval()
        if pref_module is not None:
            pref_module.eval()

        for _ in (tqdm(range(num_repeats), desc="num_repeats") if 1 < num_repeats else range(num_repeats)):

            # loop over the data
            for i, data in enumerate(tqdm(loader, desc="dataset")):

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    data = get_data_tuple(data, device)

                    q, q_p, _ = predict(
                        model, pref_module, data, is_pairwise, output_feats=False, use_scales=use_scales
                    )

                    if is_pairwise:
                        loss = mae_loss(q_p, q)
                    else:
                        batch_size = q.shape[0]
                        loss, loss_mae, loss_rank, loss_pears = loss_func_iqa(
                            q_p, q, batch_size, device, w_mae_loss, w_rank_loss, w_pears_loss
                        )

                y.append(q.cpu())
                yp.append(q_p.cpu())

                if log_writer and not is_debug:
                    if is_pairwise:
                        writer_log_losses_pairwise(writer, split_name, loss, step, tag, force_add=True)
                    else:
                        writer_log_losses(writer, split_name, loss, loss_mae, loss_rank, loss_pears, step, tag,
                                          force_add=True)

                if output_logger is not None:
                    values = list(np.array(q_p.cpu()))
                    values_s = []
                    for value in values:
                        values_s.append(str(value))
                    output_logger(i, tag, ",".join(values_s))

                step += 1

    if 0 < len(y):
        correlations = compute_correlations_cat_flat(y, yp, num_repeats)
    else:
        correlations = None

    if log_writer and not is_debug:
        writer_log_correlations(writer, split_name, correlations, epoch, force_add=True, tag=tag)

    return step, correlations


def train_domain_adaptation():
    global_config["is_domain_adaptation"] = True  # must be set prior to validate_configs()
    return train()


def train():
    # must call validate configs before running training
    validate_configs()

    is_debug = global_config["is_debug"]
    is_verbose = global_config["is_verbose"]

    is_vtamiq = global_config["model"] in models_vtamiq

    is_pairwise_t = dataset_is_pairwise(dataset_target())
    do_train = global_config["do_train"]
    do_val = global_config["do_val"]
    do_test = global_config["do_test"]
    is_test_only = not do_train and not do_val and do_test

    log(f"Current run with do_train={do_train}, do_val={do_val}, do_test={do_test}.")

    use_pref_module = global_config["use_pref_module"]

    # assert not (is_pairwise and not is_full_reference_iqa)  # why is this here?
    assert not (is_debug and is_test_only), "Debug mode disables saving model; can't run test."
    assert do_train or do_test, "Run must have at least training or testing stage."
    assert not (not do_train and do_val), "Validation run requires training to be enabled."

    device = get_device()
    checkpoint_file = global_config["load_checkpoint_file"]
    model = get_model(device, checkpoint_file)

    output_dir = global_config["output_dir"]
    output_dir += "/{}".format(int(time.time()))
    output_dir += "-" + dataset_target()
    output_dir += "-" + global_config["model"]

    if is_vtamiq:
        naming_model_config = vtamiq_config

        if naming_model_config is not None:
            output_dir += "-{}-{}L-{}R".format(
                naming_model_config["vit_config"]["variant"],
                len(model.transformer.encoder.layers),
                naming_model_config["num_rcabs"]
            )

    if is_test_only:
        output_dir += "-TESTSET-" + str(dataloader_config_base[SPLIT_NAME_TEST][PATCH_COUNT])
    else:
        output_dir += "-{}e-{}b-{}p".format(
            global_config["num_epochs"],
            dataloader_config_base[SPLIT_NAME_TRAIN][BATCH_SIZE],
            dataloader_config_base[SPLIT_NAME_TRAIN][PATCH_COUNT]
        )

    if use_pref_module:
        output_dir += "-pref"

    # freeze transformer if fine-tuning on a dataset with a VTAMIQ model pretrained on another dataset
    frozen_model = False
    if is_vtamiq:
        allow_freeze = freeze_config["freeze_vtamiq"]
        freeze_dict = freeze_dict_vtamiq

    else:
        raise NotImplementedError()

    freeze_model = not is_test_only and (
            allow_freeze or
            (freeze_config["freeze_conditional"] and
             (checkpoint_file is None or dataset_target() not in checkpoint_file))
    )

    # keep transformer weights frozen until an appropriate number of epochs are completed
    freeze_end_after_epochs = freeze_config["freeze_end_after_epochs"][dataset_target()]

    if freeze_model:
        output_dir += "-frz"

    # store final output_dir
    if global_config["output_tag"]:
        output_dir += f"-{global_config['output_tag']}"
    global_config["output_dir_final"] = output_dir

    save_val_outputs = global_config["save_val_outputs"] and not is_debug
    save_test_outputs = global_config["save_test_outputs"] and not is_debug
    output_qs_path = output_dir + "/" + global_config["save_test_outputs_txt"]
    val_logger = FileLogger(output_qs_path if save_val_outputs else None, verbose=False)
    test_logger = FileLogger(output_qs_path if save_test_outputs else None, verbose=False)

    if is_debug:
        loger_run = Logger(verbose=is_verbose)  # FileLogger with None as filepath disables logging to file
        writer = None

    else:
        os.makedirs(output_dir, exist_ok=True)

        loger_run = FileLogger("{}/{}".format(output_dir, global_config["output_txt"]), verbose=is_verbose)

        writer = SplitSummaryWriter(
            logdir=output_dir,
            log_every_n_steps=global_config["tensorlog_every_n_steps"],
            max_queue=100,
            flush_secs=10
        )

    if not is_debug:
        loger_run(f"tensorboard --logdir='{output_dir}'")

    # Set the seed if specified
    seed = global_config["seed"]
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if (do_val or do_test) and not do_train:
        global_config["num_epochs"] = 1

    dataloader_config_t = deepcopy(dataloader_config_base)

    # exclusive or, use full dataset when only one run is enabled (train xor val xor test)
    train_xor_val_xor_test = (do_train ^ do_val ^ do_test) and not (do_train and do_val and do_test)
    use_full_dataset = train_xor_val_xor_test and global_config["allow_use_full_dataset"]
    loader_train_t, loader_val_t, loader_test_t, dataset_factory = get_dataloaders(
        use_full_dataset=use_full_dataset, dataloader_config=dataloader_config_t)

    if do_train:
        log_loader_indices(loger_run, loader_train_t, SPLIT_NAME_TRAIN)
    if do_val:
        log_loader_indices(loger_run, loader_val_t, SPLIT_NAME_VAL)
    if do_test:
        log_loader_indices(loger_run, loader_test_t, SPLIT_NAME_TEST)

    checkpoint_every_n_batches = global_config["checkpoint_every_n_batches"]
    if checkpoint_every_n_batches <= 0:
        checkpoint_every_n_batches = 999999999999

    log(f"Model {global_config['model']} info:")
    if global_config["print_flops"]:
        print_flops(model)
    print_parameters(model, full=global_config["print_params"] or global_config["is_debug"])

    pref_module = get_pref_module(use_pref_module, device, checkpoint_file)

    optimizer, scheduler = get_optimizer_scheduler([model, pref_module], loader_train_t)
    scaler = torch.cuda.amp.GradScaler(init_scale=global_config['grad_scale'])

    if freeze_model:
        loger_run("Model Freezing params...")
        frozen_model = True
        model.set_freeze_state(freeze_state=True, freeze_dict=freeze_dict)

        if global_config["print_params"]:
            log("Parameters after freeze:")
            print_parameters(model)

    loger_run("Configuration completed.")

    w_mae_loss = global_config["weight_mae_loss"]
    w_rank_loss = global_config["weight_rank_loss"]
    w_pears_loss = global_config["weight_pears_loss"]

    best_spearman_train = accumulators.Max()
    best_spearman_val = accumulators.Max()
    best_spearman = -1

    correlations = None
    global_step_train = 0
    global_step_val = 0

    if not is_debug:
        save_configs(output_dir)
        save_code(output_dir)
        save_model_params(model, output_dir)

    for epoch in range(global_config["num_epochs"]):
        # increment here to start with 1, not 0
        epoch += 1

        loger_run("Beginning epoch {:03d}".format(epoch))

        # check if need unfreeze model
        if frozen_model and freeze_end_after_epochs < epoch:
            loger_run("VTAMIQ: Unfreezing params...")
            model.set_freeze_state(freeze_state=False, freeze_dict=freeze_dict)
            frozen_model = False  # remove this flag to prevent calling this clause again

            log("Parameters after unfreeze:")
            if global_config["print_params"]:
                print_parameters(model)

        is_best_so_far = False  # this variable will be updated by train and validation runs

        if do_train:
            log("Starting Training loop...")
            global_step_train, correlations = do_training(
                model, scaler, optimizer, scheduler, device, loader_train_t, pref_module,
                w_mae_loss, w_rank_loss, w_pears_loss, is_pairwise_t,
                is_debug, output_dir, loger_run, writer, checkpoint_every_n_batches,
                epoch, global_step_train
            )

            is_best_so_far = best_spearman_train.add(correlations[SROCC_FIELD])
            if is_best_so_far:
                best_spearman = best_spearman_train.value()
                loger_run('Best training SROCC {}!'.format(correlations[SROCC_FIELD]))
            else:
                loger_run(f'Current training SROCC {correlations[SROCC_FIELD]} (best={best_spearman_train.value()}).')

            if not is_debug and global_config["train_save_latest"]:
                loger_run("Saving latest model: epoch=[{}], SROCC=[{}]".format(epoch, correlations[SROCC_FIELD]))
                save_checkpoint(
                    output_dir, "latest.pth", get_models_dict(model, pref_module),
                    optimizer, scaler, epoch, correlations[SROCC_FIELD]
                )

        if do_val:
            log("Starting Validation loop...")
            num_repeats_val = global_config["num_repeats_val"]
            global_step_val, correlations = do_validation(
                model, pref_module, device,
                is_pairwise_t, is_debug,
                w_mae_loss, w_rank_loss, w_pears_loss,
                SPLIT_NAME_VAL, loader_val_t, global_step_val, epoch, writer,
                num_repeats=num_repeats_val, output_logger=val_logger, tag="val"
            )

            is_best_so_far = best_spearman_val.add(correlations[SROCC_FIELD])
            if is_best_so_far:
                best_spearman = best_spearman_val.value()
                loger_run('Best validation SROCC {}!'.format(correlations[SROCC_FIELD]))
            else:
                loger_run(f'Current validation SROCC {correlations[SROCC_FIELD]} (best={best_spearman_val.value()}).')

        # save best based on train/validation results
        if not is_test_only:
            loger_run("Completed epoch {}".format(epoch))

            if is_best_so_far:
                loger_run('Best SROCC {}!'.format(best_spearman))
                if not is_debug:
                    loger_run("Saving best model: epoch=[{}], SROCC=[{}]".format(epoch, best_spearman))
                    save_checkpoint(
                        output_dir, "best.pth", get_models_dict(model, pref_module),
                        optimizer, scaler, epoch, best_spearman
                    )
            else:
                loger_run('Current SROCC {}.'.format(correlations[SROCC_FIELD]))

        # apply loss function decays
        w_mae_loss *= global_config["weight_mae_loss_decay"]
        w_rank_loss *= global_config["weight_rank_loss_decay"]
        w_pears_loss *= global_config["weight_pears_loss_decay"]

    # training/validation is complete
    # pre test cleanup
    del optimizer
    del scheduler
    del scaler
    del loader_train_t
    del loader_val_t
    torch.cuda.empty_cache()  # release used VRAM

    if do_test:
        log("Doing Test.")

        # reload the best saved model from the current session, if training was done
        if do_train and not is_debug:
            saved_model_path = "{}/{}.pth".format(
                output_dir,
                "latest" if (global_config["test_use_latest"] and global_config["train_save_latest"]) else "best"
            )
            model = get_model(device, saved_model_path, force_load_pretrained=True)

        num_repeats_test = global_config["num_repeats_test"]
        _, correlations = do_validation(
            model, pref_module, device,
            is_pairwise_t, is_debug, w_mae_loss, w_rank_loss, w_pears_loss,
            SPLIT_NAME_TEST, loader_test_t, 0, 0, writer,
            num_repeats=num_repeats_test, output_logger=test_logger, tag="test"
        )

        if correlations is not None:
            # logger('Test split:', test_loader_t.dataset.splits_dict[SPLIT_NAME_TEST].indices)
            loger_run(
                f'Test stats:\n' +
                f'{SROCC_FIELD}={correlations[SROCC_FIELD]}\n' +
                f'{KROCC_FIELD}={correlations[KROCC_FIELD]}\n' +
                f'{PLCC_FIELD}={correlations[PLCC_FIELD]}\n' +
                f'{RMSE_FIELD}={correlations[RMSE_FIELD]}\n' +
                f'{PLCC_NOFIT_FIELD}={correlations[PLCC_NOFIT_FIELD]}\n' +
                f'{RMSE_NOFIT_FIELD}={correlations[RMSE_NOFIT_FIELD]}\n'
            )

    if not is_debug:
        writer.close()

    # post test cleanup
    del model
    if use_pref_module:
        del pref_module
    del loader_test_t
    torch.cuda.empty_cache()  # release used VRAM; this helps when train() is performed multiple times

    return correlations


if __name__ == "__main__":
    train()
