"""
Andrei Chubarau:
Work flow inspired by https://github.com/epfml/attention-cnn
"""


import time

from tqdm import tqdm

from tensorboardX import SummaryWriter
import tabulate

import torch.nn.functional as functional


from utils.accumulators import Mean, Max
from utils.correlations import compute_correlations
from utils.logging import FileLogger

from modules.vtamiq.vtamiq import VTAMIQ
from modules.Attention.ViT.transformer import get_vit_config

from modules.utils import *

from run_config import *


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
            accumuls[tag] = Mean()

        accumul = accumuls[tag]
        tag = tags[tag]

        # add new value to mean accumulator
        if isinstance(scalar_value, torch.Tensor):
            scalar_value = scalar_value.detach().cpu().item()
        accumul.add(scalar_value)

        # log mean value if needed and reset accumulator
        if force_add or step % self.log_every_n_steps == 0:
            super(SplitSummaryWriter, self).add_scalar(tag, accumul.value(), global_step=step, walltime=walltime)
            accumul.reset()


def get_optimizer(parameters):
    """
    Create an optimizer for a given model
    :param model_parameters: a list of parameters to be trained
    :return: Tuple (optimizer, scheduler)
    """
    if global_config["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            parameters,
            lr=global_config["optimizer_learning_rate"],
            momentum=global_config["optimizer_momentum"],
            weight_decay=global_config["optimizer_weight_decay"],
        )
    elif global_config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(parameters,
                                     lr=global_config["optimizer_learning_rate"],
                                     weight_decay=global_config["optimizer_weight_decay"])
    elif global_config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(parameters,
                                      lr=global_config["optimizer_learning_rate"],
                                      weight_decay=global_config["optimizer_weight_decay"])
    else:
        raise ValueError("Unexpected value for optimizer")

    if global_config["scheduler_type"] == "lambda":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda e: global_config["optimizer_learning_rate_decay_lambda"] ** e,
            verbose=True,
        )
        # print("Adam/AdamW/AdaBound optimizers ignore all learning rate schedules.")
    elif global_config["scheduler_type"] == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=global_config["optimizer_decay_at_epochs"],
            gamma=global_config["optimizer_learning_rate_decay_multistep"],
            verbose=True,
        )
    else:
        raise ValueError("Unexpected value for scheduler")

    return optimizer, scheduler


def get_model_state_dict(filename, device):
    if filename is None:
        return None

    """Load model from a checkpoint"""
    print("Loading model parameters from '{}'".format(filename))
    with open(filename, "rb") as f:
        checkpoint_data = torch.load(f, map_location=device)

    return checkpoint_data["model_state_dict"]


def get_model(device, is_full_reference, checkpoint_file=None):
    """
    :param device: instance of torch.device
    :return: An instance of torch.nn.Module
    """

    pretrained_model = checkpoint_file is not None
    model = global_config["model"]
    model_state_dict = get_model_state_dict(checkpoint_file, device) if pretrained_model else None

    if model == MODEL_VTAMIQ or model == MODEL_VTAMIQp2b:
        print("Using Model: {}.".format(model))

        # no need to load pretrained ViT transformer if already loading a pretrained VTAMIQ
        vit_load_pretrained = vtamiq_runtime_config["allow_pretrained_weights"]

        model = VTAMIQ(
            get_vit_config(vtamiq_config["vit_variant"]),
            **vtamiq_config,
            vit_load_pretrained=vit_load_pretrained,
            is_full_reference=is_full_reference
            )

        if pretrained_model:
            def pop_layers_fromn_model_state_dict(layer_prefix):
                for layer_name in list(model_state_dict.keys()):
                    if layer_name.contains(layer_prefix):
                        model_state_dict.pop(layer_name)

            if not vtamiq_runtime_config["allow_load_model_vit"]:
                print("Will not load transformer weights from checkpoint file.")
                pop_layers_fromn_model_state_dict("transformer.")

            if not vtamiq_runtime_config["allow_load_model_diffnet"]:
                print("Will not load diffnet weights from checkpoint file.")
                pop_layers_fromn_model_state_dict("diffnet.")

    elif model == MODEL_LPIPS:
        from modules.lpips.lpips_model import LPIPSm
        model = LPIPSm()

    elif model == MODEL_PIEAPP:
        from modules.PerceptualImageError.model.PieAPPv0pt1_PT import PieAPP
        model = PieAPP()

    else:
        raise TypeError("[{}] model is unsupported.".format(model))

    if pretrained_model:
        load_model(model_state_dict, model)

    model.to(device, dtype=torch.float32)
    if device == torch.device("cuda"):
        # from torch.nn.parallel import DistributedDataParallel as DDP
        print("Model {} using GPU".format(global_config["model"]))
        # model = DDP(model)
        # model = torch.nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    return model


def print_parameters(model):
    # compute number of parameters
    num_params, _ = get_num_parameter(model, trainable=False)
    num_bytes = num_params * 4  # assume float32 for all
    print(f"Number of parameters: {human_format(num_params)} ({sizeof_fmt(num_bytes)} for float32)")
    num_trainable_params, trainable_parameters = get_num_parameter(model, trainable=True)
    print("Number of trainable parameters:", human_format(num_trainable_params))

    # Print detailed number of parameters
    print(tabulate.tabulate(trainable_parameters))


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


def store_checkpoint(output_dir, filename, model, epoch, test_accuracy):
    """Store a checkpoint file to the output directory"""
    path = os.path.join(output_dir, filename)

    # Ensure the output directory exists
    directory = os.path.dirname(path)
    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)

    time.sleep(
        1
    )  # workaround for RuntimeError('Unknown Error -1') https://github.com/pytorch/pytorch/issues/10577
    torch.save(
        {
            "epoch": epoch,
            "test_accuracy": test_accuracy,
            "model_state_dict": OrderedDict([
                (key, value) for key, value in model.state_dict().items()
            ]),
        },
        path,
    )


def store_model_params(model, output_dir):
    # save model layout
    num_params, parameters = get_num_parameter(model, trainable=False)
    path = os.path.join(output_dir, "model_params.txt")
    with open(path, "w") as f:
        f.write("Number of parameters: {} ({} for float32)\n".format(
            human_format(num_params),
            sizeof_fmt(num_params * 4)))
        f.write(tabulate.tabulate(parameters))


def get_data_tuple(batch, device):
    convert = lambda x: x.to(device, dtype=torch.float32)
    out_data = tuple()  # q values
    for data in batch:  # patches
        out_data += (convert(data), )
    return out_data


def model_predict(model, data, device, is_full_reference_iqa, is_pairwise, need_pos_input):
    data = get_data_tuple(data, device)

    if is_pairwise:
        q, patches_ref, patches_dist1, patches_dist2, patches_pos, patches_scales = data[:6]

        if need_pos_input:
            q1 = model((patches_ref, patches_dist1), patches_pos, patches_scales)
            q2 = model((patches_ref, patches_dist2), patches_pos, patches_scales)
        else:
            q1 = model((patches_ref, patches_dist1))
            q2 = model((patches_ref, patches_dist2))

        q_p = 1. / (1. + torch.exp(q1 - q2))

    else:
        if is_full_reference_iqa:
            q, patches_ref, patches_dist, patches_pos, patches_scales = data[:5]
            patches = (patches_ref, patches_dist)
        else:
            q, patches_dist, patches_pos, patches_scales = data[:4]
            patches = (patches_dist,)

        if need_pos_input:
            q_p = model(patches, patches_pos, patches_scales)
        else:
            q_p = model(patches)

    return q, q_p


def main():
    # must call validate configs before running training
    validate_configs()

    is_debug = global_config["is_debug"]
    is_verbose = global_config["is_verbose"]
    is_pairwise = global_config["dataset"] == DATASET_PIEAPP_TRAIN

    is_vtamiq = global_config["model"] == MODEL_VTAMIQ or global_config["model"] == MODEL_VTAMIQp2b
    is_lpips = global_config["model"] == MODEL_LPIPS

    is_full_reference_iqa = not DATASET_TO_USE() == "KONIQ10k" and dataset_config_base["full_reference"]

    need_pos_input = is_vtamiq  # need to be passed pos info on forward() call

    checkpoint_file = global_config["load_checkpoint_file"]

    # will run on CUDA if there is a GPU available
    device = torch.device("cuda" if not global_config["no_cuda"] and torch.cuda.is_available() else "cpu")
    model = get_model(device, is_full_reference_iqa, checkpoint_file)

    output_dir = global_config["output_dir"]
    output_dir += "/{}".format(int(time.time()))

    output_dir += "-" + DATASET_TO_USE()
    output_dir += "-p{}".format(dataset_config_base["patch_dim"])

    output_dir += "-" + global_config["model"]

    if is_vtamiq:
        output_dir += "-{}-{}L-{}R".format(
            vtamiq_config["vit_variant"],
            len(model.transformer.encoder.layer),
            vtamiq_config["num_rcabs_per_group"]
        )

    output_dir += "-{}e-{}b-{}p".format(
        global_config["num_epochs"],
        dataloader_config[SPLIT_NAME_TRAIN][BATCH_SIZE],
        dataloader_config[SPLIT_NAME_TRAIN][PATCH_COUNT])

    if not global_config["do_train"] and not global_config["do_val"] and global_config["do_test"]:
        output_dir += "-TESTSET"

    # store final output_dir
    global_config["output_dir"] = output_dir

    validate_configs()

    if is_vtamiq:
        # freeze transformer if
        # 1. not loading a pretrained VTAMIQ
        # 2. fine-tuning on a dataset with a VTAMIQ model pretrained on another dataset
        need_freeze_transformer = \
            vtamiq_runtime_config["vtamiq_allow_freeze"] and \
            (checkpoint_file is None or DATASET_TO_USE() not in checkpoint_file)

        # need_freeze_transformer = False  # OVERRIDE

        # keep transformer weights frozen until this many epochs are completed
        freeze_end_epoch = vtamiq_runtime_config["freeze_end_epoch"][DATASET_TO_USE()]
        # 2 epochs for KADID
        # 5 epochs for TID and LIVE

    if not is_debug:
        os.makedirs(output_dir, exist_ok=True)

        # save config in YAML file
        store_config_files(output_dir)

        store_model_params(model, output_dir)

        writer = SplitSummaryWriter(
            logdir=output_dir,
            log_every_n_steps=global_config["tensorlog_every_n_steps"],
            max_queue=100,
            flush_secs=10
        )

    # FileLogger with None as filepath disables logging to file
    logger_path = None if is_debug else ("{}/{}".format(output_dir, global_config["output_txt"]))
    logger = FileLogger(logger_path, verbose=is_verbose)

    logger_debug_path = None if is_debug else ("{}/{}".format(output_dir, global_config["debug_txt"]))
    logger_debug = FileLogger(logger_debug_path, verbose=is_verbose)

    logger(f"tensorboard --logdir='{output_dir}'")

    # Set the seed
    seed = global_config["seed"]
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if (global_config["do_val"] or global_config["do_test"]) and not global_config["do_train"]:
        global_config["num_epochs"] = 1

    train_loader, val_loader, test_loader, dataset = get_dataloaders(dataset_factory=None)

    logger("splits_dict:", dataset.splits_dict_ref)

    max_steps = global_config["num_epochs"]
    if global_config["optimizer_cosine_lr"]:
        max_steps *= len(train_loader.dataset) // global_config["batch_size_train"] + 1

    checkpoint_every_n_epoch = global_config["checkpoint_every_n_epoch"]
    if checkpoint_every_n_epoch <= 0:
        checkpoint_every_n_epoch = 999999999999

    checkpoint_every_n_batches = global_config["checkpoint_every_n_batches"]
    if checkpoint_every_n_batches <= 0:
        checkpoint_every_n_batches = 999999999999

    if global_config["print_flops"]:
        print_flops(model)

    if global_config["print_params"]:
        print_parameters(model)

    optimizer, scheduler = get_optimizer(model.parameters())

    if is_vtamiq and need_freeze_transformer:
        # freeze VTAMIQ transformer
        model.set_freeze_state(True,
                               diffnet=vtamiq_runtime_config["diffnet_freeze"],
                               encoder=vtamiq_runtime_config["freeze_encoder"],
                               embed_patch=vtamiq_runtime_config["freeze_embeddings_patch"],
                               embed_pos=vtamiq_runtime_config["freeze_embeddings_pos"],
                               embed_scale=vtamiq_runtime_config["freeze_embeddings_scale"],
                               )
        logger("VTAMIQ: Froze parameters...")

    global_step_train = 0
    global_step_val = 0

    logger("Configuration completed.")

    w_mae_loss = global_config["weight_mae_loss"]
    w_rank_loss = global_config["weight_rank_loss"]
    w_pears_loss = global_config["weight_pears_loss"]

    def spearman_loss(x, y):
        """
        Function that measures Spearman’s correlation coefficient between target logits and output logits:
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
        loss = torch.zeros(1, device=device)

        if num_images < 2:
            return loss

        dp = torch.abs(d)

        combinations = torch.combinations(torch.arange(num_images), 2)
        combinations_count = max(1, len(combinations))

        for i, j in combinations:
            rl = torch.clamp_min(-(y[i] - y[j]) * (d[i] - d[j]) / (torch.abs(y[i] - y[j]) + eps), min=0)
            loss += rl / max(dp[i], dp[j])  # normalize by maximum value

        if norm_num:
            loss = loss / combinations_count  # mean

        return loss

    def mae_loss(d, y):
        return functional.l1_loss(d, y)

    def mse_loss(d, y):
        return functional.mse_loss(d, y)

    def loss_func(d, y, num_images):
        w_sum = w_mae_loss + w_rank_loss + w_pears_loss
        loss_mae = mae_loss(d, y)
        loss_rank = rank_loss(d, y, num_images)
        loss_pears = pears_loss(d, y)
        return (w_mae_loss * loss_mae + w_rank_loss * loss_rank + w_pears_loss * loss_pears) / w_sum, \
               loss_mae.detach().item(), \
               loss_rank.detach().item(), \
               loss_pears.detach().item()

    def tensor_list_flat_cat(tensor_list):
        # concatenate and flatten all tensors into one long vector
        return torch.cat(tensor_list, dim=0).flatten()

    def compute_scores(ys, yp):
        ys = np.array(tensor_list_flat_cat(ys), dtype=np.float).flatten()
        yp = np.array(tensor_list_flat_cat(yp), dtype=np.float).flatten()
        spearman, kendall, pearson, rmse = compute_correlations(ys, yp)
        return spearman, kendall, pearson, rmse

    def writer_log_losses(split_name, loss, loss_mae, loss_rank, loss_pears, step):
        writer.add_scalar(split_name, "loss", loss, step)
        writer.add_scalar(split_name, "mae_loss", loss_mae, step)
        writer.add_scalar(split_name, "rank_loss", loss_rank, step)
        writer.add_scalar(split_name, "pears_loss", loss_pears, step)

    def writer_log_losses_pairwise(split_name, loss, step):
        writer.add_scalar(split_name, "mae_loss", loss, step)

    def do_training(scaler, split_name, loader, step):
        # Enable training mode (automatic differentiation + batch norm)
        model.train()

        ys = []
        yp = []
        # loop over training data
        for batch_i, data in enumerate(tqdm(loader)):
            # Compute gradients for the batch
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                q, q_p = model_predict(model, data, device, is_full_reference_iqa, is_pairwise, need_pos_input)
                batch_size = q.shape[0]

                if is_pairwise:
                    loss = mse_loss(q_p, q)
                else:
                    loss, loss_mae, loss_rank, loss_pears = loss_func(q_p, q, batch_size)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            if is_lpips:
                model.clamp_weights()

            ys.append(q.detach().cpu())
            yp.append(q_p.detach().cpu())

            if not is_debug:
                if is_pairwise:
                    writer_log_losses_pairwise(split_name, loss, step)
                else:
                    writer_log_losses(split_name, loss, loss_mae, loss_rank, loss_pears, step)

            step += 1

            if not is_debug and (batch_i + 1) % checkpoint_every_n_batches == 0:  # +1 to skip early save
                logger("Saving batch checkpoint: epoch=[{}], split=[{}], batch_i=[{}]".format(
                    epoch, split_name, batch_i)
                )
                store_checkpoint(output_dir, "{:04d}-{:04d}.pth".format(epoch, batch_i), model, epoch, -1)

        # compute statistics over validation results
        return step, compute_scores(ys, yp)

    def do_validation(scaler, split_name, loader, step, log_writer=True, save_test_outputs=False):
        if global_config["save_test_outputs"]:
            test_logger = FileLogger(output_dir + "/" + global_config["save_test_outputs_txt"])
        y = []
        yp = []
        with torch.no_grad():
            model.eval()
            # loop over validation data
            for i, data in enumerate(tqdm(loader)):
                with torch.cuda.amp.autocast():
                    q, q_p = model_predict(model, data, device, is_full_reference_iqa, is_pairwise, need_pos_input)

                    if is_pairwise:
                        loss = mae_loss(q_p, q)
                    else:
                        batch_size = q.shape[0]
                        loss, loss_mae, loss_rank, loss_pears = loss_func(q_p, q, batch_size)

                y.append(q.cpu())
                yp.append(q_p.cpu())

                logger_debug('q_p:', np.array(q_p.cpu()), "y:", np.array(q.cpu()))

                if log_writer and not is_debug:
                    if is_pairwise:
                        writer_log_losses_pairwise(split_name, loss, step)
                    else:
                        writer_log_losses(split_name, loss, loss_mae, loss_rank, loss_pears, step)

                if save_test_outputs and global_config["save_test_outputs"]:
                    values = list(np.array(q_p.cpu()))
                    values_s = []
                    for value in values:
                        values_s.append(str(value))
                    test_logger(i, ",".join(values_s))

                step += 1

        if 0 < len(y):
            cor = compute_scores(y, yp)
        else:
            cor = 0., 0., 0., 0.

        return step, cor

    def do_split_loop(scaler, split_name, loader, step, best_spearman, is_train=True, do_logs=True):
        do_split_func = do_training if is_train else do_validation
        step, correlations = do_split_func(scaler, split_name, loader, step)  # complete pass over the entire loader
        spearman, kendall, pearson, rmse = correlations

        if do_logs and not is_debug:
            # writer.add_scalar("train/kendall", kendall, epoch)
            writer.add_scalar(split_name, "SROCC", spearman, epoch, force_add=True)
            writer.add_scalar(split_name, "PLCC", pearson, epoch, force_add=True)
            writer.add_scalar(split_name, "RMSE", rmse, epoch, force_add=True)
            if is_train:
                writer.add_scalar(split_name, "LR", scheduler.get_last_lr()[0], epoch, force_add=True)

        if is_train:
            # Update the optimizer's learning rate after training epoch end
            if global_config["optimizer_cosine_lr"]:
                scheduler.step(step)
            else:
                scheduler.step()

        is_best_so_far = best_spearman.add(spearman)
        if is_best_so_far:
            logger('{} best spearman {}!'.format(split_name, spearman))
        else:
            logger('{} spearman {}.'.format(split_name, spearman))

        return step, spearman, is_best_so_far, correlations

    def get_results_dict(spearman, kendall, pearson, rmse):
        return {
            SROCC_FIELD: spearman,
            KROCC_FIELD: kendall,
            PLCC_FIELD: pearson,
            RMSE_FIELD: rmse,
        }

    scaler = torch.cuda.amp.GradScaler()

    best_spearman_train = Max()
    best_spearman_val = Max()

    results = None

    for epoch in range(global_config["num_epochs"]):
        epoch += 1  # start with 1, not 0

        logger("Beginning epoch {:03d}".format(epoch))

        # check if need unfreeze VTAMIQ
        if is_vtamiq and vtamiq_runtime_config["vtamiq_allow_freeze"] and epoch > freeze_end_epoch:
            logger("VTAMIQ: Unfreezing transformer layers.")
            model.set_freeze_state(False)
            vtamiq_runtime_config["vtamiq_allow_freeze"] = False  # remove this flag to prevent calling this clause again

        # default parameters
        split_name = "missing"
        spearman = -1
        is_best_so_far = False  # this variable will be updated by train and validation runs

        if global_config["do_train"]:
            print("Starting Training loop...")
            split_name = SPLIT_NAME_TRAIN
            global_step_train, spearman, is_best_so_far, correlations = \
                do_split_loop(scaler, split_name, train_loader, global_step_train, best_spearman_train, is_train=True)
            results = get_results_dict(*correlations)

        if global_config["do_val"]:
            print("Starting Validation loop...")
            split_name = SPLIT_NAME_VAL
            global_step_val, spearman, is_best_so_far, correlations = \
                do_split_loop(scaler, split_name, val_loader, global_step_val, best_spearman_val, is_train=False)
            results = get_results_dict(*correlations)

        logger("Completed epoch {}".format(epoch))

        # apply loss function decays
        global_config["weight_mae_loss"] *= global_config["weight_mae_loss_decay"]
        global_config["weight_rank_loss"] *= global_config["weight_rank_loss_decay"]
        global_config["weight_pears_loss"] *= global_config["weight_pears_loss_decay"]

        if global_config["do_train"] and not is_debug:
            # Store checkpoints for the best model so far
            if global_config["train_save_latest"] or is_best_so_far:
                logger("Saving best model: epoch=[{}], split=[{}], SROCC=[{}]".format(
                    epoch, split_name, spearman)
                )
                store_checkpoint(output_dir, "best.pth", model, epoch, spearman)

            if epoch % checkpoint_every_n_epoch == 0:
                logger("Saving checkpoint: epoch=[{}], split=[{}], SROCC=[{}]".format(
                    epoch, split_name, spearman)
                )
                store_checkpoint(output_dir, "{:04d}.pth".format(epoch), model, epoch, spearman)

    # training/validation is complete
    if global_config["do_test"]:
        print("Doing Test.")

        # if training was done during the current session, reload the best saved model from the current session
        if global_config["do_train"]:
            model = get_model(device, is_full_reference_iqa, "{}/best.pth".format(output_dir))

        step, correlations = do_validation(scaler, SPLIT_NAME_TEST, test_loader, 0, log_writer=False, save_test_outputs=True)
        spearman, kendall, pearson, rmse = correlations

        # logger('Test split:', test_loader.dataset.splits_dict[SPLIT_NAME_TEST].indices)
        logger('Test stats:\n' +
               '{}={}\n'.format(SROCC_FIELD, spearman) +
               '{}={}\n'.format(KROCC_FIELD, kendall) +
               '{}={}\n'.format(PLCC_FIELD, pearson) +
               '{}={}\n'.format(RMSE_FIELD, rmse)
               )

        results = get_results_dict(*correlations)

    if not is_debug:
        writer.close()

    # del model
    # del optimizer
    # del train_loader
    # del val_loader
    # del test_loader
    torch.cuda.empty_cache()  # release all used video memory; this helps when train() is performed multiple times

    return results


if __name__ == "__main__":
    main()
