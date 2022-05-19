from data.patch_datasets import dataset_split
from collections import OrderedDict
from copy import deepcopy
import os, yaml

import numpy as np

from modules.Attention.ViT.transformer import VIT_VARIANT_B16, VIT_VARIANT_L16
from modules.vtamiq.vtamiq import VTAMIQ

# ************** CONSTANTS **************

DATASET_TID2013 = "TID2013"
DATASET_TID2008 = "TID2008"
DATASET_LIVE = "LIVE"
DATASET_CSIQ = "CSIQ"
DATASET_PIEAPP_TEST = "PieAPPTestset"
DATASET_PIEAPP_TRAIN = "PieAPPTrainset"
DATASET_PIPAL = "PIPAL"
DATASET_PIPAL_VAL = "PIPALVal"
DATASET_PIPAL_VAL22 = "PIPALVal22"
DATASET_PIPAL_TEST = "PIPALTest"
DATASET_PIPAL_TEST22 = "PIPALTest22"
DATASET_KADID10K = "KADID10k"
DATASET_KONIQ10K = "KONIQ10k"
DATASET_KADIS700k = "KADIS700k"

SPLIT_NAME_TRAIN = "Training"
SPLIT_NAME_VAL = "Validation"
SPLIT_NAME_TEST = "Testing"

SPLIT_TYPE_RANDOM = "random"
SPLIT_TYPE_INDICES = "indices"

PATCH_COUNT = "patch_count"
BATCH_SIZE = "batch_size"
SHUFFLE = "shuffle"
PATCH_FLIP = "allow_img_flip"
IMG_ZERO_ERORR_Q_PROB = "img_zero_error_q_prob"

SROCC_FIELD = 'SROCC'
KROCC_FIELD = 'KROCC'
PLCC_FIELD = 'PLCC'
RMSE_FIELD = 'RMSE'

MODEL_SIQ = "SIQ"
MODEL_SIQG = "SIQG"
MODEL_SIQL = "SIQL"

MODEL_VTAMIQ = "VTAMIQ"
MODEL_VTAMIQp2b = "VTAMIQp2b"
MODEL_VTAMIQp2bd = "VTAMIQp2bd"
MODEL_VTAMIQp2bdqw = "VTAMIQp2bdwq"
MODEL_VTAMIQr = "VTAMIQr"
MODEL_VTAMIQrp2bqw = "VTAMIQrp2bwq"
MODEL_VTAMIQm = "VTAMIQm"

MODEL_LPIPS = "LPIPS"
MODEL_PIEAPP = "PIEAPP"

MODEL_STATE_DICT = "model_state_dict"
Q_MAPPER_STATE_DICT = "q_mapper_state_dict"

# ************** MODELS **************

vtamiq_models = {
    MODEL_VTAMIQ: VTAMIQ,
}

# ************** CONFIGS **************

global_config = OrderedDict(
    is_debug=False,
    is_verbose=True,

    # default value for number of processes
    # dataloader_num_workers=16,
    # dataloader_num_workers=8,
    # dataloader_num_workers=4,
    # dataloader_num_workers=1,
    dataloader_num_workers=-1,  # -1 to control this in script (based on dataset), otherwise to override
    dataloader_pin_memory=False,
    dataloader_persistent_workers=False,

    no_cuda=False,  # toggle to disable gpu

    do_train=False,
    do_val=False,
    do_test=True,

    train_save_latest=True,
    test_use_latest=True,  # use latest or best SROCC model for testing
    save_optimizer=False,

    model=MODEL_VTAMIQ,

    use_display_model=False,  # controls display simulation and PU encoding
    use_q_mapper=False,  # additional module to remap Q

    use_teacher_model=False,  # for pseudo meta labels
    use_teacher_model_train=False,

    dataset=DATASET_LIVE,

    load_checkpoint_file=None,

    # === TRAINING PARAMS ===

    seed=-1,  # -1 for random seed

    num_epochs=20,

    # === OPTIMIZER ===
    optimizer="AdamW",
    optimizer_learning_rate=0.00005,
    # optimizer_learning_rate=0.0001,
    optimizer_learning_rate_decay_multistep=0.1,  # decay rate
    optimizer_learning_rate_decay_lambda=0.85,  # decay rate
    optimizer_weight_decay=0.01,  # 0.025

    grad_scale=256,

    scheduler_type="multistep",  # ["multistep", "lambda"]

    optimizer_cosine_lr=False,
    optimizer_warmup_ratio=0.0,  # period of linear increase for lr scheduler
    optimizer_decay_at_epochs=[14, 18],
    optimizer_momentum=0.9,

    # loss function weights
    # NOTE: MAE loss magnitude is usually ~10x larger than other losses, so assigning smaller weight may be appropriate.
    # training without MAE results in unpredictable scale of the output values (outside 0-1);
    # we don't use activation functions to ensure 0-1 (e.g. sigmoid)
    # weighted combination of the losses performs better.
    weight_mae_loss=1,
    weight_rank_loss=1,
    weight_pears_loss=0.2,

    # decays for loss function weights
    # example usage: want a schedule for losses, i.e. control the weight over time
    # use loss_decay < 1 to multiplicatively modify the weight of particular loss each epoch
    weight_mae_loss_decay=0.975,  # reduce MAE loss slightly over time
    weight_rank_loss_decay=1,
    weight_pears_loss_decay=1,

    # === LOGGING ===
    print_flops=False,
    print_params=True,

    checkpoint_every_n_batches=3000,  # useful for datasets with very large size (KADIS)*

    tensorlog_every_n_steps=10,  # fewer logs for tensorboard writer to limit the amount of logged data

    output_dir="./output",
    output_txt="output.txt",
    debug_txt="debug.txt",

    save_test_outputs=True,
    save_test_outputs_txt="test.txt"
)

display_model_config = OrderedDict(
    rand_distrib_normal=False,
    rand_distrib_ambient_mean=1500,  # randomize ambient around this mean
    rand_distrib_ambient_delta=500,  # randomize ambient with this deviation (sigma if using normal distribution)
    display_Lmax=1000,  # max luminance (cd/m2) of display
    display_lmax_delta=500,
)

q_mapper_config = OrderedDict(
    hidden=32,
    sigmoid=True,
    batch_norm=True,
    non_lin_act=True,
)

# ************** CONFIGURATION FOR MODELS **************

vtamiq_config = OrderedDict(
    use_diff_embedding=False,  # reference image aware difference modulation

    vit_variant=VIT_VARIANT_B16,  # choose from ["B16", "L16"] (VIT_VARIANT_B16, VIT_VARIANT_L16)
    vit_num_keep_layers=6,  # -1 for keeping all layers
    vit_use_scale_embedding=True,  # toggle to use scale embedding for ViT

    num_transformer_diff_layers=2,
    num_residual_groups=4,
    num_rcabs_per_group=4,

    dropout=0.1,

    patch2batch=256  # patch count for encoding with ViT, if p2b strategy is used
)

vtamiq_runtime_config = OrderedDict(
    # there are two stages to pretraining VTAMIQ:
    # i) pretrain only ViT on ImageNet,
    # ii) pretrain the complete VTAMIQ on KADIS-700k

    # toggle whether using pretrained ViT (Imagenet) is allowed
    allow_pretrained_weights=True,

    # if using pretrained VTAMIQ (not just pretrained ViT), toggle whether loading ViT or Diffnet weights is allowed
    allow_pretrained_weights_vit=True,
    allow_pretrained_weights_diffnet=True,

    # Note: freezing features is useful when quality predictor is untrained, but pretrained ViT is used.
    # We don't want to overwrite pretrained BERT features while quality predictor is outputting garbage.
    # Instead, freeze the transformer, spend several epochs training quality predictor, then unfreeze the transformer
    # for combined fine-tuning.
    freeze_vtamiq=False,  # global toggle to allow freezing
    freeze_conditional=False,  # allow freezing based on dataset and checkpoint parameters
    freeze_dict=OrderedDict(
        freeze_diffnet=False,
        freeze_transformer_diff=False,
        freeze_encoder=True,  # ViT encoder
        freeze_embeddings_patch=True,  # ViT patch embeddings
        freeze_embeddings_pos=True,  # ViT positional embeddings
        freeze_embeddings_scale=False,  # ViT scale embeddings
        freeze_q_predictor=False,
        freeze_w_predictor=False,
    ),

    # when to end freezing ViT weights (based on dataset)
    freeze_end_epoch={
        DATASET_TID2013: 1,
        DATASET_TID2008: 2,
        DATASET_LIVE: 3,
        DATASET_CSIQ: 2,
        DATASET_PIPAL: 1,
        DATASET_PIPAL_VAL: 0,
        DATASET_PIPAL_VAL22: 0,
        DATASET_PIPAL_TEST: 0,
        DATASET_PIPAL_TEST22: 0,
        DATASET_PIEAPP_TRAIN: 1,
        DATASET_PIEAPP_TEST: 0,
        DATASET_KADID10K: 1,
        DATASET_KONIQ10K: 1,
        DATASET_KADIS700k: 1,
    }
)
# ************** DATASETS **************

DATASET_USED = lambda: global_config["dataset"]

# this will be passed to each dataset via __init__
dataset_config_base = OrderedDict(
    full_reference=True,  # can set this to False to force NR IQA; only used for FR datasets
    return_full_resolution=False,  # toggle for returning the full resolution images in addition to the patches

    resolution="half",  # only relevant for KONIQ10k dataset, otherwise ignored

    patch_dim=16,
    patch_num_scales=5,  # 5 scales -> 0: 16, 1: 32, 2: 64, 3: 128, 4: 256...

    normalize=True,  # apply normalization on preprocess
    normalize_imagenet=False,  # normalize using imagenet's mean and std dev

    patch_sampler_config=OrderedDict(
        centerbias_weight=0.25,
        diffbased_weight=1,
        uniform_weight=0.1,
    ),
)

dataset_split_config_base = OrderedDict(
    split_type=SPLIT_TYPE_INDICES,  # pick from [SPLIT_TYPE_INDICES, SPLIT_TYPE_RANDOM]
)

dataloader_config = {
    SPLIT_NAME_TRAIN: {
        BATCH_SIZE: 6,  # x256: [train/test] 20, 12 (pairwise); x512: 8, 5 (pairwise)
        SHUFFLE: True,
        PATCH_COUNT: 512,
        PATCH_FLIP: True,
        IMG_ZERO_ERORR_Q_PROB: 0.01,
    },
    SPLIT_NAME_VAL: {
        BATCH_SIZE: 40,
        SHUFFLE: False,
        PATCH_COUNT: 1024,
        PATCH_FLIP: False,
        IMG_ZERO_ERORR_Q_PROB: -1,
    },
    SPLIT_NAME_TEST: {
        BATCH_SIZE: 40,
        SHUFFLE: False,
        PATCH_COUNT: 1024,
        PATCH_FLIP: False,
        IMG_ZERO_ERORR_Q_PROB: -1,
    },
}

tid2013_split_config = {
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 15,
        SPLIT_NAME_VAL: 5,
        SPLIT_NAME_TEST: 5,
    },
    SPLIT_TYPE_INDICES: {
        SPLIT_NAME_TRAIN: [i for i in range(15)],
        SPLIT_NAME_VAL: [i for i in range(15, 20)],
        SPLIT_NAME_TEST: [i for i in range(20, 25)],
    },
}

# TID 2008 has the same format as TID2013, but with less distorted images. Hence the same config can be used
tid2008_split_config = deepcopy(tid2013_split_config)

live_split_config = {
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 17,
        SPLIT_NAME_VAL: 6,
        SPLIT_NAME_TEST: 6,
    },
    SPLIT_TYPE_INDICES: {
        SPLIT_NAME_TRAIN: [i for i in range(17)],
        SPLIT_NAME_VAL: [i for i in range(17, 23)],
        SPLIT_NAME_TEST: [i for i in range(23, 29)],
    },
}

csiq_split_config = {
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 18,
        SPLIT_NAME_VAL: 6,
        SPLIT_NAME_TEST: 6,
    },
    SPLIT_TYPE_INDICES: {
        SPLIT_NAME_TRAIN: [i for i in range(18)],
        SPLIT_NAME_VAL: [i for i in range(18, 24)],
        SPLIT_NAME_TEST: [i for i in range(24, 30)],
    },
}

pieapptrain_split_config = {
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 130,
        SPLIT_NAME_VAL: 10,
        SPLIT_NAME_TEST: 0,
    },
    SPLIT_TYPE_INDICES: {
        SPLIT_NAME_TRAIN: [i for i in range(130)],
        SPLIT_NAME_VAL: [i for i in range(130, 140)],
        SPLIT_NAME_TEST: [i for i in range(140)],  # not used
    },
}

pieapptest_split_config = {
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 24,
        SPLIT_NAME_VAL: 8,
        SPLIT_NAME_TEST: 8,
    },
    SPLIT_TYPE_INDICES: {
        SPLIT_NAME_TRAIN: [i for i in range(24)],  # not used
        SPLIT_NAME_VAL: [i for i in range(24, 32)],  # not used
        SPLIT_NAME_TEST: [i for i in range(40)],  # whole dataset is test
    },
}

pipal_split_config = {
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 120,
        SPLIT_NAME_VAL: 40,
        SPLIT_NAME_TEST: 40,
    },
    SPLIT_TYPE_INDICES: {
        SPLIT_NAME_TRAIN: [i for i in range(120)],
        SPLIT_NAME_VAL: [i for i in range(120, 160)],
        SPLIT_NAME_TEST: [i for i in range(160, 200)],
        },
}

pipaltest_split_config = {
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 0,
        SPLIT_NAME_VAL: 0,
        SPLIT_NAME_TEST: 25,
    },
    SPLIT_TYPE_INDICES: {
        SPLIT_NAME_TRAIN: [i for i in range(15)],  # not used
        SPLIT_NAME_VAL: [i for i in range(15, 20)],  # not used
        SPLIT_NAME_TEST: [i for i in range(25)]},  # whole dataset is test
}

kadid10k_split_config = {
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 49,
        SPLIT_NAME_VAL: 16,
        SPLIT_NAME_TEST: 16,
    },
    SPLIT_TYPE_INDICES: {  # 49, 4
        SPLIT_NAME_TRAIN: [i for i in range(0, 49)],
        SPLIT_NAME_VAL: [i for i in range(49, 65)],
        SPLIT_NAME_TEST: [i for i in range(65, 81)],
    },
}

kadis700k_split_config = {
    # NOTE: KADIS should be used with "indices" split type
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 134260 - 2,  # 134260 total images
        SPLIT_NAME_VAL: 1,
        SPLIT_NAME_TEST: 1,
    },
    SPLIT_TYPE_INDICES: {
        # add the whole dataset to test split since only test is used anyway
        SPLIT_NAME_TRAIN: [i for i in range(134260)],
        SPLIT_NAME_VAL: [0],
        SPLIT_NAME_TEST: [i for i in range(134260)],
    },
}

koniq10k_split_config = {
    SPLIT_TYPE_RANDOM: {
        SPLIT_NAME_TRAIN: 6045,
        SPLIT_NAME_VAL: 2014,
        SPLIT_NAME_TEST: 2014,
    },
    SPLIT_TYPE_INDICES: {
        SPLIT_NAME_TRAIN: [i for i in range(6045)],
        SPLIT_NAME_VAL: [i for i in range(6045, 8059)],
        SPLIT_NAME_TEST: [i for i in range(8059, 10073)],
    },
}


def get_dataset_configs(dataset_name):
    # FULL-REFERENCE DATASETS
    if dataset_name == DATASET_TID2013:
        from data.datasets.tid import TID2013Dataset
        dataset_type = TID2013Dataset
        splits_config = tid2013_split_config

    elif dataset_name == DATASET_TID2008:
        from data.datasets.tid import TID2008Dataset
        dataset_type = TID2008Dataset
        splits_config = tid2008_split_config

    elif dataset_name == DATASET_KADID10K:
        from data.datasets.kadid10k import KADID10kDataset
        dataset_type = KADID10kDataset
        splits_config = kadid10k_split_config

    elif dataset_name == DATASET_KADIS700k:
        from data.datasets.kadis700k import KADIS700kDataset
        dataset_type = KADIS700kDataset
        splits_config = kadis700k_split_config

    elif dataset_name == DATASET_LIVE:
        from data.datasets.live import LIVEDataset
        dataset_type = LIVEDataset
        splits_config = live_split_config

    elif dataset_name == DATASET_CSIQ:
        from data.datasets.csiq import CSIQDataset
        dataset_type = CSIQDataset
        splits_config = csiq_split_config

    elif dataset_name == DATASET_PIEAPP_TRAIN:
        from data.datasets.pieapp_dataset import PieAPPTrainPairwise
        dataset_type = PieAPPTrainPairwise
        splits_config = pieapptrain_split_config

    elif dataset_name == DATASET_PIEAPP_TEST:
        from data.datasets.pieapp_dataset import PieAPPTestset
        dataset_type = PieAPPTestset
        splits_config = pieapptest_split_config

    elif dataset_name == DATASET_PIPAL:
        from data.datasets.pipal import PIPAL
        dataset_type = PIPAL
        splits_config = pipal_split_config

    elif dataset_name == DATASET_PIPAL_VAL:
        from data.datasets.pipal import PIPALVal
        dataset_type = PIPALVal
        splits_config = pipaltest_split_config  # same config as Test

    elif dataset_name == DATASET_PIPAL_VAL22:
        from data.datasets.pipal import PIPALVal22
        dataset_type = PIPALVal22
        splits_config = pipaltest_split_config  # same config as Test

    elif dataset_name == DATASET_PIPAL_TEST:
        from data.datasets.pipal import PIPALTest
        dataset_type = PIPALTest
        splits_config = pipaltest_split_config

    # NO-REFERENCE DATASETS
    elif dataset_name == DATASET_KONIQ10K:
        from data.datasets.koniq10k import KONIQ10k
        dataset_type = KONIQ10k
        splits_config = koniq10k_split_config

    else:
        raise ValueError("Unexpected value for config[dataset] {}".format(dataset_name))

    splits_indices = get_dataset_splits(dataset_type, splits_config, dataset_split_config_base["split_type"])

    return dataset_type, splits_indices


def get_dataset_splits(dataset, dataset_split_config, split_type):
    if split_type == SPLIT_TYPE_INDICES:
        print("Using predefined split indices.")

        return dataset_split_config[SPLIT_TYPE_INDICES]

    elif split_type == SPLIT_TYPE_RANDOM:
        print("Using random split indices.")

        # generate a random ordering given the number of images in the dataset
        split_counts = dataset_split_config[SPLIT_TYPE_RANDOM]

        if sum([split_counts[split_name] for split_name in split_counts]) != dataset.num_ref_images:
            raise ValueError("PatchDataset: sum of random split counts does not match the "
                             "number of images in dataset.")

        split_indices = np.random.permutation(dataset.num_ref_images)

        splits_indices = {}
        count = 0
        for split_name in split_counts:
            split_indices_count = split_counts[split_name]  # number of indices in current split
            splits_indices[split_name] = sorted(deepcopy(split_indices[count: count + split_indices_count]))  # slice
            count += split_indices_count

        return splits_indices

    else:
        raise ValueError("PatchDataset: unsupported split_type {}.".format(split_type))


class DatasetFactory(object):
    def __init__(self,
                 ):
        self.dataset_cache = {}

    def get_dataset(self, dataset_name):
        if dataset_name in self.dataset_cache:
            dataset = self.dataset_cache[dataset_name]
        else:
            dataset = DatasetFactory.__make_dataset(dataset_name)
            self.dataset_cache[dataset_name] = dataset

        return dataset

    @staticmethod
    def __make_dataset(dataset_name):
        iqa_dataset_config_base = deepcopy(dataset_config_base)

        dataset_type, split_config = get_dataset_configs(dataset_name)

        dataset = dataset_type(
            **iqa_dataset_config_base
        )

        for split_name in split_config:
            split = dataset_split(name=split_name, indices=split_config[split_name])
            dataset.add_split(split)

        return dataset


def get_dataloaders(dataset_factory: DatasetFactory = None):
    if dataset_factory is None:
        dataset_factory = DatasetFactory()

    dataset = dataset_factory.get_dataset(DATASET_USED())

    def get_dataloader(dataset, split_name):
        dataloader_params = dataloader_config[split_name]

        from data.patch_datasets import PatchDatasetLoader
        split_loader = PatchDatasetLoader(
            dataset=dataset,
            split_name=split_name,
            batch_size=dataloader_params[BATCH_SIZE],
            patch_count=dataloader_params[PATCH_COUNT],
            allow_img_flip=dataloader_params[PATCH_FLIP],
            img_zero_error_q_prob=dataloader_params[IMG_ZERO_ERORR_Q_PROB],
            shuffle=dataloader_params[SHUFFLE],
            num_workers=global_config["dataloader_num_workers"],
            pin_memory=global_config["dataloader_pin_memory"],
            persistent_workers=global_config["dataloader_persistent_workers"]
        )
        return split_loader

    train_loader = get_dataloader(dataset, SPLIT_NAME_TRAIN)
    val_loader = get_dataloader(dataset, SPLIT_NAME_VAL)
    test_loader = get_dataloader(dataset, SPLIT_NAME_TEST)

    return train_loader, val_loader, test_loader, dataset


# ************** FUNCTIONS **************


def save_configs(output_dir):
    def dump(config, name):
        path = os.path.join(output_dir, "{}.yaml".format(name))
        with open(path, "w") as f:
            yaml.dump(dict(config), f, sort_keys=False)

    dump(global_config, "config")

    if global_config["model"] in vtamiq_models:
        dump(vtamiq_config, "vtamiq_config")
        dump(vtamiq_runtime_config, "vtamiq_runtime_config")

    dump(dataset_config_base, "dataset_config_base")
    dump(dataloader_config, "dataloader_config")

    is_used = lambda database_name: DATASET_USED() == database_name

    if global_config["use_q_mapper"]:
        dump(q_mapper_config, "q_mapper_config")

    if is_used(DATASET_TID2013):
        dump(tid2013_split_config, "tid_dataset_config")
    elif is_used(DATASET_TID2008):
        dump(tid2008_split_config, "tid_dataset_config")
    elif is_used(DATASET_LIVE):
        dump(live_split_config, "live_dataset_config")
    elif is_used(DATASET_KADID10K):
        dump(kadid10k_split_config, "kadid10k_dataset_config")
    elif is_used(DATASET_KADIS700k):
        dump(kadis700k_split_config, "kadis700k_dataset_config")
    elif is_used(DATASET_KONIQ10K):
        dump(koniq10k_split_config, "koniq10k_dataset_config")
    elif is_used(DATASET_PIPAL):
        dump(pipal_split_config, "pipal_split_config")
    elif is_used(DATASET_PIPAL_TEST) or is_used(DATASET_PIPAL_VAL) or is_used(DATASET_PIPAL_VAL22):
        dump(pipaltest_split_config, "pipaltest_split_config")
    elif is_used(DATASET_CSIQ):
        dump(csiq_split_config, "csiq_split_config")
    elif is_used(DATASET_PIEAPP_TRAIN):
        dump(pieapptrain_split_config, "pieapptrain_split_config")
    elif is_used(DATASET_PIEAPP_TEST):
        dump(pieapptest_split_config, "pieapp_split_config")


def validate_configs():
    assert not (
            DATASET_USED() == DATASET_KADIS700k and
            dataset_split_config_base["split_type"] == SPLIT_TYPE_RANDOM
    ), "split_type must be '{}' when using KADIS700k dataset.".format(SPLIT_TYPE_RANDOM)

    # if DATASET_TO_USE() == DATASET_KADIS700k:
    #     vtamiq_runtime_config["allow_freeze_vit"] = False
    #     print("Using", DATASET_KADIS700k, 'dataset. Freezing VTAMIQ ViT was disabled.')

    if global_config["dataloader_num_workers"] == -1:
        dataset = DATASET_USED()
        if dataset == DATASET_LIVE or dataset == DATASET_TID2008 or dataset == DATASET_CSIQ:
            global_config["dataloader_num_workers"] = 4
        elif dataset == DATASET_TID2013 or dataset == DATASET_PIEAPP_TEST:
            global_config["dataloader_num_workers"] = 6
        elif dataset == DATASET_KADID10K or \
                dataset == DATASET_KONIQ10K or \
                dataset == DATASET_KADIS700k or \
                dataset == DATASET_PIPAL or \
                dataset == DATASET_PIPAL_VAL or \
                dataset == DATASET_PIPAL_VAL22 or \
                dataset == DATASET_PIPAL_TEST or \
                dataset == DATASET_PIPAL_TEST22 or \
                dataset == DATASET_PIEAPP_TRAIN:
            global_config["dataloader_num_workers"] = 8
        print("Setting global_config[dataloader_num_workers]={}".format(global_config["dataloader_num_workers"]))

        if dataset == DATASET_PIEAPP_TRAIN:
            print("WARNING: using Pairwise training mode.")


# print("WARNING: call validate_configs() if externally modifying config dicts.")
