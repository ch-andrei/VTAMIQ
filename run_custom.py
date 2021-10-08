from run_config import *
import run_main


def train_custom_vtamiq():
    # load a KADIS pretrained model, better than nothing
    global_config["load_checkpoint_file"] = "./output/1613318904-KADIS700k-p16-QBERT-B16-6L-4R-20e-9b-500p/best.pth"
    vtamiq_runtime_config["allow_pretrained_vit"] = True  # load default pre-trained ViT transformer
    vtamiq_runtime_config["allow_load_model_vit"] = False  # prevent loading ViT from load_checkpoint_file
    vtamiq_runtime_config["allow_load_model_diffnet"] = True  # allow loading Diffnet from load_checkpoint_file
    vtamiq_runtime_config["allow_freeze_vit"] = False
    vtamiq_config["vit_num_keep_layers"] = -1

    global_config["optimizer_learning_rate"] = 0.0001

    global_config["num_epochs"] = 20

    global_config["optimizer_decay_at_epochs"] = [14, 18]

    # global_config["scheduler_type"] = "multistep"
    # global_config["optimizer_learning_rate_decay_multistep"] = 0.1

    global_config["scheduler_type"] = "lambda"
    global_config["optimizer_learning_rate_decay_lambda"] = 0.875

    global_config["weight_rank_loss_decay"] = 0.95

    dataset_config_base["patch_config"] = {
        SPLIT_NAME_TRAIN: 512,
        SPLIT_NAME_VAL: 1024,
        SPLIT_NAME_TEST: 1024,
    }

    dataloader_config[SPLIT_NAME_TRAIN] = {
        BATCH_SIZE: 8,
        SHUFFLE: True,
    }

    dataloader_config[SPLIT_NAME_VAL] = {
        BATCH_SIZE: 16,
        SHUFFLE: False,
    }

    dataloader_config[SPLIT_NAME_TEST] = {
        BATCH_SIZE: 16,
        SHUFFLE: False,
    }

    run_main.main()


def test_custom_vtamiq():
    global_config["load_checkpoint_file"] = None

    global_config["do_train"] = False
    global_config["do_val"] = False
    global_config["do_test"] = True

    global_config["dataset"] = DATASET_CSIQ

    # dataset_split_config_base["split_type"] = SPLIT_TYPE_RANDOM
    dataset_split_config_base["split_type"] = SPLIT_TYPE_INDICES

    dataloader_config[SPLIT_NAME_TEST][BATCH_SIZE] = 12
    dataloader_config[SPLIT_NAME_TEST][PATCH_COUNT] = 2048

    run_main.main()


def train_custom_lpips():
    global_config["dataset"] = DATASET_PIEAPP_TRAIN
    dataset_split_config_base["split_type"] = SPLIT_TYPE_INDICES

    global_config["model"] = "LPIPS"

    global_config["load_checkpoint_file"] = None

    global_config["do_train"] = True
    global_config["do_val"] = True
    global_config["do_test"] = False

    global_config["optimizer_lplearning_rate"] = 0.00005
    global_config["num_epochs"] = 3
    global_config["optimizer_decay_at_epochs"] = [1, 2]
    global_config["scheduler_type"] = "multistep"
    global_config["optimizer_learning_rate_decay_multistep"] = 0.1

    dataset_config_base["patch_dim"] = 224

    dataloader_config[SPLIT_NAME_TRAIN][BATCH_SIZE] = 32
    dataloader_config[SPLIT_NAME_TRAIN][PATCH_COUNT] = 4
    dataloader_config[SPLIT_NAME_VAL][BATCH_SIZE] = 32
    dataloader_config[SPLIT_NAME_VAL][PATCH_COUNT] = 4
    dataloader_config[SPLIT_NAME_TEST][BATCH_SIZE] = 32
    dataloader_config[SPLIT_NAME_TEST][PATCH_COUNT] = 4

    run_main.main()


def custom_test():
    # Example pre-trained model
    global_config["load_checkpoint_file"] = \
        "./output/1633010482-PieAPPTrainset-p16-VTAMIQ-B16-6L-4R-2e-12b-256p/best.pth"

    global_config["model"] = MODEL_VTAMIQ

    global_config["do_train"] = False
    global_config["do_val"] = False
    global_config["do_test"] = True

    global_config["dataset"] = DATASET_PIEAPP_TEST

    dataset_split_config_base["split_type"] = SPLIT_TYPE_INDICES
    # dataset_split_config_base["split_type"] = SPLIT_TYPE_RANDOM

    dataloader_config[SPLIT_NAME_TEST][BATCH_SIZE] = 10
    dataloader_config[SPLIT_NAME_TEST][PATCH_COUNT] = 2048

    run_main.main()


def custom_run():
    global_config["load_checkpoint_file"] = None

    global_config["model"] = MODEL_VTAMIQ

    global_config["do_train"] = True
    global_config["do_val"] = True
    global_config["do_test"] = False

    # global_config["dataset"] = DATASET_KADIS700k
    # global_config["dataset"] = DATASET_PIEAPP_TRAIN
    global_config["dataset"] = DATASET_LIVE

    global_config["optimizer_learning_rate"] = 0.0001
    global_config["num_epochs"] = 20
    global_config["optimizer_decay_at_epochs"] = [10, 15]
    global_config["scheduler_type"] = "multistep"
    global_config["optimizer_learning_rate_decay_multistep"] = 0.1

    dataset_split_config_base["split_type"] = SPLIT_TYPE_INDICES
    # dataset_split_config_base["split_type"] = SPLIT_TYPE_RANDOM

    run_main.main()


if __name__ == "__main__":
    custom_run()
