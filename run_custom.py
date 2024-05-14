from train_config import *
import train


def test_custom_vtamiq():
    global_config["load_checkpoint_file"] = None

    global_config["do_train"] = False
    global_config["do_val"] = False
    global_config["do_test"] = True

    global_config["dataset"] = DATASET_CSIQ

    # dataset_split_config_base["split_type"] = SPLIT_TYPE_RANDOM
    dataset_split_config_base["split_type"] = SPLIT_TYPE_INDICES

    train.train()


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

    train.train()


def custom_run():
    global_config["is_debug"] = True
    global_config["dataloader_num_workers"] = 1

    # TRAIN PARAMS
    global_config["model"] = MODEL_VTAMIQ
    global_config["load_checkpoint_file"] = None

    global_config["do_train"] = True
    global_config["do_val"] = True
    global_config["do_test"] = True
    global_config["allow_use_full_dataset_test"] = False

    global_config["dataset"] = DATASET_TID2013

    global_config["num_epochs"] = 20
    global_config["optimizer_learning_rate"] = 0.0001
    global_config["scheduler_type"] = "lambda"

    train.train()


if __name__ == "__main__":
    custom_run()
