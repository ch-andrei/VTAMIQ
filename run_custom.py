from run_config import *
import run_main


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

    # dataset_split_config_base["split_type"] = SPLIT_TYPE_INDICES
    dataset_split_config_base["split_type"] = SPLIT_TYPE_RANDOM

    run_main.main()


if __name__ == "__main__":
    custom_run()
