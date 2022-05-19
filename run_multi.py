import time

from utils.logging import FileLogger

import run_main
from run_config import *


def parse_runs(runs, logger):
    fields = [SROCC_FIELD, KROCC_FIELD, PLCC_FIELD, RMSE_FIELD]
    stats = {field: [] for field in fields}
    for field in fields:
        for run in runs:
            stats[field].append(run[field])
        result = np.array(stats[field], float)
        logger("{}: mean=[{}], median=[{}], std.dev.=[{}]".format(
            field,
            np.mean(result),
            np.median(result),
            np.std(result),
        ))


def main():
    num_runs = 20  # do at least 2 runs

    global_config["dataset"] = DATASET_LIVE

    dataset_split_config_base["split_type"] = SPLIT_TYPE_RANDOM  # randomize reference images for splits

    global_config["do_train"] = True
    global_config["do_val"] = True
    global_config["do_test"] = True

    global_config["train_save_latest"] = True

    # can start with a pre-trained model
    # global_config["load_checkpoint_file"] = "./output/1619982510-KADIS700k-p16-VTAMIQ-B16-6L-4R-2e-20b-256p/best.pth"

    global_config["optimizer_learning_rate"] = 0.0001
    global_config["num_epochs"] = 20
    global_config["optimizer_decay_at_epochs"] = [10, 15]

    global_config["scheduler_type"] = "multistep"
    global_config["optimizer_learning_rate_decay_multistep"] = 0.1

    output_dir = "./output/{}-multirun-{}".format(int(time.time()), DATASET_USED())
    output_file = "results.txt"

    os.makedirs(output_dir, exist_ok=True)
    logger = FileLogger("{}/{}".format(output_dir, output_file), verbose=True)

    runs = []
    for i in range(num_runs):
        logger("Starting run", i)
        global_config["output_dir"] = output_dir  # tell the script to write to cross-validation folder
        run_ = run_main.main()  # get results
        logger("Finished run", i, ":", run_)
        runs.append(run_)

    parse_runs(runs, logger)


if __name__ == "__main__":
    main()
