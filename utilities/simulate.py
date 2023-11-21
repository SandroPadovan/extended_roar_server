import json
import os
import random
import logging

from environment.settings import TRAINING_CSV_FOLDER_PATH
from environment.state_handling import get_storage_path, is_multi_fp_collection, set_rw_done
from utilities.metrics import write_resource_metrics_to_file, write_syscall_metrics_to_file


# ==============================
# SIMULATE CLIENT BEHAVIOR
# ==============================

UNLIMITED_CONFIGURATIONS = [1, 2]
AVERAGE_RATES = {  # calculated by auxiliary script ´find_avg_rate.py´
    1: 565565.651186441,
    2: 632834.8006,
}


def simulate_sending_fp(config_num):
    config_dir = os.path.join(os.curdir, "rw-configs")
    if config_num in UNLIMITED_CONFIGURATIONS:  # config defines a rate of 0, so we need to collect it from metrics
        rate = AVERAGE_RATES[config_num]
    else:
        with open(os.path.join(config_dir, "config-{}.json".format(config_num)), "r") as config_file:
            config = json.load(config_file)
            rate = int(config["rate"])

    while True:
        try:
            config_fp_dir = os.path.join(TRAINING_CSV_FOLDER_PATH, "infected-c{}".format(config_num), "resource_fp")
            config_syscall_dir = os.path.join(TRAINING_CSV_FOLDER_PATH, "infected-c{}".format(config_num), "syscalls")

            fp_files = os.listdir(config_fp_dir)
            fp_filename = random.choice(fp_files)

            filename_timestamp = fp_filename[fp_filename.index("-")+1:fp_filename.index(".")]
            syscall_filename = f"sc-{filename_timestamp}.csv"

            with open(os.path.join(config_fp_dir, fp_filename)) as fp_file:
                fp = fp_file.read()

            write_resource_metrics_to_file(rate, fp, get_storage_path(), is_multi_fp_collection())
            write_syscall_metrics_to_file(os.path.join(config_syscall_dir, syscall_filename),
                                          get_storage_path(), is_multi_fp_collection())
        except FileNotFoundError as e:
            logging.warning(e)
            continue
        break


def simulate_sending_rw_done():
    set_rw_done()
