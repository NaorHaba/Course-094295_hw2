# This file will implement a script to check our score on the held out test dir
import os

import run_train_eval_as_func
from config import Config

if __name__ == '__main__':
    config = Config()

    # # change val dir to val_temp dir
    # os.rename(config.VAL_DATA_DIR, config.VAL_DATA_DIR + '_temp')
    #
    # # change test dir to val dir
    # os.rename(config.TEST_DATA_DIR, config.VAL_DATA_DIR)

    # run test
    run_train_eval_as_func.run()

    # # change val dir to test dir
    # os.rename(config.VAL_DATA_DIR, config.TEST_DATA_DIR)
    #
    # # change val_temp dir to val dir
    # os.rename(config.VAL_DATA_DIR + '_temp', config.VAL_DATA_DIR)
