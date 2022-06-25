# This file will run a script that will split the data into train, val, and test sets.

import os
import shutil
from typing import List
import random

from config import Config

# set seed
random.seed(42)


def copy_file(img_path, new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    new_path = os.path.join(new_dir, os.path.basename(img_path))
    shutil.copy(img_path, new_path)


def get_image_paths_per_letter(data_dir):
    img_paths_per_letter = {let: [] for let in config.ROMAN_LETTERS}
    assert os.path.exists(data_dir), f"{data_dir} does not exist"
    assert set(os.listdir(data_dir)) == config.ROMAN_LETTERS, f"{data_dir} does not contain all roman letters"
    for letter_name in os.listdir(data_dir):
        letter_dir = os.path.join(data_dir, letter_name)
        for img_path in os.listdir(letter_dir):
            img_paths_per_letter[letter_name].append(os.path.join(letter_dir, img_path))
    return img_paths_per_letter


def split_train_val_test_for_letter(letter: str, letter_img_paths: List):
    train_img_paths = []
    val_img_paths = []
    test_img_paths = []
    # shuffle the list
    random.shuffle(letter_img_paths)
    for img_path in letter_img_paths:
        if len(train_img_paths) < len(letter_img_paths) * (1 - config.VAL_RATIO - config.TEST_RATIO):
            train_img_paths.append(img_path)
        elif len(val_img_paths) < len(letter_img_paths) * config.VAL_RATIO:
            val_img_paths.append(img_path)
        else:
            test_img_paths.append(img_path)
    # letter dirs
    letter_train_dir = os.path.join(config.TRAIN_DATA_DIR, letter)
    letter_val_dir = os.path.join(config.VAL_DATA_DIR, letter)
    letter_test_dir = os.path.join(config.TEST_DATA_DIR, letter)

    for img_path in train_img_paths:
        copy_file(img_path, letter_train_dir)
    for img_path in val_img_paths:
        copy_file(img_path, letter_val_dir)
    for img_path in test_img_paths:
        copy_file(img_path, letter_test_dir)


def split_train_val_test():
    img_paths_per_letter = get_image_paths_per_letter(data_dir=config.ALL_DATA_DIR)
    for letter in img_paths_per_letter:
        split_train_val_test_for_letter(letter, img_paths_per_letter[letter])
    # sanity check, new location contains all files
    count = 0
    for split in [config.TRAIN_DATA_DIR, config.VAL_DATA_DIR, config.TEST_DATA_DIR]:
        split_images = get_image_paths_per_letter(data_dir=split)
        for letter in split_images:
            count += len(split_images[letter])
    assert count == sum([len(img_paths_per_letter[let]) for let in img_paths_per_letter]), f"{count} != {len(img_paths_per_letter)}"


if __name__ == '__main__':
    config = Config()

    # split data
    print("Splitting data...")
    split_train_val_test()
    print("Done.")
