# This file will implement a script to choose the best augmented examples according to the saved augmentation information
# and copy them to the train dir

import os
import pickle
import shutil

from config import Config


def select_augmentation_data(selected_augmentation):
    # select and copy augmented samples to train dir
    for letter in config.ROMAN_LETTERS:
        for aug_img_path in selected_augmentation[letter]:
            new_path = os.path.join(config.TRAIN_DATA_DIR, letter,
                                    os.path.dirname(aug_img_path).split('/')[-1] + '_' + os.path.basename(aug_img_path))
            shutil.copy(aug_img_path, new_path)


if __name__ == '__main__':
    config = Config()
    with open(config.AUGMENTATION_INFO_PATH, 'rb') as f:
        selected_augmentations = pickle.load(f)

    select_augmentation_data(selected_augmentations)
