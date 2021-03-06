# This file will implement a script that prepares the train data from the original and augmented train data.
import os
import pickle
import random
import shutil
import time

import torch

import run_train_eval_as_func
from config import Config
import wandb

# set seed
random.seed(10)


def run(i, global_best_acc):
    # select and copy augmented samples to train dir
    selected_augmented_samples = {}
    for letter in config.ROMAN_LETTERS:
        selected_augmented_samples[letter] = []
        letter_origin_imgs = os.listdir(os.path.join(config.TRAIN_DATA_DIR, letter))
        augmented_amount_from_each_image = int(config.TOTAL_IMGS_PER_LETTER / len(letter_origin_imgs)) + 1
        for img in letter_origin_imgs:
            if img.endswith(".png"):
                img_augmented = os.listdir(os.path.join(config.AUGMENTED_DATA_DIR, letter, img[:-4]))
                # sample from the augmented data without replacement
                img_augmented = random.sample(img_augmented, augmented_amount_from_each_image)
                selected_augmented_samples[letter].extend([os.path.join(config.AUGMENTED_DATA_DIR, letter, img[:-4], aug_img) for aug_img in img_augmented])
        # sample from the selected list exactly the allowed amount
        selected_augmented_samples[letter] = random.sample(selected_augmented_samples[letter], config.TOTAL_IMGS_PER_LETTER)

        for aug_img_path in selected_augmented_samples[letter]:
            new_path = os.path.join(config.TRAIN_DATA_DIR, letter, os.path.dirname(aug_img_path).split('/')[-1] + '_' + os.path.basename(aug_img_path))
            shutil.copy(aug_img_path, new_path)

    # save selected samples by categories to WandB
    augmented_statistics = {
        'rotate': len([img for letter in selected_augmented_samples for img in selected_augmented_samples[letter] if 'rotate' in img]),
        'horizontal_flip': len([img for letter in selected_augmented_samples for img in selected_augmented_samples[letter] if 'horizontal_flip' in img]),
        'vertical_flip': len([img for letter in selected_augmented_samples for img in selected_augmented_samples[letter] if 'vertical_flip' in img]),
        'blur_3': len([img for letter in selected_augmented_samples for img in selected_augmented_samples[letter] if 'blur_3' in img]),
        'blur_7': len([img for letter in selected_augmented_samples for img in selected_augmented_samples[letter] if 'blur_7' in img]),
        'blur_11': len([img for letter in selected_augmented_samples for img in selected_augmented_samples[letter] if 'blur_11' in img]),
        'amount_per_letter': config.TOTAL_IMGS_PER_LETTER
    }

    with wandb.init(project="hw2", entity='course094295', config=augmented_statistics):
        # run train_eval.py and get best_acc
        best_acc, model_wts = run_train_eval_as_func.run()
        # upload results to WandB
        wandb.log({"best_acc": best_acc})

        # save selected samples object to WandB
        with open(os.path.join(wandb.run.dir, "selected_augmentations.pkl"), "wb") as f:
            pickle.dump(selected_augmented_samples, f)

        # save model weights
        if best_acc > global_best_acc:
            global_best_acc = best_acc
            torch.save(model_wts, "best_trained_model.pt")

    # remove selected samples from train dir
    for letter in selected_augmented_samples:
        for aug_img_path in selected_augmented_samples[letter]:
            new_path = os.path.join(config.TRAIN_DATA_DIR, letter, os.path.dirname(aug_img_path).split('/')[-1] + '_' + os.path.basename(aug_img_path))
            os.remove(new_path)

    return global_best_acc


if __name__ == '__main__':
    config = Config()

    # amount per letter options
    options = [200, 300, 400, 500, 600, 700, 802]

    global_best_acc = 0
    # loging to WandB
    wandb.login()
    for i in range(100):
        print("Starting run {}".format(i), "************************************************")
        # select amount per letter
        config.TOTAL_IMGS_PER_LETTER = random.choice(options)
        print("Running with amount per letter: {}".format(config.TOTAL_IMGS_PER_LETTER))
        global_best_acc = run(i, global_best_acc)
