# This file will run a script that will augment the train data after the data has been split.
import os
import random
from itertools import product

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, Compose, RandomVerticalFlip, GaussianBlur, ColorJitter

from config import Config

# set seed
random.seed(42)


class AddGaussianNoise(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor + np.random.randn(tensor.shape[0], tensor.shape[1], tensor.shape[2])


def rotate_image(letter):
    r_param = augmentation_params['rotate']
    if r_param == 0:
        return None
    else:
        return f'rotate_{r_param}', RandomRotation(r_param, fill=255)


def horizontal_flip_image(letter):
    if letter == 'vii' or letter == 'viii' or letter == 'ix':
        return None
    hf_param = augmentation_params['horizontal_flip']
    if hf_param == 0:
        return None
    else:
        return f'horizontal_flip_{hf_param}', RandomHorizontalFlip(hf_param)


def vertical_flip_image(letter):
    if letter != 'x':
        return None
    vf_param = augmentation_params['vertical_flip']
    if vf_param == 0:
        return None
    else:
        return f'vertical_flip_{vf_param}', RandomVerticalFlip(vf_param)


def blur_image(letter):
    b_param = augmentation_params['blur']
    if b_param == 0:
        return None
    else:
        return f'blur_{b_param}', GaussianBlur(b_param)


def deduce_augmented_letter(letter, transformer_names):
    if 'horizontal_flip' in transformer_names:
        if letter == 'iv':
            return 'vi'
        elif letter == 'vi':
            return 'iv'

    return letter



def augment_data():
    augmentation_methods = ['rotate', 'horizontal_flip', 'vertical_flip', 'blur']
    # shuffle order
    random.shuffle(augmentation_methods)
    for letter in config.ROMAN_LETTERS:
        letter_dir = os.path.join(config.TRAIN_DATA_DIR, letter)

        transformations = []
        for method in augmentation_methods:
            transformation = augmentation_methods_to_func[method](letter)
            if transformation is not None:
                transformations.append(transformation)
        transformer = Compose([t[1] for t in transformations])

        for img_path in os.listdir(letter_dir):
            if img_path.endswith('.png'):
                img = cv2.imread(os.path.join(letter_dir, img_path))
                img = torch.from_numpy(img).permute(2, 0, 1)

                transformer_names = [t[0] for t in transformations]
                augmented_letter = deduce_augmented_letter(letter, transformer_names)
                augmented_path = os.path.join(config.AUGMENTED_DATA_DIR, augmented_letter, img_path.replace('.png', ''),
                                              '_'.join(transformer_names) + '.png')
                # make directory if it doesn't exist
                if not os.path.exists(os.path.dirname(augmented_path)):
                    os.makedirs(os.path.dirname(augmented_path))
                new_img = transformer(img).numpy().transpose((1, 2, 0))
                cv2.imwrite(augmented_path, new_img)


def imshow(inp, title=None):
    """Imshow for Tensors."""
    # inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


if __name__ == '__main__':
    config = Config()

    augmentation_parameters_options = {
        'rotate': [0, 30],
        'horizontal_flip': [0, 1],
        'vertical_flip': [0, 1],
        'blur': [0, 3, 7, 11]
    }

    # dict of augmentation methods to functions
    augmentation_methods_to_func = {'rotate': rotate_image,
                                    'horizontal_flip': horizontal_flip_image,
                                    'vertical_flip': vertical_flip_image,
                                    'blur': blur_image,
                                    }

    print("Augmenting data...")
    # running through all augmentation parameter combinations:

    all_combs = product(*[opts for opts in augmentation_parameters_options.values()])
    for i, comb in enumerate(all_combs):
        if i == 0:
            continue
        augmentation_params = dict(zip(augmentation_parameters_options.keys(), comb))
        print("Augmenting with: ", augmentation_params)
        augment_data()
