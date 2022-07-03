import os


class Config:

    def __init__(self):
        # General
        self.TOTAL_IMGS_PER_LETTER = 802

        # important paths
        self.MAIN_DATA_DIR = "data"
        self.ALL_DATA_DIR = os.path.join(self.MAIN_DATA_DIR, "all_data")
        self.TRAIN_DATA_DIR = os.path.join(self.MAIN_DATA_DIR, "train")
        self.VAL_DATA_DIR = os.path.join(self.MAIN_DATA_DIR, "val")
        self.TEST_DATA_DIR = os.path.join(self.MAIN_DATA_DIR, "test")
        self.AUGMENTED_DATA_DIR = os.path.join(self.MAIN_DATA_DIR, "augmented_train")
        self.AUGMENTATION_INFO_PATH = os.path.join(self.MAIN_DATA_DIR, "selected_augmentations.pkl")

        # Roman Letters
        self.ROMAN_LETTERS = {'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x'}

        # split data
        self.VAL_RATIO = 0.1
        self.TEST_RATIO = 0.2
