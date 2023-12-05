import cv2

import os


class CustomDataset(Dataset):

    def __init__(self, path):
        self.samples = []
        for name_folder in ['1', '0']:
            if name_folder == '1':
                self.samples.extend(
                    [(path + name_folder + '/' + name_file, 1) for name_file in os.listdir(path + name_folder)])
            else:
                self.samples.extend(
                    [(path + name_folder + '/' + name_file, 0) for name_file in os.listdir(path + name_folder)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return cv2.resize(cv2.imread(self.samples[idx][0], cv2.IMREAD_GRAYSCALE), (150, 150)), self.samples[idx][1]
