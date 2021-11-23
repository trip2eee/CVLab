import os
import torch
import numpy as np
from torch.utils.data import Dataset, dataloader
import cv2

class ImageData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.classes = []

        idx_sample = 0
        with open(os.path.join(root_dir, 'class.txt'), 'r') as f:
            
            while True:
                line = f.readline()
                if line == '':
                    break

                data_class = int(line)
                self.classes.append(data_class)
                img_path = os.path.join(root_dir, '{:04d}.png'.format(idx_sample))
                image = cv2.imread(img_path)

                image = image.astype(np.float32) / 255
                image = np.transpose(image, (2, 0, 1))
                self.images.append(image)

                idx_sample += 1

        self.classes = np.array(self.classes, dtype=np.int64)
        self.images = np.array(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index]
        y = self.classes[index]

        return x, y
        