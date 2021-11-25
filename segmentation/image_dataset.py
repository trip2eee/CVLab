import os
import torch
import numpy as np
from torch.utils.data import Dataset, dataloader
import cv2

class ImageData(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.labels = []

        list_image_files = os.scandir(root_dir)

        for image_file in list_image_files:

            if image_file.name.startswith('image'):
                image_path = os.path.join(root_dir, image_file.name)
                label_path = os.path.join(root_dir, 'label' + image_file.name[5:])

                # print("{} - {}".format(image_path, label_path))
            
                image = cv2.imread(image_path)
                label = cv2.imread(label_path)

                image = image.astype(np.float32) / 255
                image = np.transpose(image, (2, 0, 1))
                self.images.append(image)

                label = label[:,:,0] / 50
                self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.int64)
        self.images = np.array(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]

        return x, y

if __name__ == "__main__":
    image_data = ImageData("./train")

