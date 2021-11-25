from model_segnet import SegNet
import torch
from image_dataset import ImageData
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

import metrics

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = SegNet(4).to(device)
model.load_state_dict(torch.load('best_weight_segnet.pth'))
model.eval()

data_test = ImageData('test')
test_dataloader = DataLoader(data_test, batch_size=1, shuffle=False)

# Evaluation
conf_matrix = None
for X, y in test_dataloader:
    X = X.to(device)
    y_pred = model(X)

    if conf_matrix is None:
        conf_matrix = metrics.create_conf_matrix(y_true=y, y_pred=y_pred)
    else:
        conf_matrix += metrics.create_conf_matrix(y_true=y, y_pred=y_pred)

print("Confusion Matrix")
print(conf_matrix)
IoU = metrics.compute_IoU(conf_matrix)
print("IoU")
print(IoU)
mIoU = IoU.mean()
print("mIoU: {}".format(mIoU))


plt.figure("prediction")
image_to_show = 10

def convert_into_rgb(y):
    color_map = [[0, 0, 0],[0, 255, 0], [255, 0, 0], [255, 255, 0]]
    _, height, width = y.shape
    
    image_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    
    for idx_cls in range(1, 4):
        idx = np.where(y.reshape(height, width) == idx_cls)
        image_rgb[idx[0], idx[1], :] = color_map[idx_cls]

    return image_rgb

idx_image = 0
for X, y in test_dataloader:
    X = X.to(device)
    y_pred = model(X)
    y_pred = torch.argmax(y_pred, dim=1)

    _, height, width = y.shape

    y = y.numpy()
    image_y = convert_into_rgb(y)
    plt.subplot(image_to_show, 2, idx_image*2 + 1)    
    plt.imshow(image_y)

    y_pred = y_pred.cpu().numpy()
    image_y_pred = convert_into_rgb(y_pred)
    plt.subplot(image_to_show, 2, idx_image*2 + 2)
    plt.imshow(image_y_pred)

    idx_image += 1
    if idx_image >= image_to_show:
        break
plt.show()

