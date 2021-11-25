"""
    Metrics.py 
"""
from numpy.core import numeric
import torch
from torch import nn
import numpy as np

def create_conf_matrix(y_true, y_pred):
    """
        This function computes confusion matrix.
    """
    batch, num_classes, rows, cols = y_pred.shape

    y_pred = torch.argmax(y_pred, dim=1).cpu().numpy()
    y_true = y_true.cpu().numpy()

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    for idx_true in range(0, num_classes):
        for idx_pred in range(0, num_classes):
            idx = np.where((y_pred == idx_pred) & (y_true == idx_true))
            conf_matrix[idx_true, idx_pred] = idx[0].size

    return conf_matrix


def compute_IoU(conf_matrix):
    """
        This function computes IoU from confusion matrix.
    """
    num_classes = conf_matrix.shape[0]

    iou = np.zeros(num_classes)
    for i in range(num_classes):
        
        inter = conf_matrix[i, i]
        union = conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - inter
        iou[i] = inter / union
    
    return iou


if __name__ == "__main__":
    y_true = [[0, 1, 2],
              [0, 1, 2],
              [0, 1, 2]]
    
    y_pred = [
        [[[1, 0, 1],
          [1, 0, 0],
          [1, 0, 0]],

        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 1]]]
    ]
    
    """
    3  0  0
    0  3  0
    1  0  2
    """
    y_true = torch.Tensor(y_true)
    y_pred = torch.Tensor(y_pred)

    conf_matrix_gt = [[3, 0, 0], [0, 3, 0], [1, 0, 2]]
    conf_matrix = create_conf_matrix(y_true=y_true, y_pred=y_pred)

    print(conf_matrix)

    print(conf_matrix == conf_matrix_gt)

    IoU = compute_IoU(conf_matrix)
    print(IoU)

    y = y_true.reshape(-1, 1, 3, 3)
    y = y.permute(3, 1, 2, 0)
    print(y)

