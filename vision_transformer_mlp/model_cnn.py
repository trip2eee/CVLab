import torch
from torch import nn

class ShapeCNN(nn.Module):

    def __init__(self):
        super(ShapeCNN, self).__init__()

        out_channels = 8
        num_classes = 2

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels*4, out_channels=out_channels*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Dropout(0.5),

            nn.Linear(in_features=3*3*out_channels*4, out_features=num_classes)
        )

    def forward(self, x):
        logits = self.model(x)

        return logits
