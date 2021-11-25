import torch
from torch import nn

class SegNet(nn.Module):
    def __init__(self, num_classes=4):
        super(SegNet, self).__init__()

        out_channels = 16

        self.layers = nn.Sequential(
            self.__create_encoder(3, out_channels, 2),
            self.__create_encoder(out_channels, out_channels*2, 2),
            self.__create_encoder(out_channels*2, out_channels*4, 3),
            self.__create_encoder(out_channels*4, out_channels*8, 3),
            # self.__create_encoder(out_channels*8, out_channels*16, 3),

            # self.__create_decoder(out_channels*16, out_channels*8, 3),
            self.__create_decoder(out_channels*8, out_channels*4, 3),
            self.__create_decoder(out_channels*4, out_channels*2, 3),
            self.__create_decoder(out_channels*2, out_channels, 2),
            self.__create_decoder(out_channels, out_channels, 2, last_decoder=True),
            nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
        )

    def __create_encoder(self, in_channels, out_channels, num_layers, kernel_size=3):
        layers = []
        padding = kernel_size // 2

        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))
            else:
                layers.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))
            
            layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        return nn.Sequential(*layers)

    def __create_decoder(self, in_channels, out_channels, num_layers, last_decoder=False, kernel_size=3):
        layers = []
        padding = kernel_size // 2

        layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        for i in range(num_layers):
            if i < (num_layers-1):
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding))
            else:
                layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding))
            
            if not last_decoder:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers(x)

        return y

if __name__ == "__main__":
    x = torch.Tensor([[1, 2],
                      [3, 4]])
    x = x.reshape((1, 1, 2, 2))
    
    y = nn.UpsamplingNearest2d(scale_factor=2)(x)

    print(y)
