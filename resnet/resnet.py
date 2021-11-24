"""
    ResNet
    refered to the implementation https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
"""
import torch
from torch import nn

class BasicBlock(nn.Module):    
    """
        ResNet Basic block
    """
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, kernel_size=3):
        super(BasicBlock, self).__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.activation = nn.ReLU()

        # shortcut
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        y = self.block(x)
        
        if self.shortcut is None:
            y += x
        else:
            y += self.shortcut(x)

        y = self.activation(y)
        return y

class BottleNeckBlock(nn.Module):
    """
        ResNet Bottle Neck Block
    """
    expansion = 4
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, kernel_size=3):
        super(BottleNeckBlock, self).__init__()
        
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        self.activation = nn.ReLU()

        # shortcut
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        y = self.block(x)

        if self.shortcut is None:
            y += x
        else:
            y += self.shortcut(x)

        y = self.activation(y)

        return y

class ResNet(nn.Module):
    def __init__(self, size, dim_output, num_classes):
        super(ResNet, self).__init__()

        self.in_channels = dim_output
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim_output, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(dim_output),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    
        if size == 18:
            self.conv2_x = self.__make_layer(BasicBlock, dim_output*1, dim_output*1, dim_output*1, 2, 1)
            self.conv3_x = self.__make_layer(BasicBlock, dim_output*1, dim_output*2, dim_output*2, 2, 2)
            self.conv4_x = self.__make_layer(BasicBlock, dim_output*2, dim_output*4, dim_output*4, 2, 2)
            self.conv5_x = self.__make_layer(BasicBlock, dim_output*4, dim_output*8, dim_output*8, 2, 2)
            dim_output5_x = dim_output * 8
        elif size == 24:
            self.conv2_x = self.__make_layer(BasicBlock, dim_output*1, dim_output*1, dim_output*1, 3, 1)
            self.conv3_x = self.__make_layer(BasicBlock, dim_output*1, dim_output*2, dim_output*2, 4, 2)
            self.conv4_x = self.__make_layer(BasicBlock, dim_output*2, dim_output*4, dim_output*4, 6, 2)
            self.conv5_x = self.__make_layer(BasicBlock, dim_output*4, dim_output*8, dim_output*8, 3, 2)
            dim_output5_x = dim_output * 8
        elif size == 50:
            self.conv2_x = self.__make_layer(BottleNeckBlock, dim_output*1,  dim_output*1, dim_output*4,  3, 1)
            self.conv3_x = self.__make_layer(BottleNeckBlock, dim_output*4,  dim_output*2, dim_output*8,  4, 2)
            self.conv4_x = self.__make_layer(BottleNeckBlock, dim_output*8,  dim_output*4, dim_output*16, 6, 2)
            self.conv5_x = self.__make_layer(BottleNeckBlock, dim_output*16, dim_output*8, dim_output*32, 3, 2)
            dim_output5_x = dim_output * 32
        elif size == 101:
            self.conv2_x = self.__make_layer(BottleNeckBlock, dim_output*1,  dim_output*1, dim_output*4,  3,  1)
            self.conv3_x = self.__make_layer(BottleNeckBlock, dim_output*4,  dim_output*2, dim_output*8,  4,  2)
            self.conv4_x = self.__make_layer(BottleNeckBlock, dim_output*8,  dim_output*4, dim_output*16, 23, 2)
            self.conv5_x = self.__make_layer(BottleNeckBlock, dim_output*16, dim_output*8, dim_output*32, 3,  2)
            dim_output5_x = dim_output * 32
        else:
            # size == 152:
            self.conv2_x = self.__make_layer(BottleNeckBlock, dim_output*1,  dim_output*1, dim_output*4,  3,  1)
            self.conv3_x = self.__make_layer(BottleNeckBlock, dim_output*4,  dim_output*2, dim_output*8,  8,  2)
            self.conv4_x = self.__make_layer(BottleNeckBlock, dim_output*8,  dim_output*4, dim_output*16, 36, 2)
            self.conv5_x = self.__make_layer(BottleNeckBlock, dim_output*16, dim_output*8, dim_output*32, 3,  2)
            dim_output5_x = dim_output * 32
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim_output5_x, num_classes)

        
    def __make_layer(self, block, in_channels, mid_channels, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i, stride in enumerate(strides):
            if i == 0:
                layers.append(block(in_channels, mid_channels, out_channels, stride))
            else:
                layers.append(block(out_channels, mid_channels, out_channels, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2_x(y)
        y = self.conv3_x(y)
        y = self.conv4_x(y)
        y = self.conv5_x(y)
        y = self.avg_pool(y)
        y = self.flatten(y)
        y = self.fc(y)

        return y

