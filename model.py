import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision  # 加载图片
from torchvision import transforms  # 图片变换

import numpy as np
import matplotlib.pyplot as plt  # 绘图
import os
import glob
from PIL import Image


# 定义下采样模块
class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()  # super函数调用父类（nn.Module）的构造函数
        self.conv_relu = nn.Sequential(  # 定义一个卷积层
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(inplace=True),
        )
        self.bn = nn.BatchNorm2d(num_features=out_channels)  # 归一化处理

    def forward(self, input, is_bn=True):  # 覆写Module类的前向函数
        output = self.conv_relu(input)
        if is_bn:
            output = self.bn(output)
        return output


# 定义上采样模块
class UpSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()  # super函数调用父类（nn.Module）的构造函数
        self.convTranspose_relu = nn.Sequential(  # 定义一个反卷积层
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(num_features=out_channels)  # 归一化处理
        )

    def forward(self, input, is_drop=False):  # 覆写Module类的前向函数
        output = self.convTranspose_relu(input)
        if is_drop:
            output = F.dropout2d(output)
        return output


# 定义Generator，采用U-Net
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.downSampling1 = DownSampling(3, 64)  # 3*256*256 -> 64*128*128
        self.downSampling2 = DownSampling(64, 128)  # 64*128*128 -> 128*64*64
        self.downSampling3 = DownSampling(128, 256)  # 128*64*64 -> 256*32*32
        self.downSampling4 = DownSampling(256, 512)  # 256*32*32 -> 512*16*16
        self.downSampling5 = DownSampling(512, 512)  # 512*16*16 -> 512*8*8
        self.downSampling6 = DownSampling(512, 512)  # 512*8*8 -> 512*4*4

        self.upSampling1 = UpSampling(512, 512)  # 512*4*4 -> 512*8*8
        self.upSampling2 = UpSampling(1024, 512) # 512*8*8 -> 1024*8*8 -> 512*16*16 为了减少数据损失，将下采样中得到的数据拼接进来一起进行上采样
        self.upSampling3 = UpSampling(1024, 256)  # 512*16*16 -> 256*32*32
        self.upSampling4 = UpSampling(512, 128)  # 256*32*32 -> 128*64*64
        self.upSampling5 = UpSampling(256, 64)  # 128*64*64 -> 64*128*128
        self.lastLayer = nn.ConvTranspose2d(  # 64*128*128 -> 3*256*256
            in_channels=128,
            out_channels=3,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

    def forward(self, x):
        x1 = self.downSampling1(x)
        x2 = self.downSampling2(x1)
        x3 = self.downSampling3(x2)
        x4 = self.downSampling4(x3)
        x5 = self.downSampling5(x4)
        x6 = self.downSampling6(x5)

        x6 = self.upSampling1(x6, is_drop=True)
        x6 = torch.cat([x6, x5], dim=1)

        x6 = self.upSampling2(x6, is_drop=True)
        x6 = torch.cat([x6, x4], dim=1)

        x6 = self.upSampling3(x6, is_drop=True)
        x6 = torch.cat([x6, x3], dim=1)

        x6 = self.upSampling4(x6, is_drop=True)
        x6 = torch.cat([x6, x2], dim=1)

        x6 = self.upSampling5(x6, is_drop=True)
        x6 = torch.cat([x6, x1], dim=1)

        x6 = torch.tanh(self.lastLayer(x6))  # 对最后一层的处理
        return x6


# 定义判别器
class Discriminator(nn.Module):  # 根据输入的一对数据（草图+素描图）判断素描图是真实的或生成的，因此输入规模为 6*256*256
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = DownSampling(6, 64)  # 6*256*256 -> 64*128*128
        self.down2 = DownSampling(64, 128)  # 64*128*128 -> 128*64*64
        self.conv = nn.Conv2d(  # 128*64*64 -> 256*62*62
            in_channels=128,
            out_channels=256,
            kernel_size=3
        )
        self.bn = nn.BatchNorm2d(256)
        self.last = nn.Conv2d(256, 1, 3)  # 256*62*62 -> 1*60*60

    def forward(self, line, sketch):
        x = torch.cat([line, sketch], dim=1)
        x = self.down1(input=x, is_bn=False)
        x = self.down2(input=x, is_bn=True)
        x = F.dropout2d(self.bn(F.leaky_relu(self.conv(x))))
        x = torch.sigmoid(self.last(x))
        return x



