# -*- coding: utf-8 -*-
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # print(x.shape)
        # [N, C, H, W]
        x = self.conv(x)
        # print(x.shape)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Decoder(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 bilinear: bool = True,
                 base_c: int = 64,
                 down_times: int = 2
                 ):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.down_times = down_times

        # self.in_conv = DoubleConv(self.in_channels, base_c)

        # self.up1 = Up(base_c * 16, base_c * 8, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, self.out_channels)
        # self._features_up8 = nn.Sequential(
        #     self.up1,
        #     self.up2,
        #     self.up3,
        #     self.up4
        # )
        self._features_up4 = nn.Sequential(
            self.up2,
            self.up3,
            self.up4
        )
        self._features_up2 = nn.Sequential(
            self.up3,
            self.up4
        )
        self._features_up1 = nn.Sequential(
            self.up4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.down_times == 1:
            out = self._features_up1(x)
        elif self.down_times == 2:
            out = self._features_up2(x)
        else:
            out = self._features_up4(x)

        x = self.out_conv(out)
        return x


if __name__ == '__main__':
    x = torch.randn((1, 256, 8, 8)).cuda()
    net = Decoder(in_channels=x.shape[1], out_channels=3).cuda()
    x4 = net(x, 2)
    print(x4.shape)
