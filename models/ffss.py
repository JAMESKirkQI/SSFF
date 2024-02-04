# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from backbone.encoder import *
from backbone.decoder import *


class FFSS(nn.Module):
    def __init__(self, encoder: str, decoder: str, in_channels: int = 3, num_classes: int = 10,
                 nf: int = 64, pretrain: bool = True, down_times: int = 2):
        super(FFSS, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_channels = nf * (2 ** down_times)
        self.down_times = down_times
        self.pretrain = pretrain
        if encoder == "resnet18":
            self.encoder = resnet18(self.pretrain, nclasses=10, down_times=self.down_times)
        if encoder == "resnet34":
            self.encoder = resnet34(self.pretrain, nclasses=10, down_times=self.down_times)
        if decoder == "upsample":
            self.decoder = Decoder(in_channels=self.feature_channels, out_channels=self.in_channels, bilinear=True,
                                   base_c=nf, down_times=self.down_times)

    def forward(self, x):
        x = self.encoder(x)
        if self.pretrain:
            reconstruction = self.decoder(x)
            return x, reconstruction
        else:
            return x


if __name__ == '__main__':
    x = torch.randn((10, 3, 32, 32)).cuda()
    down_times = 3
    num_classes = 10
    pretrain: bool = True
    # pretrain: bool = False
    model = FFSS("resnet18", "upsample", pretrain=pretrain, num_classes=num_classes, down_times=down_times).cuda()
    if pretrain:
        x, reconstruction = model(x)
        print(x.shape, reconstruction.shape)
    else:
        x = model(x)
        print(x.shape)
