# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from utils.torch_utils import vae_loss_function
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
        self.conv_vae_mu = nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=(3, 3), padding=1)
        self.conv_vae_var = nn.Conv2d(self.feature_channels, self.feature_channels, kernel_size=(3, 3), padding=1)
        if encoder == "resnet18":
            self.encoder = resnet18(self.pretrain, nclasses=10, down_times=self.down_times)
        if encoder == "resnet34":
            self.encoder = resnet34(self.pretrain, nclasses=10, down_times=self.down_times)
        if decoder == "upsample":
            self.decoder = Decoder(in_channels=self.feature_channels, out_channels=self.in_channels, bilinear=True,
                                   base_c=nf, down_times=self.down_times)

        self.classifier = nn.Linear(self.feature_channels, self.num_classes)

    def reparameter(self, mu, log_var):
        eps = torch.randn(size=(mu.shape)).cuda()
        std = torch.sqrt(torch.exp(log_var)).cuda()
        z = mu + eps * std
        return z

    def forward(self, x):
        ret = dict()
        f = self.encoder(x)
        ret.update({'encode': f})
        if self.pretrain:
            # TODO 这个地方看看能不能舍去conv 直接对f进行N~(0,1)的分布规约
            mu = self.conv_vae_mu(f)
            log_var = self.conv_vae_var(f)
            z = self.reparameter(mu, log_var)

            reconstruction = self.decoder(z)
            reconstruction_loss, reparameter_loss = vae_loss_function(reconstruction, x, mu, log_var)
            ret.update({'reconstruction_loss': reconstruction_loss})
            ret.update({'reparameter_loss': reparameter_loss})

            return ret
        else:
            x = avg_pool2d(f, f.shape[2])
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            ret.update({'output': x})
            return ret


if __name__ == '__main__':
    x = torch.randn((2, 3, 32, 32)).cuda()
    down_times = 3
    num_classes = 10
    pretrain: bool = True
    # pretrain: bool = False
    model = FFSS("resnet18", "upsample", pretrain=pretrain, num_classes=num_classes, down_times=down_times).cuda()
    if pretrain:
        # x, reconstruction, loss = model(x)
        ret = model(x)
        # print(x.shape, reconstruction.shape)
        print(ret["reconstruction_loss"])
        print(ret["reparameter_loss"])
    else:
        x = model(x)
        print(x.shape)
