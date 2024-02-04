# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from datasets import Cifar10_self_supervised
from models.ffss import FFSS
from torchsummary import summary


def train() -> None:
    models_number = 2
    batch_size = 1
    down_times = 3
    num_classes = 10
    pretrain: bool = True
    nw = 0
    epochs=100
    weight_decay=0.0005
    momentum=0.937
    lr=0.001
    train_dataset = Cifar10_self_supervised(models_number=models_number, expand_ratio=1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)
    model = FFSS("resnet18", "upsample", pretrain=pretrain, num_classes=num_classes,
                   down_times=down_times).cuda()
    pg = [p for p in model.parameters() if p.requires_grad]
    summary(model, input_size=(3, 32, 32), batch_size=-1)
    optimizer = optim.SGD(pg, lr=lr, momentum=momentum,
                          weight_decay=weight_decay, nesterov=True)

    for epoch in range(epochs):
        pass
