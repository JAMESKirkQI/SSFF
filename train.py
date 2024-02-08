# -*- coding: utf-8 -*-
import sys
import logging

import torch
import torch.optim as optim
from tqdm import tqdm

from datasets import Cifar10_self_supervised
from models.ffss import FFSS
from utils.meter import AverageMeter


def train() -> None:
    models_number = 3
    batch_size = 2
    down_times = 3
    num_classes = 10
    pretrain: bool = True
    nw = 0
    epochs = 1
    weight_decay = 0.0005
    momentum = 0.937
    lr = 0.001
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_dataset = Cifar10_self_supervised(models_number=models_number, expand_ratio=1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)
    models = [FFSS("resnet18", "upsample", pretrain=pretrain, num_classes=num_classes,
                   down_times=down_times).to(device) for i in range(models_number)]
    print(len(models))
    pgs = []
    optimizers = []
    for model in models:
        model.train()
        pg = [p for p in model.parameters() if p.requires_grad]
        # summary(model, input_size=(3, 32, 32), batch_size=-1)
        optimizer = optim.SGD(pg, lr=lr, momentum=momentum,
                              weight_decay=weight_decay, nesterov=True)
        optimizer.zero_grad()
        pgs.append(pg)
        optimizers.append(optimizer)

    meters = {
        "loss": AverageMeter(),
        "reparameter_loss": AverageMeter(),
        "reconstruction_loss": AverageMeter(),
        "forgetting_logits_loss": AverageMeter(),
        "forgetting_features_loss": AverageMeter(),
    }

    for epoch in range(epochs):
        sample_num = 0
        data_loader = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]
            batch_size = images.shape[1]
            feature_outputs=[]

            for index, model in enumerate(models):
                ret = model(images[index])
                meters['reparameter_loss'].update(ret.get('reparameter_loss', 0), batch_size)
                meters['reconstruction_loss'].update(ret.get('reconstruction_loss', 0), batch_size)

                feature_outputs.append(ret.get('encode', 0))
            # TODO 做一下feature上面的特征拉取工作FCCL+
            for index, model in enumerate(models):
                # TODO 把各个模型根据label交集的补集做一个对比学习
                #  先把交集的补集写完，mask要给出来然后算loss后乘一下mask
                pass
                # TODO 首先在channel方向，然后在logits方向，两个损失待实现