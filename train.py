# -*- coding: utf-8 -*-
import sys

import torch
import torch.optim as optim
from tqdm import tqdm

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
    epochs = 1
    weight_decay = 0.0005
    momentum = 0.937
    lr = 0.001
    train_dataset = Cifar10_self_supervised(models_number=models_number, expand_ratio=1)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)
    models = [FFSS("resnet18", "upsample", pretrain=pretrain, num_classes=num_classes,
                   down_times=down_times).cuda() for i in range(models_number)]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(len(models))
    pgs=[]
    optimizers=[]
    for model in models:
        pg = [p for p in model.parameters() if p.requires_grad]
        # summary(model, input_size=(3, 32, 32), batch_size=-1)
        optimizer = optim.SGD(pg, lr=lr, momentum=momentum,
                              weight_decay=weight_decay, nesterov=True)
        pgs.append(pg)
        optimizers.append(optimizer)

    for epoch in range(epochs):

        for model in models:
            model.train()
        loss_function = torch.nn.CrossEntropyLoss()
        accu_loss = torch.zeros(1).to(device)  # 累计损失
        accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
        for optimizer in optimizers:
            optimizer.zero_grad()
        sample_num = 0
        data_loader = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images, labels = data
            sample_num += images.shape[0]
            # print(images.shape, labels.shape)
            # break
