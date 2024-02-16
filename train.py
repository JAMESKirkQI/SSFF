# -*- coding: utf-8 -*-
import random
import sys
import logging

import torch
import torch.optim as optim
from tqdm import tqdm

from datasets import Cifar10_self_supervised
from models.ffss import FFSS
from utils import forgetting_loss_function, random_number_except, get_lr_scheduler, set_optimizer_lr
from utils.meter import AverageMeter


def train() -> None:
    models_number = 3
    batch_size = 2
    down_times = 3
    num_classes = 10
    expand_ratio = 4
    pretrain: bool = True
    nw = 0
    epochs = 1
    momentum = 0.937
    lr = 1
    log_period = 10
    # optimizer_type = 'adam'
    # weight_decay = 0
    # Init_lr = 1e-3
    # Min_lr = Init_lr * 0.01
    # lr_decay_type = "cos"
    # nbs = 64
    # lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    # lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    # Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    # Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    train_dataset = Cifar10_self_supervised(models_number=models_number, expand_ratio=expand_ratio)
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
        for param in model.parameters():
            param.requires_grad = True
        pg = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(pg, lr=lr, weight_decay=1e-5)
        # print(optimizer)
        # optimizer = optim.SGD(pg, lr=lr, momentum=momentum,
        #                       weight_decay=weight_decay, nesterov=True)
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

    logger = logging.getLogger("SSFF.train")
    logger.info('start training')
    for epoch in range(epochs):
        sample_num = 0
        data_loader = tqdm(train_dataloader, file=sys.stdout)
        for step, data in enumerate(data_loader):
            images = data[0].to(device)
            labels = data[1].to(device)
            sample_num += images.shape[0]
            batch_size = images.shape[1]
            rets = []
            feature_outputs_target = []

            for index, model in enumerate(models):
                # lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, epochs)
                # set_optimizer_lr(optimizers[index], lr_scheduler_func, epoch)
                ret = model(images[index])
                rets.append(ret)
                meters['reparameter_loss'].update(ret.get('reparameter_loss', 0), batch_size)
                meters['reconstruction_loss'].update(ret.get('reconstruction_loss', 0), batch_size)
                feature_outputs_target.append(ret.get('encode', 0).clone().detach())

            for index, model in enumerate(models):
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                select_index = random_number_except(index, models_number)
                forgetting_loss = forgetting_loss_function(rets[index].get('encode', 0),
                                                           feature_outputs_target[select_index],
                                                           labels[index], labels[select_index])
                reparameter_loss = rets[index].get('reparameter_loss', 0)
                reconstruction_loss = rets[index].get('reconstruction_loss', 0)
                print("channel loss:", forgetting_loss[0])
                print("logits loss:", forgetting_loss[1])
                print("reparameter_loss:", reparameter_loss)
                print("reconstruction_loss:", reconstruction_loss)
                # optimizer.zero_grad()

                total_loss = forgetting_loss[0] + forgetting_loss[1] + reparameter_loss + reconstruction_loss
                # total_loss.backward()
                print("total_loss:{}".format(total_loss))
                with torch.autograd.detect_anomaly():
                    # 反向传播的部分代码
                    optimizer.zero_grad()
                    total_loss.backward()

                optimizer.step()
                # optimizers[index].step()
                # optimizers[index].zero_grad()
                meters['forgetting_features_loss'].update(forgetting_loss[0], batch_size)
                meters['forgetting_logits_loss'].update(forgetting_loss[1], batch_size)
