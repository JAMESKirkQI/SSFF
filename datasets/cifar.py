# -*- coding: utf-8 -*-
import copy
import random

import torch
from torchvision.transforms import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
import numpy as np
from matplotlib import pyplot as plt
from utils.forget_box import draw_forgetting_box
from utils.utils import init_seeds

init_seeds()


def CIFAR_transform(img):
    augmentation_transforms = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomRotation(90),
        RandomResizedCrop(32, scale=(0.8, 1), ratio=(0.8, 1.2)),
        RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8)]
    )
    return augmentation_transforms(img)


def bulid_Cifar10_transform(img):
    augmentation_transforms = Compose([
        ToTensor(),
        Normalize((0.4802, 0.4480, 0.3975),
                  (0.2770, 0.2691, 0.2821))
    ])
    return augmentation_transforms(img)


class Cifar10_supervised_ntask(Dataset):
    def __init__(self, path='/data', train=True, transform=bulid_Cifar10_transform):
        self.CIFAR10 = datasets.CIFAR10(root=path, train=train, download=True, transform=CIFAR_transform)
        self.transform = transform

    def __getitem__(self, index):
        images, labels = self.CIFAR10[index]
        images = self.transform(images)
        labels = torch.FloatTensor(labels)
        return images, labels

    def __len__(self):
        return len(self.CIFAR10)

    def collate_fn(batch):
        images, labels = zip(*batch)
        return images, labels


class Cifar10_self_supervised(Dataset):
    def __init__(self, path='/data', models_number=4, expand_ratio=3, train=True, transform=bulid_Cifar10_transform,
                 size_ratio=None,
                 # size_ratio=(0.25, 0.25),
                 mask_value=0):
        self.CIFAR10 = datasets.CIFAR10(root=path, train=train, download=True, transform=CIFAR_transform)
        self.transform = transform
        self.size_ratio = size_ratio
        self.mask_value = mask_value
        self.models_number = models_number
        self.expand_ratio = expand_ratio


    def __getitem__(self, index):
        masked_images = []
        images = []
        labels = []
        image = np.array(self.CIFAR10[index][0])
        images.append(image)
        for i in range(self.expand_ratio):
            image = np.array(self.CIFAR10[random.randint(0, len(self.CIFAR10))][0])
            images.append(image)
        _, masked_image, x_cut, y_cut, w_cut, h_cut = draw_forgetting_box(images[0], size_ratio=self.size_ratio,
                                                                          mask_value=self.mask_value)
        masked_images.append(masked_image)
        labels.append([x_cut, y_cut, w_cut, h_cut])

        for k in range(self.models_number):
            for image in images[1:]:
                image, masked_image, x_cut, y_cut, w_cut, h_cut = draw_forgetting_box(image, size_ratio=self.size_ratio,
                                                                                      mask_value=self.mask_value)
                masked_images.append(masked_image)
                labels.append([x_cut, y_cut, w_cut, h_cut])
        all_models_images = [[masked_images[0]] for i in range(self.models_number)]
        all_models_labels = [[labels[0]] for i in range(self.models_number)]
        for j in range(self.models_number):
            all_models_images[j].extend(masked_images[j * self.expand_ratio + 1:(j + 1) * self.expand_ratio + 1])
            all_models_labels[j].extend(labels[j * self.expand_ratio + 1:(j + 1) * self.expand_ratio + 1])
        for i in range(self.models_number):
            for j in range(self.expand_ratio + 1):
                all_models_images[i][j] = self.transform(all_models_images[i][j])

        return all_models_images, all_models_labels

    def __len__(self):
        return len(self.CIFAR10)

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        bs_images = []
        bs_labels = []
        for index, imgs in enumerate(images):
            bs_images.append(torch.stack([torch.stack(i, dim=0) for i in imgs], dim=0))
        images = torch.cat(bs_images, dim=1)

        for label in labels:
            bs_labels.append(torch.stack([torch.FloatTensor(t) for t in label], 0))
        labels = torch.cat(bs_labels, dim=1)
        del bs_images, bs_labels
        return images, labels


if __name__ == '__main__':
    train_dataset = Cifar10_self_supervised()
    # print(dataset.CIFAR10)
    batch_size = 2
    nw = 0
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   num_workers=nw,
                                                   # Shuffle=True unless rectangular training is used
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   collate_fn=train_dataset.collate_fn)
    for img, target in train_dataloader:
        print(img.shape)
        print(target.shape)
        # img_torch = torch.zeros(img[0][0].shape)
        # x = torch.stack([torch.stack(i, dim=0) for i in img], dim=0)
        # print(x.shape)
        # print([len(t) for t in target])
        # print(torch.stack([torch.FloatTensor(t) for t in target], 0))
        # print(torch.stack([torch.FloatTensor(t) for t in target], 0).shape)
        # np_img = np.array(img)
        # print(np_img, np_img.shape, type(target))
        # pil_img = Image.fromarray(np.uint8(img))
        # print(img, target)
        # img.show()
        break
    # T1 = torch.tensor([[1, 2, 3],
    #                    [4, 5, 6],
    #                    [7, 8, 9]])
    # # 假设是时间步T2的输出
    # T2 = torch.tensor([[10, 20, 30],
    #                    [40, 50, 60],
    #                    [70, 80, 90]])
    # y1 = [T1, T2]
    # y2 = [T2, T1]
    # y = [y1, y2]
    # y_tensor = torch.stack([torch.stack(i, dim=0) for i in y], dim=0)
    # y_copy = copy.deepcopy(y_tensor)
    # yy = torch.cat([y_tensor, y_copy], dim=1)
    # print(y_tensor.shape)
    # print(yy.shape)
    # print(yy)
