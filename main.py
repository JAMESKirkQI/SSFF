# -*- coding: utf-8 -*-
import os
import sys
import socket
import torch.multiprocessing

from train import train
from utils import init_seeds

torch.multiprocessing.set_sharing_strategy('file_system')

conf_path = os.getcwd()
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')

from argparse import ArgumentParser
import setproctitle
import torch
import uuid
import datetime


def parse_args():
    parser = ArgumentParser(description='You Only Need Me', allow_abbrev=False)
    parser.add_argument('--device_id', type=int, default=7, help='The Device Id for Experiment')
    parser.add_argument('--parti_num', type=int, default=0, help='The Number for Participants')  # Domain 4 Label 10
    parser.add_argument('--decoder', type=str, default='resnet',
                        # fccl,fcclss, fedmd fedavg fedprox feddf moon fedmatch fcclplus fcclsuper
                        help='Model name.', choices=["resnet18"])
    parser.add_argument('--structure', type=str, default='homogeneity')  # 'homogeneity' heterogeneity

    parser.add_argument('--dataset', type=str, default='fl_digits',
                        # fl_digits, fl_officehome fl_office31,fl_officecaltech
                        choices=["cifar10"], help='Which scenario to perform experiments on.')
    parser.add_argument('--beta', type=int, default=0.1, help='The Beta for Label Skew')
    parser.add_argument('--pub_aug', type=str, default='weak')  # weak strong

    parser.add_argument('--get_time', action='store_true')

    torch.set_num_threads(4)
    parser.add_argument('--seed', type=int, default=0,
                        help='The random seed.')

    parser.add_argument('--csv_log', action='store_true',
                        help='Enable csv logging')
    args = parser.parse_args()

    if args.seed is not None:
        init_seeds(args.seed)
    if args.parti_num == 0:
        if args.dataset in ['cifar10']:
            args.parti_num = 10
        if args.dataset in ['fl_digits', 'fl_officehome', 'fl_officecaltech']:
            args.parti_num = 4
        if args.dataset in ['fl_office31']:
            args.parti_num = 3

    return args


def main(args=None):
    if args is None:
        args = parse_args()
    train()


if __name__ == '__main__':
    main()
