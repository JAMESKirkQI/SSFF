# -*- coding: utf-8 -*-
batch_size   = 64

config = {}
networks = {}
net_opt = {}
net_opt['num_classes'] = 10
net_opt['num_stages']  = 3
networks['model'] = {'def_file': 'backbone/resnet.py', 'pretrained': None, 'opt': net_opt}
config['networks'] = networks