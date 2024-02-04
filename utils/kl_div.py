# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    print(torch.arange(0, 10).view(2, -1).view(-1))

    x = torch.randn((2, 2, 2, 2)).view(2, 2, -1)
    y = torch.ones((2, 2, 2, 2)).view(2, 2, -1)

    # 因为要用y指导x,所以求x的对数概率，y的概率
    logp_x = F.log_softmax(x, dim=-1)
    p_y = F.softmax(y, dim=-1)

    kl_sum = F.kl_div(logp_x, p_y, reduction='sum')
    kl_mean = F.kl_div(logp_x, p_y, reduction='mean')
    print(kl_sum)
    print(kl_mean)
