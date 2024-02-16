import math
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


def init_seeds(seed=0):
    torch.manual_seed(seed)

    # Reduce randomness (may be slower on Tesla GPUs) # https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:
        cudnn.deterministic = False
        cudnn.benchmark = True


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-4
            m.momentum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def model_info(model, verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, 224, 224),), verbose=False)
        fs = ', %.1f GFLOPS' % (macs / 1E9 * 2)
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


class ModelEMA:
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.
    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    I've tested with the sequence in my own train.py for torch.DataParallel, apex.DDP, and single-GPU.
    """

    def __init__(self, model, decay=0.9999, device=''):
        # make a copy of the model for accumulating moving average of weights
        self.ema = deepcopy(model)
        self.ema.eval()
        self.updates = 0  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # decay exponential ramp (to help early epochs)
        self.device = device  # perform ema on different device from model if set
        if device:
            self.ema.to(device=device)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)
        with torch.no_grad():
            if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
                msd, esd = model.module.state_dict(), self.ema.module.state_dict()
            else:
                msd, esd = model.state_dict(), self.ema.state_dict()

            for k, v in esd.items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

    def update_attr(self, model):
        # Assign attributes (which may change during training)
        for k in model.__dict__.keys():
            if not k.startswith('_'):
                setattr(self.ema, k, getattr(model, k))


reconstruction_function = nn.BCEWithLogitsLoss(reduction='sum').to(device)  # mse loss


def vae_loss_function(recon_x, x, mu, log_var):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(log_var.exp()).mul_(-1).add_(1).add_(log_var)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE, KLD


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def calculate_isd_sim(features, mask, temp=0.002):
    features = F.normalize(features, dim=1)
    features = features * mask
    sim_q = torch.einsum("ijkl,pjkl->ip", features, features)
    logits_mask = torch.scatter(
        torch.ones_like(sim_q),
        1,
        torch.arange(sim_q.size(0)).view(-1, 1).to(device),
        0
    )
    row_size = sim_q.size(0)
    sim_q = sim_q[logits_mask.bool()].view(row_size, -1)
    return sim_q / temp


def forgetting_loss_function(x, y, x_labels, y_labels):
    # x_label [0.56250, 0.50000, 0.37500, 0.34375]
    # calculate mask set
    bs, _, w, h = x.shape
    mask = torch.ones(bs, 1, w, h).to(device)

    for i, label in enumerate(zip(x_labels, y_labels)):
        x_label, y_label = label
        x1, x1_cut = math.floor(x_label[0] * w), math.floor(x_label[2] * w)
        y1, y1_cut = math.floor(x_label[1] * h), math.floor(x_label[3] * h)
        mask[i, :, x1:x1 + x1_cut, y1:y1 + y1_cut] = 0
        x2, x2_cut = math.floor(y_label[0] * w), math.floor(y_label[2] * w)
        y2, y2_cut = math.floor(y_label[1] * h), math.floor(y_label[3] * h)
        mask[i, :, x2:x2 + x2_cut, y2:y2 + y2_cut] = 0

    # calculate loss for channel
    q_1_bn = ((x - x.mean(0)) / x.std(0)) * mask
    q_2_bn = ((y - y.mean(0)) / y.std(0)) * mask
    # empirical cross-correlation matrix
    v = torch.einsum("ijkl,iokl->jo", q_1_bn, q_2_bn)
    v = torch.div(v, bs)

    on_diag_index = ~torch.diagonal(v).isnan()
    off_diag_index = ~off_diagonal(v).isnan()
    on_diag = torch.diagonal(v).add_(-1).pow_(2)[on_diag_index].sum()
    off_diag = off_diagonal(v).add_(1).pow_(2)[off_diag_index].sum()
    col_loss = on_diag + 0.0051 * off_diag

    # calculate loss for logits
    x_logits = calculate_isd_sim(x, mask, temp=0.002)
    y_logits = calculate_isd_sim(y, mask, temp=0.002)
    inputs = F.log_softmax(x_logits, dim=1)
    targets = F.softmax(y_logits, dim=1)
    loss_distill = F.kl_div(inputs, targets, reduction='batchmean')
    loss_distill = 3.0 * loss_distill

    return col_loss, loss_distill
