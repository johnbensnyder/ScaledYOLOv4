import os
import argparse
import math
from copy import deepcopy
from pathlib import Path
import yaml

import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.common import *
from models.experimental import MixConv2d, CrossConv, C3
from models.yolo import Detect, parse_model
from utils.general import check_anchor_order, make_divisible, check_file
from utils.torch_utils import (
    time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, select_device)
from utils.datasets import create_dataloader
from utils.losses import YOLOLoss

class Yolo(pl.LightningModule):
    def __init__(self, cfg):  # model, input channels, number of classes
        super().__init__()
        self.cfg = cfg
        
        self.model, self.save = parse_model(deepcopy(self.cfg), ch=[self.cfg['ch']])  # model, savelist, ch_out
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, self.cfg['ch'], s, s))])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.model.to('cuda')
        self.info()
        print('')
        self.optimizer = getattr(torch.optim, self.cfg['optim']['type'])
        self.loss_func = YOLOLoss(self)
    
    def forward(self, x, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si)
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite('img%g.jpg' % s, 255 * xi[0].numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            return self.forward_once(x, profile)  # single-scale inference, train
        
    def forward_once(self, x, profile=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                try:
                    import thop
                    o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # FLOPS
                except:
                    o = 0
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                print('%10.1f%10.0f%10.1fms %-40s' % (o, m.np, dt[-1], m.type))

            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        if profile:
            print('%.1fms total' % sum(dt))
        return x
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            print(('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))
            
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        print('Fusing layers... ', end='')
        for m in self.model.modules():
            if type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatability
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                m.bn = None  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def info(self):  # print model information
        model_info(self)
    
    def train_dataloader(self):
        dataloader, dataset = create_dataloader(self.cfg['data']['train'], 
                                                self.cfg['img_size'], 
                                                self.cfg['batch_size'], 
                                                int(max(self.stride)), 
                                                single_cls=False, 
                                                hyp=self.cfg, 
                                                augment=True,
                                                cache=self.cfg['cache_images'], 
                                                rect=self.cfg['rect'], 
                                                local_rank=self.cfg['local_rank'],
                                                world_size=self.cfg['world_size'],
                                                num_workers=self.cfg['num_workers'])
        return dataloader
    
    '''def val_dataloader(self):
        dataloader, dataset = create_dataloader(self.cfg['data']['val'], 
                                                self.cfg['img_size'], 
                                                self.cfg['batch_size'], 
                                                int(max(self.stride)), 
                                                single_cls=False, 
                                                hyp=self.cfg, 
                                                augment=False,
                                                cache=self.cfg['cache_images'], 
                                                rect=self.cfg['rect'], 
                                                local_rank=self.cfg['local_rank'],
                                                world_size=self.cfg['world_size'],
                                                num_workers=self.cfg['num_workers'])
        return dataloader'''
    
    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), lr=0.001)
        #self.schedule = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(opt,
        #                                                                               1,
        #                                                                               300)
        return opt
    
    def training_step(self, batch):
        opt = self.optimizers()
        opt.zero_grad()
        imgs, targets, _, _ = batch
        imgs = imgs.to('cuda', non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0
        pred = self(imgs)
        loss, loss_items = self.loss_func(pred, targets.to('cuda'))
        loss *= self.cfg['world_size']
        loss.backward()
        opt.step()
        self.scheduler.step()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=False, sync_dist=True)
        