import argparse
import math
import os
import random
import time
from pathlib import Path
from ast import literal_eval

from utils.torch_utils import select_device

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='', help='hyperparameters path, i.e. data/hyp.scratch.yaml')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[896, 896], help='train,test sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const='get_last', default=False,
                        help='resume from given path/last.pt, or most recent run if blank')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', type=str, default="False", help='cache images for faster training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', type=str, default="False", help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', type=str, default="False", help='train as single-class dataset')
    parser.add_argument('--adam', type=str, default="False", help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', type=str, default="True", help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--logdir', type=str, default='runs/', help='logging directory')
    opt = parser.parse_args()
    opt.cache_images=literal_eval(opt.cache_images)
    opt.multi_scale=literal_eval(opt.multi_scale)
    opt.single_cls=literal_eval(opt.single_cls)
    opt.adam=literal_eval(opt.adam)
    opt.sync_bn=literal_eval(opt.sync_bn)
    return opt

def train(hyp, opt, device):
    epochs, batch_size, total_batch_size, weights, rank = \
        opt.epochs, opt.batch_size, opt.total_batch_size, opt.weights, int(os.environ.get("RANK", -1))
    

def main(opt):
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    device = select_device(opt.device, batch_size=opt.batch_size)
    opt.total_batch_size = opt.batch_size
    opt.world_size = 1
    opt.global_rank = -1
    if int(os.environ.get("WORLD_SIZE", 1))>1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        assert torch.cuda.device_count() > local_rank
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  
        opt.world_size = int(os.environ.get("WORLD_SIZE", 1)) 
        opt.global_rank = int(os.environ.get("RANK", -1)) 
        assert opt.batch_size % opt.world_size == 0, '--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.total_batch_size // opt.world_size
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    train(hyp, opt, device)