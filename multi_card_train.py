import argparse
import os

import torch
import yaml

from train.train import train

os.environ['NCCL_LL_THRESHOLD'] = '0'

parser = argparse.ArgumentParser(description='Train model on multiple cards')
parser.add_argument('--config', help='path to yaml config file')
parser.add_argument('--local_rank', type=int, help='local gpu id')
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
torch.distributed.init_process_group(backend='nccl', init_method='env://')
config['local_rank'] = args.local_rank
torch.cuda.set_device(args.local_rank)

train(config)
