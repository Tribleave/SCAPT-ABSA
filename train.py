import argparse

import yaml

from train.train import train

parser = argparse.ArgumentParser(description='Train model')
parser.add_argument('--config', required=True, help='path to yaml config file')
parser.add_argument('--checkpoint', help='path to model checkpoint')
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
if 'checkpoint' in args:
    config['checkpoint'] = args.checkpoint

train(config)
