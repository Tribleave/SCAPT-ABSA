import argparse

import yaml

from train.train import train

parser = argparse.ArgumentParser(description='Evaluate model')
parser.add_argument('--config', required=True, help='path to yaml config file')
parser.add_argument('--checkpoint', required=True, help='path to model checkpoint')
args = parser.parse_args()

config = yaml.safe_load(open(args.config))
config['checkpoint'] = args.checkpoint
config.pop('train_file')
if 'dev_file' in config:
    config.pop('dev_file')

train(config)
