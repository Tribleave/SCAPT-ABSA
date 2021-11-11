import argparse
import yaml
from preprocess.process import preprocess

parser = argparse.ArgumentParser(description='Preprocess')
parser.add_argument('--config', required=True, help='path to yaml config file')
args = parser.parse_args()
config = yaml.safe_load(open(args.config))

preprocess(config)
