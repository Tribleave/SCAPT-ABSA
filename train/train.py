from train.misc import set_seed, set_device
from train.trainer.pretrain import fp16_multi_pretrain
from train.trainer.aspect_finetune import aspect_finetune


def train(config):
    set_seed(config['seed'])
    if config['mode'] == 'fp16_multi_pretrain':
        return fp16_multi_pretrain(config)
    set_device(config['device'])
    if config['mode'] == 'aspect_finetune':
        return aspect_finetune(config)
    raise TypeError(f"Not supported train mode {config['mode']}")
