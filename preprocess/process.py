import os
import pickle
from preprocess.utils import (
        parse_term,
        preprocess_pretrain,
        preprocess_finetune,
        )


MODE = {
    'preprocess_pretrain': preprocess_pretrain,
    'preprocess_finetune': preprocess_finetune,
}


def preprocess(config):
    base_path = config['base_path']
    train_path = os.path.join(base_path, config['train_file'])
    dev_path = (os.path.join(base_path, config['dev_file'])
                if 'dev_file' in config
                else None)
    test_path = (os.path.join(base_path, config['test_file'])
                 if 'test_file' in config
                 else None)

    lowercase = config['lowercase'] if 'lowercase' in config else False
    remove_list = config['remove_list'] if 'remove_list' in config else []

    train_data = parse_term(train_path,
                            lowercase=lowercase, remove_list=remove_list)
    dev_data = parse_term(dev_path,
                          lowercase=lowercase,
                          remove_list=remove_list) if dev_path else None
    test_data = parse_term(test_path,
                           lowercase=lowercase,
                           remove_list=remove_list) if test_path else None

    if config['mode'] in MODE:
        make_data = MODE[config['mode']]
    else:
        raise TypeError(f"Not supported preprocess mode {config['mode']}")

    train_data = make_data(train_data)
    dev_data = make_data(dev_data) if dev_data else None
    test_data = make_data(test_data) if test_data else None

    train_save_path = '.'.join(
            train_path.split('.')[:-1]) + '_{}.pkl'.format(config['mode'])
    pickle.dump(train_data, open(train_save_path, 'wb'))

    if dev_data:
        dev_save_path = '.'.join(
                dev_path.split('.')[:-1]) + '_{}.pkl'.format(config['mode'])
        pickle.dump(dev_data, open(dev_save_path, 'wb'))

    if test_data:
        test_save_path = '.'.join(
                test_path.split('.')[:-1]) + '_{}.pkl'.format(config['mode'])
        pickle.dump(test_data, open(test_save_path, 'wb'))
