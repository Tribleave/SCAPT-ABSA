import numpy as np
import random
import os
import gc
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def set_device(device=-1):
    if device == -1:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
    else:
        if isinstance(device, int):
            os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
        elif isinstance(device, list):
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device))


def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def get_model_path(model_name, model_path):
    now = datetime.now().strftime('%m%d%H%M%S')
    dir_name = "{}_{}".format(model_name, now)
    path = os.path.join(model_path, dir_name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def prepare_dataset(config, absa_dataset, collate_fn=default_collate):
    data_path = config['data_path']
    train_path = os.path.join(data_path, config['train_file']) if 'train_file' in config else None
    dev_path = os.path.join(data_path, config['dev_file']) if 'dev_file' in config else None
    test_path = os.path.join(data_path, config['test_file']) if 'test_file' in config else None

    train_loader = DataLoader(absa_dataset(train_path),
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=config['num_workers_per_loader'],
                              collate_fn=collate_fn) if train_path else None
    dev_loader = DataLoader(absa_dataset(dev_path),
                            batch_size=config['batch_size'],
                            shuffle=False,
                            num_workers=config['num_workers_per_loader'],
                            collate_fn=collate_fn) if dev_path else None
    test_loader = DataLoader(absa_dataset(test_path),
                             batch_size=config['batch_size'],
                             shuffle=False,
                             num_workers=config['num_workers_per_loader'],
                             collate_fn=collate_fn) if test_path else None
    if dev_loader is None and test_loader is not None:
        dev_loader = test_loader
    return train_loader, dev_loader, test_loader


def get_masked_inputs_and_labels(bert_tokens, aspect_mask):
    # Modified from https://keras.io/examples/nlp/masked_language_modeling/
    # from transformers import BertTokenizer
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # mask_token_id = tokenizer.convert_tokens_to_ids('[MASK]')
    mask_token_id = 103
    # Select aspect tokens
    total_count = (bert_tokens > mask_token_id).long().sum().item()
    aspect_count = aspect_mask.long().sum().item()
    threshold = min(1.0, 0.15 * total_count / aspect_count if aspect_count else 0.0)
    mask = (np.random.rand(*aspect_mask.shape) < threshold) & aspect_mask.bool().numpy()

    # 15% BERT masking
    selected_count = mask.sum()
    threshold = max(0.0,
                    ((0.15 * total_count - selected_count) / (total_count - selected_count)
                     if total_count - selected_count else 0.0))
    mask |= np.random.rand(*bert_tokens.shape) < threshold

    mask[bert_tokens < mask_token_id] = False
    labels = -1 * np.ones(bert_tokens.shape, dtype=int)
    # Set labels for masked tokens
    labels[mask] = bert_tokens.numpy()[mask]

    bert_tokens_masked = np.copy(bert_tokens)
    # Set 80% of selected tokens to [MASK]
    # Leave 10% unchanged
    set_mask = mask & (np.random.rand(*bert_tokens.shape) < 0.90)
    bert_tokens_masked[set_mask] = mask_token_id

    # Set 10% to a random token
    # aka. Set 1 / 9 in set_mask to random token
    set_random = set_mask & (np.random.rand(*bert_tokens.shape) < 1. / 9)
    bert_tokens_masked[set_random] = np.random.randint(
        mask_token_id + 1, 29645, set_random.sum()
    )
    return bert_tokens_masked, labels
