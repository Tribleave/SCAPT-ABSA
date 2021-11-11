import os
import pickle

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics

from train.misc import get_model_path, prepare_dataset
from train.model import build_absa_model, build_optimizer, LabelSmoothLoss


class ABSADataset(Dataset):
    def __init__(self, path):
        super(ABSADataset, self).__init__()
        data = pickle.load(open(path, 'rb'))
        self.raw_texts = data['raw_texts']
        self.raw_aspect_terms = data['raw_aspect_terms']
        self.bert_tokens = [torch.LongTensor(
            token) for token in data['bert_tokens']]
        self.aspect_mask = [torch.LongTensor(
            mask) for mask in data['aspect_masks']]
        self.implicits = torch.LongTensor(data['implicits'])
        self.labels = torch.LongTensor(data['labels'])
        self.len = len(data['labels'])

    def __getitem__(self, index):
        return (self.bert_tokens[index],
                self.aspect_mask[index],
                self.labels[index],
                self.raw_texts[index],
                self.raw_aspect_terms[index],
                self.implicits[index])

    def __len__(self):
        return self.len


def collate_fn(batch):
    bert_tokens, aspect_masks, labels, raw_texts, raw_aspect_terms, implicits = zip(*batch)

    bert_masks = pad_sequence([torch.ones(tokens.shape) for tokens in bert_tokens], batch_first=True)
    bert_tokens = pad_sequence(bert_tokens, batch_first=True)
    aspect_masks = pad_sequence(aspect_masks, batch_first=True)
    labels = torch.stack(labels)
    implicits = torch.stack(implicits)

    return (bert_tokens,
            bert_masks,
            aspect_masks,
            labels,
            raw_texts,
            raw_aspect_terms,
            implicits)


sentiment_dict = {
    0: 'positive',
    1: 'negative',
    2: 'neutral',
    3: 'conflict',
}


def evaluate(config, model, data_loader, criterion=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    total_samples, correct_samples = 0, 0
    total_explicit, correct_explicit = 0, 0
    total_implicit, correct_implicit = 0, 0
    total_loss = 0
    model.eval()
    labels_all, preds_all = None, None

    with torch.no_grad():
        for batch in data_loader:
            bert_tokens, bert_masks, aspect_masks, labels, raw_texts, raw_aspect_terms, implicits = batch
            bert_tokens = bert_tokens.to(device)
            bert_masks = bert_masks.to(device)
            aspect_masks = aspect_masks.to(device)
            labels = labels.to(device)
            implicits = implicits.to(device)

            output = model(
                input_ids=bert_tokens,
                attention_mask=bert_masks,
                aspect_mask=aspect_masks,
                output_attentions=True,
                return_dict=True
            )
            logits = output.sentiment
            loss = criterion(logits, labels) if criterion else 0
            batch_size = bert_tokens.size(0)
            if loss > 0:
                total_loss += batch_size * loss.item()
            total_samples += batch_size
            pred = logits.argmax(dim=-1)
            if labels_all is None:
                labels_all = labels
                preds_all = pred
            else:
                labels_all = torch.cat((labels_all, labels), dim=0)
                preds_all = torch.cat((preds_all, pred), dim=0)
            correct_samples += (pred == labels).long().sum().item()
            total_explicit += (1 - implicits).long().sum().item()
            correct_explicit += ((1 - implicits) & (pred == labels)).long().sum().item()
            total_implicit += implicits.long().sum().item()
            correct_implicit += (implicits & (pred == labels)).long().sum().item()

    accuracy = correct_samples / total_samples
    f1 = metrics.f1_score(y_true=labels_all.cpu(),
                          y_pred=preds_all.cpu(),
                          labels=[0, 1, 2], average='macro')
    average_loss = total_loss / total_samples if criterion else 0.0
    explicit_acc = correct_explicit / total_explicit if total_explicit else 0.0
    implicit_acc = correct_implicit / total_implicit if total_implicit else 0.0
    return accuracy, f1, average_loss, explicit_acc, implicit_acc


def aspect_finetune(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, dev_loader, test_loader = prepare_dataset(config,
                                                            absa_dataset=ABSADataset,
                                                            collate_fn=collate_fn)
    model = build_absa_model(config).to(device)
    state_dict = config.get('checkpoint', None)
    if isinstance(state_dict, str):
        model.load_state_dict(torch.load(state_dict))
    elif state_dict:
        model.load_state_dict(state_dict)

    if train_loader is None:
        val_acc, val_f1, _, explicit_acc, implicit_acc = evaluate(config, model, dev_loader)
        print("valid f1: {:.4f}, valid acc: {:.4f}, explicit acc: {:.4f}, implicits acc: {:.4f}".format(val_f1, val_acc, explicit_acc, implicit_acc))
        return

    model_path = (get_model_path(config['model'], config['model_path'])
                  if 'model_path' in config else None)
    optimizer = build_optimizer(config, model)
    scheduler = get_linear_schedule_with_warmup(optimizer, config['warm_up'], config['epoch'] * len(train_loader))
    classification_criterion = LabelSmoothLoss(smoothing=config['label_smooth'])

    max_val_accuracy = max_val_f1 = 0.
    min_val_loss = float('inf')
    max_explicit_acc = 0.
    max_implicit_acc = 0.
    global_step = 0

    for epoch in range(config['epoch']):
        total_loss = 0.
        total_samples = 0
        correct_samples = 0
        for idx, batch in enumerate(train_loader):
            global_step += 1
            model.train()

            bert_tokens, bert_masks, aspect_masks, labels, raw_texts, raw_aspect_terms, implicits = batch
            bert_tokens = bert_tokens.to(device)
            bert_masks = bert_masks.to(device)
            aspect_masks = aspect_masks.to(device)
            labels = labels.to(device)

            output = model(
                input_ids=bert_tokens,
                attention_mask=bert_masks,
                aspect_mask=aspect_masks,
                output_attentions=True,
                return_dict=True
            )
            logits = output.sentiment
            classification_loss = classification_criterion(logits, labels)
            classification_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            batch_size = bert_tokens.size(0)
            total_loss += batch_size * classification_loss.item()
            total_samples += batch_size
            pred = logits.argmax(dim=-1)
            correct_samples += (pred == labels).long().sum().item()
            valid_frequency = config['valid_frequency'] if 'valid_frequency' in config else len(train_loader) - 1

            if idx % valid_frequency == 0 and idx != 0:
                train_loss = total_loss / total_samples
                train_accuracy = correct_samples / total_samples
                total_loss = total_samples = correct_samples = 0
                val_acc, val_f1, val_loss, explicit_acc, implicit_acc = evaluate(config, model, dev_loader, classification_criterion)
                print("[Epoch {:2d}] [step {:3d}]".format(epoch, idx),
                      "train loss: {:.4f}, train acc: {:.4f}, ".format(train_loss, train_accuracy),
                      "valid loss: {:.4f}, valid f1: {:.4f}, valid acc: {:.4f}, ".format(val_loss, val_f1, val_acc),
                      "valid explicit acc: {:.4f}, valid implicit acc: {:.4f} ".format(explicit_acc, implicit_acc))

                max_val_f1 = max(max_val_f1, val_f1)
                if val_acc > max_val_accuracy:
                    max_val_accuracy = val_acc
                    min_val_loss = val_loss
                    max_explicit_acc = explicit_acc
                    max_implicit_acc = implicit_acc

                    model_file = "epoch_{}_step_{}_acc_{:.4f}_f1_{:.4f}_loss_{:.4f}.pt".format(
                        epoch, global_step, val_acc, val_f1, val_loss)
                    torch.save(model.state_dict(), os.path.join(model_path, model_file))

    print("Max valid accuracy: {:.4f}, valid f1: {:.4f}, ".format(max_val_accuracy, max_val_f1),
          "explicit acc: {:.4f}, implicit acc: {:.4f}".format(max_explicit_acc, max_implicit_acc))
    return max_val_accuracy, max_val_f1, min_val_loss
