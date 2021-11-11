import copy
import os
import pickle
import time

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import BertTokenizer

from model.transformer_decoder import TransformerDecoder, Generator, TransformerDecoderState
from train.misc import get_model_path, get_masked_inputs_and_labels
from train.model import build_absa_model
from train.model import SupConLoss
from train.optimizer import build_optim, build_optim_bert


class ABSADataset(Dataset):
    def __init__(self, path):
        super(ABSADataset, self).__init__()
        data = pickle.load(open(path, 'rb'))
        self.bert_tokens = [torch.LongTensor(bert_token) for bert_token in data['bert_tokens']]
        self.aspect_masks = [torch.LongTensor(bert_mask) for bert_mask in data['aspect_masks']]
        self.raw_texts = data['raw_texts']
        self.raw_nested_aspect_terms = data['raw_nested_aspect_terms']
        self.labels = torch.LongTensor(data['labels'])
        self.len = len(data['labels'])

    def __getitem__(self, index):
        return (self.bert_tokens[index],
                self.aspect_masks[index],
                self.labels[index],
                self.raw_texts[index],
                self.raw_nested_aspect_terms[index])

    def __len__(self):
        return self.len


def collate_fn(batch):
    bert_tokens, aspect_masks, labels, raw_texts, raw_nested_aspect_terms = zip(*batch)

    bert_masks = pad_sequence([torch.ones(tokens.shape) for tokens in bert_tokens], batch_first=True)
    bert_tokens = pad_sequence(bert_tokens, batch_first=True)
    aspect_masks = pad_sequence(aspect_masks, batch_first=True)
    labels = torch.stack(labels)

    bert_tokens_masked, masked_labels = get_masked_inputs_and_labels(bert_tokens, aspect_masks)
    bert_tokens_masked = torch.LongTensor(bert_tokens_masked)
    masked_labels = torch.LongTensor(masked_labels)

    return (bert_tokens_masked,
            masked_labels,
            bert_tokens,
            bert_masks,
            aspect_masks,
            labels,
            raw_texts,
            raw_nested_aspect_terms)


def has_opposite_labels(labels):
    return not (labels.sum().item() <= 1 or (1 - labels).sum().item() <= 1)


def set_parameter_linear(p):
    if p.dim() > 1:
        xavier_uniform_(p)
    else:
        p.data.zero_()


def set_parameter_tf(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def fp16_multi_pretrain(config):
    data_path = config['data_path']
    pretrain_path = os.path.join(data_path, config['pretrain_file'])
    pretrain_dataset = ABSADataset(pretrain_path)
    sampler = DistributedSampler(pretrain_dataset, num_replicas=dist.get_world_size(), rank=config['local_rank'])
    global_rank = dist.get_rank()
    print("Card {} start training".format(global_rank))
    pretrain_loader = DataLoader(pretrain_dataset,
                                 batch_size=config['batch_size'],
                                 shuffle=False,
                                 sampler=sampler,
                                 collate_fn=collate_fn)
    model = build_absa_model(config).cuda()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tgt_embeddings = nn.Embedding(len(tokenizer.vocab), model.config.hidden_size, padding_idx=0)
    decoder = TransformerDecoder(
        config['decoder_layers'],
        config['decoder_hidden'],
        heads=config['decoder_heads'],
        d_ff=config['decoder_ff'],
        dropout=config['decoder_dropout'],
        embeddings=tgt_embeddings
    ).cuda()
    generator = Generator(len(tokenizer.vocab), config['decoder_hidden'], tokenizer.vocab['[PAD]']).cuda()
    generator.linear.weight = decoder.embeddings.weight


    for p in decoder.modules():
        set_parameter_tf(p)
    for p in generator.parameters():
        set_parameter_tf(p)

    if "share_emb" in config:
        if 'bert' in config['mode']:
            tgt_embeddings = nn.Embedding(len(tokenizer.vocab), model.config.hidden_size, padding_idx=0)
            tgt_embeddings.weight = copy.deepcopy(model.bert.embeddings.word_embeddings.weight)
        decoder.embeddings = tgt_embeddings
        generator.linear.weight = decoder.embeddings.weight
    if 'cache' in config:
        dist.barrier()
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % config['local_rank']}
        model_dict = torch.load(config['cache'], map_location=map_location)
        decoder_dict = torch.load(config['decoder_cache'], map_location=map_location)
        generator_dict = torch.load(config['generator_cache'], map_location=map_location)
        model.load_state_dict(model_dict)
        decoder.load_state_dict(decoder_dict)
        generator.load_state_dict(generator_dict)

    model = DDP(model, device_ids=[config['local_rank']], output_device=config['local_rank'], find_unused_parameters=True)
    decoder = DDP(decoder, device_ids=[config['local_rank']], output_device=config['local_rank'], find_unused_parameters=True)
    generator = DDP(generator, device_ids=[config['local_rank']], output_device=config['local_rank'], find_unused_parameters=True)

    if global_rank == 0:
        model_path = get_model_path(config['model'], config['model_path'])
    optimizer_bert = build_optim_bert(config, model)
    optimizer_decoder = build_optim(config, decoder)
    optimizer_generator = build_optim(config, generator)
    optim = [optimizer_bert, optimizer_decoder, optimizer_generator]
    scaler = torch.cuda.amp.GradScaler()

    similar_criterion = SupConLoss()
    reconstruction_criterion = nn.NLLLoss(ignore_index=tokenizer.vocab['[PAD]'], reduction='mean')

    global_step = 0
    for epoch in range(config['epoch']):
        sampler.set_epoch(epoch)
        avg_mlm_loss = 0.
        avg_similar_loss = 0.
        avg_reconstruction_loss = 0.
        avg_loss = 0.
        current_step = 0
        start = time.time()
        for idx, batch in enumerate(pretrain_loader):
            try:
                with torch.cuda.amp.autocast():
                    model.train()
                    decoder.train()
                    generator.train()
                    model.zero_grad()
                    decoder.zero_grad()
                    generator.zero_grad()
                    current_step += 1
                    global_step += 1
                    bert_tokens_masked, masked_labels, bert_tokens, bert_masks, aspect_masks, labels, raw_texts, raw_nested_aspect_terms = batch
                    bert_tokens_masked = bert_tokens_masked.cuda()
                    masked_labels = masked_labels.cuda()
                    bert_tokens = bert_tokens.cuda()
                    aspect_masks = aspect_masks.cuda()
                    labels = labels.cuda()
                    if has_opposite_labels(labels):
                        cls_hidden, hidden_state, masked_lm_loss = model(
                            input_ids=bert_tokens_masked,
                            attention_mask=bert_masks,
                            aspect_mask=aspect_masks,
                            labels=masked_labels,
                            return_dict=True,
                            multi_card=True,
                            has_opposite_labels=True
                        )
                    else:
                        hidden_state, masked_lm_loss = model(
                            input_ids=bert_tokens_masked,
                            attention_mask=bert_masks,
                            aspect_mask=aspect_masks,
                            labels=masked_labels,
                            return_dict=True,
                            multi_card=True,
                            has_opposite_labels=False
                        )
                    decode_context = decoder(bert_tokens_masked[:, :-1], hidden_state,
                                             TransformerDecoderState(bert_tokens_masked))
                    reconstruction_text = generator(decode_context.view(-1, decode_context.size(2)))

                    reconstruction_loss = reconstruction_criterion(reconstruction_text,
                                                                   bert_tokens[:, 1:].reshape(-1))
                    if not has_opposite_labels(labels):
                        loss = config['lambda_map'] * masked_lm_loss + \
                               config['lambda_rr'] * reconstruction_loss
                    else:
                        normed_cls_hidden = F.normalize(cls_hidden, dim=-1)
                        similar_loss = similar_criterion(normed_cls_hidden.unsqueeze(1), labels=labels)
                        loss = config['lambda_map'] * masked_lm_loss + \
                               config['lambda_scl'] * similar_loss + \
                               config['lambda_rr'] * reconstruction_loss
                        avg_similar_loss += config['lambda_scl'] * similar_loss.item()

                    scaler.scale(loss).backward()

                    for o in optim:
                        o.step()
                        scaler.step(o.optimizer)
                    scaler.update()

                avg_mlm_loss += config['lambda_map'] * masked_lm_loss.item()
                avg_reconstruction_loss += config['lambda_rr'] * reconstruction_loss.item()
                avg_loss += loss.item()
            except RuntimeError as e:
                print(e)
                print("Batch Skipped!")
                continue

            if idx % config['report_frequency'] == 0 and idx != 0 and global_rank == 0:
                mlm_loss = avg_mlm_loss / current_step
                similar_loss = avg_similar_loss / current_step
                rec_loss = avg_reconstruction_loss / current_step
                pretrain_loss = avg_loss / current_step
                avg_loss = avg_mlm_loss = avg_similar_loss = avg_reconstruction_loss = 0.
                current_step = 0
                print("PRETRAIN [Epoch {:2d}] [step {:3d}]".format(epoch, idx),
                      "MAP loss: {:.4f}, SCL loss: {:.4f}, RR loss: {:.4f}, pretrain loss: {:.4f}"
                      .format(mlm_loss, similar_loss, rec_loss, pretrain_loss))
            if global_step % config['save_frequency'] == 0 and global_step != 0 and global_rank == 0:
                model_file = "epoch_{}_step_{}.pt".format(epoch, global_step)
                torch.save(model.module.state_dict(), os.path.join(model_path, model_file))
                decoder_file = "epoch_{}_step_{}_decoder.pt".format(epoch, global_step)
                torch.save(decoder.module.state_dict(), os.path.join(model_path, decoder_file))
                generator_file = "epoch_{}_step_{}_generator.pt".format(epoch, global_step)
                torch.save(generator.module.state_dict(), os.path.join(model_path, generator_file))
                print("Model saved: {}".format(model_file))
        end = time.time()
        print("[Epoch {:2d}] complete in {:.2f} seconds".format(epoch, end - start))
