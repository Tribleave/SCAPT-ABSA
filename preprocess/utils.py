import json
import pickle
from lxml import etree
from transformers import BertTokenizer
from tqdm import tqdm


def parse_xml(path, lowercase=False, remove_list=None):
    if remove_list is None:
        remove_list = []
    dataset = []
    with open(path, 'rb') as f:
        root = etree.fromstring(f.read())
        for sentence in root:
            index = sentence.get('id')
            sent = sentence.find('text').text
            if lowercase:
                sent = sent.lower()
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            accept_terms = []
            for term in terms:
                aspect = term.attrib['term']
                sentiment = term.attrib['polarity']
                implicit = term.attrib.get('implicit_sentiment', '') == "True"
                if sentiment in remove_list:
                    continue
                left_index = int(term.attrib['from'])
                right_index = int(term.attrib['to'])
                accept_terms.append({
                    'aspect_term': aspect,
                    'sentiment': sentiment,
                    'implicit': implicit,
                    'left_index': left_index,
                    'right_index': right_index,
                })
            if accept_terms:
                dataset.append({
                    'id': index,
                    'text': sent,
                    'aspect_terms': accept_terms,
                })
    return dataset


def parse_json(path, lowercase=False, remove_list=None):
    if remove_list is None:
        remove_list = []
    dataset = []
    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line)
            tokens = line['tokens']
            if len(tokens) <= 8:
                continue
            sent = line['text']
            if lowercase:
                sent = sent.lower()

            accept_terms = []
            for term in line['aspect_terms']:
                aspect = term['aspect_term']
                if lowercase:
                    aspect = aspect.lower()
                sentiment = term['sentiment']
                if sentiment in remove_list:
                    continue
                left_index = term['left_index']
                right_index = term['right_index']
                assert aspect == sent[left_index: right_index]
                accept_terms.append({
                    'aspect_term': aspect,
                    'sentiment': sentiment,
                    'left_index': left_index,
                    'right_index': right_index,
                })
            if accept_terms:
                dataset.append({
                    'text': sent,
                    'aspect_terms': accept_terms,
                })
    return dataset


def parse_term(path, lowercase=False, remove_list=None):
    file_type = path.split('.')[-1]
    if file_type == 'xml':
        return parse_xml(path, lowercase, remove_list)
    if file_type == 'json':
        return parse_json(path, lowercase, remove_list)
    if file_type == 'pkl':
        return pickle.load(open(path, 'rb'))
    raise TypeError(f"Not supported file type: {file_type}")


def preprocess_pretrain(dataset, max_token_size=512):
    if not hasattr(preprocess_pretrain, 'tokenizer'):
        preprocess_pretrain.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')
    sentiment_dict = {1: 0, 5: 1}
    raw_texts = []
    raw_nested_aspect_terms = []
    bert_tokens = []
    aspect_masks = []
    labels = []
    for data in tqdm(dataset):
        if data['aspect_terms'][0]['sentiment'] not in sentiment_dict:
            continue
        token = preprocess_pretrain.tokenizer.tokenize(data['text'])
        label = sentiment_dict.get(data['aspect_terms'][0]['sentiment'])
        aspect_terms = [aspect['aspect_term']
                        for aspect in data['aspect_terms']]
        bert_token = preprocess_pretrain.tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + token + ['[SEP]'])
        if len(bert_token) >= max_token_size:
            continue
        aspect_mask = [0] * len(bert_token)
        for aspect in data['aspect_terms']:
            left_idx = len(preprocess_pretrain.tokenizer.tokenize(
                    data['text'][:aspect['left_index']])) + 1
            length = len(preprocess_pretrain.tokenizer.tokenize(
                    data['text'][aspect['left_index']: aspect['right_index']]))
            aspect_mask[left_idx: left_idx + length] = [1] * length
        raw_texts.append(data['text'])
        raw_nested_aspect_terms.append(aspect_terms)
        bert_tokens.append(bert_token)
        aspect_masks.append(aspect_mask)
        labels.append(label)
    dataset = {
        'raw_texts': raw_texts,
        'raw_nested_aspect_terms': raw_nested_aspect_terms,
        'bert_tokens': bert_tokens,
        'aspect_masks': aspect_masks,
        'labels': labels
    }
    return dataset


def preprocess_finetune(dataset):
    if not hasattr(preprocess_finetune, 'tokenizer'):
        preprocess_finetune.tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')
    sentiment_dict = {
        'positive': 0,
        'negative': 1,
        'neutral': 2,
        'conflict': 3
    }
    raw_texts = []
    raw_aspect_terms = []
    bert_tokens = []
    aspect_masks = []
    implicits = []
    labels = []
    for data in tqdm(dataset):
        token = preprocess_finetune.tokenizer.tokenize(data['text'])
        for aspect in data['aspect_terms']:
            bert_token = preprocess_finetune.tokenizer.convert_tokens_to_ids(
                ['[CLS]'] + token + ['[SEP]']
            )
            left_idx = len(preprocess_finetune.tokenizer.tokenize(
                data['text'][:aspect['left_index']])) + 1
            length = len(preprocess_finetune.tokenizer.tokenize(
                data['text'][aspect['left_index']: aspect['right_index']]))
            aspect_mask = [0] * len(bert_token)
            aspect_mask[left_idx: left_idx + length] = [1] * length
            assert len(aspect_mask) == len(bert_token)
            if len(bert_token) >= 512:
                continue
            raw_texts.append(data['text'])
            raw_aspect_terms.append(aspect['aspect_term'])
            bert_tokens.append(bert_token)
            aspect_masks.append(aspect_mask)
            implicits.append(aspect['implicit'])
            labels.append(sentiment_dict[aspect['sentiment']])

    dataset = {
        'raw_texts': raw_texts,
        'raw_aspect_terms': raw_aspect_terms,
        'bert_tokens': bert_tokens,
        'aspect_masks': aspect_masks,
        'implicits': implicits,
        'labels': labels,
    }
    return dataset
