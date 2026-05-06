"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import os
from collections import Counter
from types import SimpleNamespace

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from transformers import AutoTokenizer


class _IndexLookup(list):
    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        return super().__getitem__(int(index))


class _Vocab:
    def __init__(self, tokens):
        self.itos = _IndexLookup(tokens)
        self.stoi = {token: idx for idx, token in enumerate(tokens)}

    def __len__(self):
        return len(self.itos)


class _Field:
    def __init__(self, init_token, eos_token):
        self.init_token = init_token
        self.eos_token = eos_token
        self.vocab = None

    def build_vocab(self, token_lists, min_freq):
        counter = Counter(token for tokens in token_lists for token in tokens)
        tokens = ['<unk>', '<pad>']
        for token in (self.init_token, self.eos_token):
            if token not in tokens:
                tokens.append(token)
        tokens.extend(sorted(token for token, count in counter.items() if count >= min_freq and token not in tokens))
        self.vocab = _Vocab(tokens)


class _TranslationDataset(Dataset):
    def __init__(self, split, src_key, trg_key, src_encode, trg_encode):
        self.split = split
        self.src_key = src_key
        self.trg_key = trg_key
        self.src_encode = src_encode
        self.trg_encode = trg_encode

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        row = self.split[idx]
        return {'src': self.src_encode(row[self.src_key]), 'trg': self.trg_encode(row[self.trg_key])}


class _Iterator:
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class DataLoader:
    source = None
    target = None

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        self.src_key, self.trg_key = ext[0][1:], ext[1][1:]
        self.tokenizers = {}
        self.local_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'datasets', 'multi30k'))
        self.cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'datasets')
        self.en_tokenizer = self.tokenizers.setdefault('.en', AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True))
        self.de_tokenizer = self.tokenizers.setdefault('.de', AutoTokenizer.from_pretrained('bert-base-multilingual-cased', use_fast=True))
        print('dataset initializing start')

    def _tokenize(self, ext, text):
        if ext == '.en':
            return self.en_tokenizer.tokenize(text.lower())
        elif ext == '.de':
            return self.de_tokenizer.tokenize(text.lower())
        else:
            raise ValueError(f'unknown ext: {ext}')

    def _encode(self, field, ext, text):
        stoi = field.vocab.stoi
        ids = [stoi[field.init_token]]
        ids.extend(stoi.get(token, stoi['<unk>']) for token in self._tokenize(ext, text))
        ids.append(stoi[field.eos_token])
        return ids

    def _collate(self, batch, device):
        src = pad_sequence([torch.tensor(item['src']) for item in batch], batch_first=True, padding_value=self.source.vocab.stoi['<pad>'])
        trg = pad_sequence([torch.tensor(item['trg']) for item in batch], batch_first=True, padding_value=self.target.vocab.stoi['<pad>'])
        return SimpleNamespace(src=src.to(device), trg=trg.to(device))
    
    def make_dataset(self):
        self.source = _Field(self.init_token, self.eos_token)
        self.target = _Field(self.init_token, self.eos_token)

        local_files = {
            'train': os.path.join(self.local_data_dir, 'train.jsonl'),
            'validation': os.path.join(self.local_data_dir, 'validation.jsonl'),
            'test': os.path.join(self.local_data_dir, 'test.jsonl'),
        }

        if all(os.path.exists(path) for path in local_files.values()):
            print(f'loading dataset from local files: {self.local_data_dir}')
            dataset = load_dataset('json', data_files=local_files, cache_dir=self.cache_dir)
        else:
            print(f'local dataset not found, loading from hub with cache: {self.cache_dir}')
            print('if this is the first run, downloading may take a while...')
            dataset = load_dataset('bentrevett/multi30k', cache_dir=self.cache_dir)

        print(f"dataset loading done: train={len(dataset['train'])}, valid={len(dataset['validation'])}, test={len(dataset['test'])}")
        return dataset['train'], dataset['validation'], dataset['test']

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab((self._tokenize(self.ext[0], row[self.src_key]) for row in train_data), min_freq)
        self.target.build_vocab((self._tokenize(self.ext[1], row[self.trg_key]) for row in train_data), min_freq)
        print(f'vocab building done: src={len(self.source.vocab)}, trg={len(self.target.vocab)}')
    
    def make_iter(self, train, validate, test, batch_size, device):
        def wrap(split):
            dataset = _TranslationDataset(split, self.src_key, self.trg_key, lambda x: self._encode(self.source, self.ext[0], x), lambda x: self._encode(self.target, self.ext[1], x))
            return _Iterator(TorchDataLoader(dataset, batch_size=batch_size, shuffle=split is train, collate_fn=lambda batch: self._collate(batch, device)))

        print('dataset initializing done')
        return wrap(train), wrap(validate), wrap(test)
