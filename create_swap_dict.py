import transformers
from torch.nn import Module, LSTM, Linear, init
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset, load_metric
import datasets
import random
import pandas as pd
from torch.optim import Adam
import gensim
from numpy import random
import numpy as np
from tqdm import tqdm
import pickle
import gensim.downloader as api


dataset = load_dataset('rotten_tomatoes')
wv = api.load('word2vec-google-news-300')

def lowercase(text):
        return text.lower()

def remove_punc(text, punc):
    exclude = set(punc+"'")
    text = "".join([(ch if ch not in exclude else "") for ch in text])
    return text

def split_on_whitespace(text):
    return text.split()

def tokenize(text, punc):
    text = lowercase(text)
    text = remove_punc(text, punc)
    return split_on_whitespace(text)

def add_swap_words(words, swap_dict):
    for word in words:
        if word not in swap_dict:
            try:
                swap_dict[word] = wv.most_similar(positive=[word], topn=10)
            except KeyError:
                pass
    return swap_dict

punc = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~—“”'
swap_dict = {}
for example in dataset['validation']:
    words = tokenize(example['text'], punc)
    swap_dict = add_swap_words(words, swap_dict)

for example in dataset['test']:
    words = tokenize(example['text'], punc)
    swap_dict = add_swap_words(words, swap_dict)

for example in dataset['train']:
    words = tokenize(example['text'], punc)
    swap_dict = add_swap_words(words, swap_dict)

dataset_imbd = load_dataset('imdb')

for example in dataset_imbd['test']:
    words = tokenize(example['text'], punc)
    swap_dict = add_swap_words(words, swap_dict)

for example in dataset_imbd['train']:
    words = tokenize(example['text'], punc)
    swap_dict = add_swap_words(words, swap_dict)

dataset_snli = load_dataset('snli')

for example in dataset_snli['validation']:
    words = tokenize(example['premise'], punc)
    swap_dict = add_swap_words(words, swap_dict)
    words = tokenize(example['hypothesis'], punc)
    swap_dict = add_swap_words(words, swap_dict)

for example in dataset_snli['train']:
    words = tokenize(example['premise'], punc)
    swap_dict = add_swap_words(words, swap_dict)
    words = tokenize(example['hypothesis'], punc)
    swap_dict = add_swap_words(words, swap_dict)

for example in dataset_snli['test']:
    words = tokenize(example['premise'], punc)
    swap_dict = add_swap_words(words, swap_dict)
    words = tokenize(example['hypothesis'], punc)
    swap_dict = add_swap_words(words, swap_dict)

with open('complete_swap_word.pickle', 'wb') as handle:
    pickle.dump(swap_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
