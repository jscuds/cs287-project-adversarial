import pickle
from datasets import load_dataset
import numpy as np
from numpy import random
import scipy.special
import pandas as pd


def load_pickle(path):
    try:
        with open(path,'rb') as f:
            return pickle.load(f)
    except:
        print(f'Load pickle error on {f}')

def write_pickle(path, d):
    try:
        with open(path,'wb') as f:
            return pickle.dump(d, f, protocol = pickle.HIGHEST_PROTOCOL)
    except:
        print(f'Write pickle error on {f}')

swap_dict = load_pickle('complete_swap_word.pickle')
rotten_datasets = load_dataset('rotten_tomatoes')

def select_neighbor_word(swap_dict, word, k = 5, sel_type = 'weighted'):
    """Selects a neighboring word for perturbation from dictionary swap_dict"""
    try:
        swap_dict[word]
    except:
        return word
    k = min(k, len(swap_dict[word]))
    words, weights = zip(*swap_dict[word][:k])
    if sel_type == 'random': # pure random
        return random.choice(words)
    elif sel_type == 'top': # take top 1 case
        return words[0]
    elif sel_type == 'weighted':
        weights = np.asarray(weights) / np.sum(weights)
        return np.random.choice(words, p=weights)
    elif sel_type == 'softmax':
        weights = scipy.special.softmax(weights)
        return np.random.choice(words, p=weights)
    else:
        raise NotImplementedError

text = []
pert_text = []
labels = []
for example in rotten_datasets['train']:
    text.append(example['text'])
    labels.append(example['label'])
    sentence = example['text'].split()
    word_idx = np.random.choice(len(sentence), 
                                size=round(len(sentence)*0.25), 
                                replace=False)
    for idx in word_idx:
        swap_word = select_neighbor_word(swap_dict, sentence[idx])
        sentence[idx] = swap_word
    
    pert_text.append(' '.join(sentence))

df = pd.DataFrame(np.array([text, pert_text, labels]).T, columns=['original_text',
                                                             'perturbed_text',
                                                             'label'])

df.to_parquet('parquet/rotten_tomatoes-PHASE3-train.parquet')
