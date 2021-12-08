import transformers
from torch.nn import Module, LSTM, Linear, init
from torch.nn.functional import sigmoid
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from datasets import load_dataset, load_metric
import datasets
import random
import pandas as pd
from torch.optim import Adam
import gensim
from numpy import random
import numpy as np
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss
import pickle5 as pickle
from copy import deepcopy
import gensim.downloader as api

PATH_TO_SWAPDICT = ''

# First, we get our dataset and embeddings

dataset = load_dataset('rotten_tomatoes')
wv = api.load('word2vec-google-news-300')


## Dataset Prep
word_to_idx = {word: i for i, word in enumerate(wv.index2word)}
word_to_idx['[PAD]'] = -1

idx_to_word = {i: word for i, word in enumerate(wv.index2word)}
idx_to_word[-1] = '[PAD]'

"""Now, we can prepare our dataset and get the Word2Vec embeddings."""

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

class MyDataset(Dataset):
    
    def __init__(self, dataset, split, punc):
        self.tokenized_texts = []
        self.labels = []

        for example in dataset[split]:
          tokens = []
          raw_words = []
          tok = tokenize(example['text'], punc)
          for word in tok:
            try:
              tokens.append(word_to_idx[word])
            except KeyError:
              pass
          if tokens:
            self.tokenized_texts.append(torch.tensor(tokens))
          self.labels.append(example['label'])        
    
    def __len__(self):
        return len(self.tokenized_texts)
        
    def __getitem__(self, idx):
        return self.tokenized_texts[idx], torch.tensor(self.labels[idx])

punc = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~—“”'
train_ds = MyDataset(dataset, 'train', punc)
val_ds = MyDataset(dataset, 'validation', punc)
test_ds = MyDataset(dataset, 'test', punc)

def w2v_pad_collate_classifier(batch):
    (tokens, y) = zip(*batch)
    xx_pad = pad_sequence(tokens, batch_first=True, padding_value=-1)
    y_stack = torch.stack(y, dim=0)
    return xx_pad, y_stack

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=w2v_pad_collate_classifier) 
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, drop_last=False, collate_fn=w2v_pad_collate_classifier) 
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, drop_last=False, collate_fn=w2v_pad_collate_classifier)

"""### Word selection criterion"""

import scipy.special
def select_neighbor_word(swap_dict, word, k = 5, sel_type = 'random'):
    """Selects a neighboring word for perturbation from dictionary swap_dict"""
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

"""### Loading Swap Dicitonary"""

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

complete_swap_ds = load_pickle(PATH_TO_SWAPDICT)

"""# Model code"""

class RottenModel(Module):
  def __init__(self, hidden_size, wv, swap_dict):
    super().__init__()
    self.hidden_size = hidden_size
    self.wv = wv
    self.lstm = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
    self.linear = Linear(self.hidden_size*2, 1)
    self.swap_dict = swap_dict

  def forward(self, input_seqs, perturb_mask=None):
    input_tokens = []
    lengths = []
    for seq in input_seqs:
      length = 0
      for idx in seq:
        if idx == -1:
            break
        length += 1
      lengths.append(length)

    if not perturb_mask is None:
      for seq, mask in zip(input_seqs, perturb_mask):
        tokenized = []
        for idx, indicator in zip(seq, mask):
          if idx != -1:
            if indicator == 0:
              tokenized.append(torch.tensor(self.wv[idx_to_word[int(idx)]]))
            else:
              tokenized.append(torch.tensor(self.wv[select_neighbor_word(self.swap_dict, idx_to_word[int(idx)],
                                                                         sel_type='softmax')]))
          else:
            tokenized.append(torch.tensor(self.wv['pad']))
        input_tokens.append(torch.stack(tokenized))

    else:
      for seq in input_seqs:
        tokenized = []
        for idx in seq:
          if idx != -1:
            tokenized.append(torch.tensor(self.wv[idx_to_word[int(idx)]]))
          else:
            tokenized.append(torch.tensor(self.wv['pad']))
        input_tokens.append(torch.stack(tokenized))

    all_tokens = torch.stack(input_tokens)
    packed_input_seqs = pack_padded_sequence(all_tokens, lengths, batch_first=True, enforce_sorted=False)
    out, (hidden_cell, context_cell) = self.lstm(packed_input_seqs.to('cuda'))
    cells = torch.cat([hidden_cell, context_cell], axis=2)
    output = self.linear(cells)
    return output[0]

"""## Training

### Base model
"""

# TRAINING LOOP
def compute_n_correct(preds, targets):
    return torch.sum(torch.round(sigmoid(preds.squeeze())) == targets).cpu().item()

hidden_size = 300
# model = model = RottenModel(hidden_size, wv, deepcopy(swap_ds))
model = RottenModel(hidden_size, wv, deepcopy(complete_swap_ds))
model = model.to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
loss_fn = BCEWithLogitsLoss()

train_losses = []
test_losses = []
train_accs = []
test_accs = []

n_epochs = 10
for epoch in range(n_epochs):
    running_loss = 0
    n_correct = []
    total_sz = 0
    for batch, targets in tqdm(train_dl, leave=False):
        preds = model(batch.to('cuda')).logits
        loss = loss_fn(preds.squeeze(), targets.to('cuda').float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.cpu().item()
        n_correct.append(compute_n_correct(preds, targets.to('cuda')))
        total_sz += len(batch)
        
    train_losses.append(running_loss / len(train_ds))
    train_accs.append(sum(n_correct)/total_sz)
    print("="*20)
    print(f"Epoch {epoch+1}/{n_epochs} Train Loss: {train_losses[-1]}")
    print(f"Epoch {epoch+1}/{n_epochs} Train Accuracy: {train_accs[-1]}" )
    
    running_loss = 0
    n_correct = []
    total_sz = 0
    with torch.no_grad():
        for batch, targets in tqdm(val_dl, leave=False):
            preds = model(batch.to('cuda'))
            loss = loss_fn(preds.squeeze(), targets.to('cuda').float())
            running_loss += loss.cpu().item()
            n_correct.append(compute_n_correct(preds, targets.to('cuda')))
            total_sz += len(batch)

    test_losses.append(running_loss / len(val_ds))
    test_accs.append(sum(n_correct)/total_sz)
    print(f"Epoch {epoch+1}/{n_epochs} Test Loss: {test_losses[-1]}")
    print(f"Epoch {epoch+1}/{n_epochs} Test Accuracy: {test_accs[-1]}" )

write_pickle(f'/content/base-model.pkl', model)

running_test_loss = 0
n_correct_test = []
total_test_sz = 0
model.eval()
with torch.no_grad():
    for batch, targets in tqdm(test_dl, leave=False):
        preds = model(batch.to('cuda'))
        loss = loss_fn(preds.squeeze(), targets.to('cuda').float())
        running_test_loss += loss.cpu().item()
        n_correct_test.append(compute_n_correct(preds, targets.to('cuda')))
        total_test_sz += len(batch)
    total_test_correct = sum(n_correct_test)
print(f"Test set accuracy: {total_test_correct/total_test_sz}")


### Perturbation trained model
#### Training Loop

# TRAINING LOOP
def compute_n_correct(preds, targets):
    return torch.sum(torch.round(sigmoid(preds.squeeze())) == targets).cpu().item()

hidden_size = 300
# perturb_model = RottenModel(hidden_size, wv, deepcopy(swap_ds))
perturb_model = RottenModel(hidden_size, wv, deepcopy(complete_swap_ds))
perturb_model = perturb_model.to('cuda')
optimizer = torch.optim.Adam(perturb_model.parameters(), lr=1e-3, weight_decay=0.001)
loss_fn = BCEWithLogitsLoss()

train_losses = []
test_losses = []
train_accs = []
test_accs = []

# sets % that a word is perturbed
# expected number of perturbed words = 10%
perturb_percent = 0.10

n_epochs = 10
for epoch in range(n_epochs):
    running_loss = 0
    n_correct = []
    total_sz = 0
    perturb_model.train()
    for batch, targets in tqdm(train_dl, leave=False):
        seq_len = len(batch[0]) # gets size of each input seq for mask creation
        perturb_mask = np.random.rand(len(batch), seq_len) # generate batch_sz * seq_len random matrix
        perturb_mask = (perturb_mask < perturb_percent).astype(int)

        preds = perturb_model(batch.to('cuda'), perturb_mask)
        loss = loss_fn(preds.squeeze(), targets.to('cuda').float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.cpu().item()
        n_correct.append(compute_n_correct(preds, targets.to('cuda')))
        total_sz += len(batch)
        
    train_losses.append(running_loss / len(train_ds))
    train_accs.append(sum(n_correct)/total_sz)
    print("="*20)
    print(f"Epoch {epoch+1}/{n_epochs} Train Loss: {train_losses[-1]}")
    print(f"Epoch {epoch+1}/{n_epochs} Train Accuracy: {train_accs[-1]}" )
    
    running_loss = 0
    n_correct = []
    total_sz = 0
    perturb_model.eval()
    with torch.no_grad():
        for batch, targets in tqdm(val_dl, leave=False):
            preds = perturb_model(batch.to('cuda'))
            loss = loss_fn(preds.squeeze(), targets.to('cuda').float())
            running_loss += loss.cpu().item()
            n_correct.append(compute_n_correct(preds, targets.to('cuda')))
            total_sz += len(batch)

    test_losses.append(running_loss / len(val_ds))
    test_accs.append(sum(n_correct)/total_sz)
    print(f"Epoch {epoch+1}/{n_epochs} Test Loss: {test_losses[-1]}")
    print(f"Epoch {epoch+1}/{n_epochs} Test Accuracy: {test_accs[-1]}" )

write_pickle("/content/perturbed-10.pkl", perturb_model)

running_test_loss = 0
n_correct_test = []
total_test_sz = 0
perturb_model.eval()
with torch.no_grad():
    for batch, targets in tqdm(test_dl, leave=False):
        preds = perturb_model(batch.to('cuda'))
        loss = loss_fn(preds.squeeze(), targets.to('cuda').float())
        running_test_loss += loss.cpu().item()
        n_correct_test.append(compute_n_correct(preds, targets.to('cuda')))
        total_test_sz += len(batch)
    total_test_correct = sum(n_correct_test)
print(f"Test set accuracy: {total_test_correct/total_test_sz}")

"""### Testing models on perturbed test set

Base model:
"""

running_test_loss = 0
n_correct_test = []
total_test_sz = 0
model.eval()
with torch.no_grad():
    for batch, targets in tqdm(test_dl, leave=False):
        seq_len = len(batch[0])
        perturb_mask = np.random.rand(len(batch), seq_len)
        perturb_mask = (perturb_mask < perturb_percent).astype(int)

        preds = model(batch.to('cuda'), perturb_mask, )
        loss = loss_fn(preds.squeeze(), targets.to('cuda').float())
        running_test_loss += loss.cpu().item()
        n_correct_test.append(compute_n_correct(preds, targets.to('cuda')))
        total_test_sz += len(batch)
    total_test_correct = sum(n_correct_test)
print(f"Test set accuracy: {total_test_correct/total_test_sz}")

running_test_loss = 0
n_correct_test = []
total_test_sz = 0
perturb_model.eval()
with torch.no_grad():
    for batch, targets in tqdm(test_dl, leave=False):
        seq_len = len(batch[0])
        perturb_mask = np.random.rand(len(batch), seq_len)
        perturb_mask = (perturb_mask < perturb_percent).astype(int)

        preds = perturb_model(batch.to('cuda'), perturb_mask)
        loss = loss_fn(preds.squeeze(), targets.to('cuda').float())
        running_test_loss += loss.cpu().item()
        n_correct_test.append(compute_n_correct(preds, targets.to('cuda')))
        total_test_sz += len(batch)
    total_test_correct = sum(n_correct_test)
print(f"Test set accuracy: {total_test_correct/total_test_sz}")

