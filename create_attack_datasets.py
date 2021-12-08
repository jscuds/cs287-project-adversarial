#!/usr/bin/env python3

##############################################################################
########################### I. GLOBALS AND IMPORTS ###########################
##############################################################################

CLUSTER = False #TODO change to FALSE if not using Cluster
DATASET = 'rotten_tomatoes'
MODEL_NAME = 'textattack/bert-base-uncased-rotten-tomatoes'
ATTACK =  'TextFoolerJin2019' #TextFoolerJin2019,  BAEGarg2019, PWWSRen2019 ### DeepWordBugGao2018 ALREADY DONE ####
SPLIT = 'train'
LR = 2e-05 
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 128
# model card for this model used was fine-tuned for 5 epochs with a batch size of 16, 
# a learning rate of 2e-05, and a maximum sequence length of 128
#https://huggingface.co/textattack/bert-base-uncased-imdb

# IMPORTS
print('***IMPORTING LIBRARIES***\n')
import os
if not os.path.exists(f'checkpoints-{DATASET}'):
	os.mkdir(f'checkpoints-{DATASET}')
# if CLUSTER: 
#     os.environ['TRANSFORMERS_CACHE'] = '$SCRATCH/protopapas_lab/Everyone/jscudder/huggingface/transformers'
#     os.environ['HF_DATASETS_CACHE'] = '$SCRATCH/protopapas_lab/Everyone/jscudder/huggingface/datasets'

from datasets import load_dataset, load_metric
import numpy as np
import pandas as pd
import re

import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from transformers import BertTokenizer, BertModel

from tqdm import tqdm # tqdm provides us a nice progress bar.

# need to use a different model for pretraining.
from transformers import BertForPreTraining, BertForSequenceClassification
from sklearn.metrics import f1_score
#from allennlp.training.metrics import F1MultiLabelMeasure, BooleanAccuracy

# USED https://textattack.readthedocs.io/en/latest/2notebook/Example_5_Explain_BERT.html
#      https://textattack.readthedocs.io/en/latest/api/attacker.html
#import tensorflow as tf
import textattack
from textattack.loggers import CSVLogger # tracks a dataframe for us.
from textattack.attack_results import SuccessfulAttackResult
from textattack import Attacker
from textattack import AttackArgs
from textattack.datasets import Dataset
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.models.wrappers import ModelWrapper

# attack recipes
from textattack.attack_recipes import TextFoolerJin2019
from textattack.attack_recipes import DeepWordBugGao2018
from textattack.attack_recipes import BAEGarg2019
from textattack.attack_recipes import PWWSRen2019

# device = cuda if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'\n***DEVICE is {device}***')
###########################################################################################
########################### II. DATA IMPORT AND MODEL LOAD/EVAL ###########################
###########################################################################################

# global dictionary for 4 attacks
ATTACK_OPTIONS = dict([('DeepWordBugGao2018', DeepWordBugGao2018), ('TextFoolerJin2019',TextFoolerJin2019), 
    ('BAEGarg2019',BAEGarg2019), ('PWWSRen2019',PWWSRen2019)])

# 1. Initialize model and load dataset
print(f'***DATASET: {DATASET}***')
print(f'***MODEL_NAME: {MODEL_NAME}***')
model = BertForSequenceClassification.from_pretrained(MODEL_NAME)

model.to(device)
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss = BCEWithLogitsLoss() #CrossEntropyLoss() #TODO verify loss function

raw_datasets = load_dataset(DATASET)


# 2. Dataset and DataLoader objects
class MyDataset(Dataset):
    def __init__(self, dataset, split):
        #self.tokenized_texts = []
        self.labels = []
        self.texts = []

        for example in dataset[split]:
            self.labels.append(example['label'])
            self.texts.append(example['text'])  
            #self.tokenized_texts.append(tokenizer(example['text'], return_tensors='pt'))
    
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        return self.texts[idx], torch.tensor(self.labels[idx])

def pad_collate_classifier(batch):
    (texts, y) = zip(*batch)
    xx_pad = tokenizer.batch_encode_plus(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_SEQ_LENGTH)
    y_stack = torch.stack(y, dim=0)
    return xx_pad, y_stack

#TODO: confirm split based on dataset
train_ds = MyDataset(raw_datasets, 'train')
test_ds = MyDataset(raw_datasets, 'test')

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=pad_collate_classifier)
test_dl = DataLoader(test_ds, batch_size=32, shuffle=True, drop_last=True, collate_fn=pad_collate_classifier)


# 3. confirm proper loaded model training
# https://textattack.readthedocs.io/en/latest/3recipes/models.html#more-details-on-textattack-fine-tuned-nlp-models-details-on-target-nlp-task-input-type-output-type-sota-results-on-paperswithcode-model-card-on-huggingface
# metric= load_metric("accuracy")
# model.eval()
# for batch in test_dl:
    
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["label"])
# acc = metric.compute()
print('\nVERIFYING ACCURACY OF LOADED MODEL')
acc = []
f1s = []
running_loss = 0
model.eval()
with torch.no_grad():
    for batch in tqdm(test_dl):
        x, ys = batch
        out = model(input_ids=x['input_ids'].to(device),
                      attention_mask=x['attention_mask'].to(device),
                      token_type_ids=x['token_type_ids'].to(device))
        preds = out.logits[:,1]
        y = ys.float().to(device)
        test_loss = loss(preds, y)
        running_loss += test_loss.cpu().item()
        preds_labels = torch.round(torch.sigmoid(preds))
        acc.append((torch.sum(preds_labels==y)/y.shape[0]).cpu().item())
        f1s.append(f1_score(ys.detach().numpy(), preds_labels.detach().cpu().numpy()))
    #test_losses.append(running_loss)
    #test_accuracies.append(np.mean(acc))
    #test_f1_scores.append(np.mean(f1s))
    #print(f"Test Loss: {running_loss}")
    
    print(f'\n***Model accuracy from HF: {np.mean(acc)}***')
    print(f"***Test Accuracy: {np.mean(acc)}***")
    print(f"***Test F1-Score: {np.mean(f1s)}***")

####################################################################################
########################### III. ATTACK AND SAVE PARQUET ###########################
####################################################################################

# 4. Run attack
model_wrapped = HuggingFaceModelWrapper(model,tokenizer)
attack = ATTACK_OPTIONS[ATTACK].build(model_wrapped)

#TODO: confirm split based on dataset
train_dataset_wrapped = HuggingFaceDataset(raw_datasets['train'])
#test_dataset_wrapped = HuggingFaceDataset(raw_datasets['test'])

#TODO: disable_stdout = False to check outputs AND ENSURE num_examples=-1
attack_args = textattack.AttackArgs(num_examples=-1,
                                    random_seed=42,
                                    disable_stdout = True,
                                    parallel=False,
                                    checkpoint_interval=200,
                                    checkpoint_dir=f'checkpoints-{DATASET}')


#TODO: confirm split based on dataset
attacker = textattack.Attacker(attack,train_dataset_wrapped,attack_args)
attack_results = attacker.attack_dataset()
print(f'\n***DONE ATTACKING WITH <{ATTACK}> ON DATASET <{DATASET}> AND MODEL <{MODEL_NAME}>***\n')

# 5. Save perturbed examples
logger = CSVLogger(color_method='file')

for result in attack_results:
    logger.log_attack_result(result)


df_1 = logger.df[['original_text','perturbed_text']].replace({'\[\[':'','\]\]':''}, regex=True)
df_1.to_parquet(f'parquet/{ATTACK}-{DATASET}-train.parquet')
print(f'\n***PARQUET FILE SAVED AT:  parquet/{ATTACK}-{DATASET}-train.parquet***\n')

print(f'\n***HEAD OF PARQUET FILE FOR {ATTACK}-{DATASET}-train.parquet***\n')
print(pd.read_parquet(f'parquet/{ATTACK}-{DATASET}-train.parquet').head())