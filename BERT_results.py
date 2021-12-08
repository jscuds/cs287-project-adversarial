import numpy as np
import pandas as pd
import re
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import pickle


PARENT_DIR = 'parquet/'
SAVE_DIR = 'results/'

### Imports / Globals
DATASET = 'rotten_tomatoes'
MODEL_NAME = 'bert-base-uncased'
ATTACK = 'BAEGarg2019' #TextFoolerJin2019, DeepWordBugGao2018, BAEGarg2019, PWWSRen2019, PHASE3
LR = 2e-05 #5e-05
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 128
FRACS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
EPOCHS = 4

print('***GLOBALS:***')
print(f'Dataset:\t{DATASET}')
print(f'HF Model:\t{MODEL_NAME}')
print(f'Attack:\t\t{ATTACK}')
print(f'LR:\t\t{LR}')
print(f'Batch Size:\t{BATCH_SIZE}')
print(f'Max Seq Length:\t{MAX_SEQ_LENGTH}')
print(f'Fracs:\t{FRACS}')
print(f'Epochs:\t{EPOCHS}')


TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)

## Creating code for running different perturb percentages"""

# load perturbed dataframe
df_perturb = pd.read_parquet(f'{PARENT_DIR}{DATASET}-{ATTACK}-train.parquet')
df_val = pd.read_parquet(f'perturbed-datasets/{DATASET}/{ATTACK}/{DATASET}-{ATTACK}-validation.parquet')
df_test = pd.read_parquet(f'perturbed-datasets/{DATASET}/{ATTACK}/{DATASET}-{ATTACK}-test.parquet')


## Dataset / Dataloader
class RottenDataset(Dataset):
  def __init__(self, df, frac, tokenizer: AutoTokenizer):
    super().__init__()
    self.texts = []
    self.labels = df['label'].astype('int')

    self.perturbed_idx = df.sample(frac=frac).index.tolist()
    for i in range(df.shape[0]):
      if i in self.perturbed_idx:
        self.texts.append(df.loc[i, 'perturbed_text'])
      else:
        self.texts.append(df.loc[i, 'original_text'])

    self.texts = tokenizer(self.texts, truncation=True)

  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    return torch.tensor(self.texts[idx].ids), torch.tensor(self.labels[idx])


def pad_collate_classifier(batch):
    (xx, yy) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=TOKENIZER.pad_token_id)
    y_stack = torch.stack(yy, dim=0)
    return xx_pad, y_stack

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

def run_batch(train_dl, val_dl, test_dl, epochs, model_name):
    model = BertForSequenceClassification.from_pretrained(model_name)
    model.to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = BCEWithLogitsLoss()

    train_losses = {}
    val_losses = {}
    train_accs = {}
    val_accs = {}
    print('Training model {}'.format(model_name))
    for epoch in range(epochs):
      running_loss = 0
      accuracies = []
      for batch, targets in tqdm(train_dl, leave=False):
          preds = model(batch.to('cuda')).logits
          loss = loss_fn(preds[:,1].squeeze(), targets.to('cuda').float())
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()
          running_loss += loss.cpu().item()
          accuracies.append(accuracy_score(targets, torch.round(torch.sigmoid(preds[:,1])).cpu().detach().numpy()))
          
      train_losses['epoch {}'.format(epoch+1)] = running_loss / len(train_ds)
      train_accs['epoch {}'.format(epoch+1)] = np.mean(accuracies)
      print("="*20)
      print(f"Epoch {epoch+1}/{epochs} Train Loss: {running_loss / len(train_ds)}")
      print(f"Epoch {epoch+1}/{epochs} Train Accuracy: {np.mean(accuracies)}" )
      
      running_loss = 0
      accuracies = []
      with torch.no_grad():
          for batch, targets in tqdm(val_dl, leave=False):
              preds = model(batch.to('cuda')).logits
              loss = loss_fn(preds[:,1].squeeze(), targets.to('cuda').float())
              running_loss += loss.cpu().item()
              accuracies.append(accuracy_score(targets, torch.round(torch.sigmoid(preds[:,1])).cpu().detach().numpy()))

      val_losses['epoch {}'.format(epoch+1)] = running_loss / len(val_ds)
      val_accs['epoch {}'.format(epoch+1)] = np.mean(accuracies)
      print(f"Epoch {epoch+1}/{epochs} Val Loss: {running_loss / len(val_ds)}")
      print(f"Epoch {epoch+1}/{epochs} Val Accuracy: {np.mean(accuracies)}" )

    running_loss = 0
    accuracies = []
    with torch.no_grad():
        for batch, targets in tqdm(test_dl, leave=False):
            preds = model(batch.to('cuda')).logits
            loss = loss_fn(preds[:,1].squeeze(), targets.to('cuda').float())
            running_loss += loss.cpu().item()
            accuracies.append(accuracy_score(targets, torch.round(torch.sigmoid(preds[:,1])).cpu().detach().numpy()))
    test_loss = running_loss / len(test_ds)
    test_accs = np.mean(accuracies)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accs}" )
    return train_losses['epoch {}'.format(epochs)], train_accs['epoch {}'.format(epochs)], val_losses['epoch {}'.format(epochs)], val_accs['epoch {}'.format(epochs)], test_loss, test_accs

train_dict = {}
val_dict = {}
test_dict = {}
for FRAC in FRACS:
  #Datasets
  train_ds = RottenDataset(df_perturb,FRAC,TOKENIZER)
  val_ds = RottenDataset(df_val,0,TOKENIZER)
  test_ds = RottenDataset(df_test,0,TOKENIZER)

  # Dataloaders
  train_dl = DataLoader(dataset=train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, collate_fn=pad_collate_classifier)
  val_dl = DataLoader(dataset=val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=pad_collate_classifier)
  test_dl = DataLoader(dataset=test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, collate_fn=pad_collate_classifier)
  train_loss, train_acc, val_loss, val_acc, test_loss, test_acc = run_batch(train_dl, val_dl, test_dl, EPOCHS, MODEL_NAME)
  train_dict[FRAC] = {'loss': train_loss, 'accuracy':train_acc}
  val_dict[FRAC] = {'loss': val_loss, 'accuracy':val_acc}
  test_dict[FRAC] = {'loss': test_loss, 'accuracy':test_acc}

write_pickle(f'{SAVE_DIR}{DATASET}-{ATTACK}-train.pkl', train_dict)
write_pickle(f'{SAVE_DIR}{DATASET}-{ATTACK}-val.pkl', val_dict)
write_pickle(f'{SAVE_DIR}{DATASET}-{ATTACK}-test.pkl', test_dict)
