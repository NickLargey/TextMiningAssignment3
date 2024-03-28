"""
Task 3
who did it: Nick Largey
Mar 25,2024
Behrooz Mansouri
470 Text Mining and Analytics
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from tqdm import tqdm

import pprint
pp = pprint.PrettyPrinter(indent=4)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output):
        super(NeuralNet, self).__init__()
        # Create a sequential module
        self.model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output),
        torch.nn.Sigmoid()
        )
    def forward(self, x):
        # Forward pass using the sequential module
        output = self.model(x)
        return output




class LyricsDataset(Dataset):
    def __init__(self, numeric_data, tfidf_data, tokenized_lyrics, pos_data, labels):
        self.numeric_data = torch.tensor(numeric_data, dtype=torch.float32)
        self.tfidf_data = torch.tensor(tfidf_data, dtype=torch.float32)
        self.tokenized_lyrics = tokenized_lyrics.clone().detach()
        self.pos_data = pos_data.clone().detach()
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'numeric_data': self.numeric_data[idx],
            'tfidf_data': self.tfidf_data[idx],
            'tokenized_lyrics': self.tokenized_lyrics[idx],
            'pos_data': self.pos_data[idx],
            'labels': self.labels[idx]
        }



class MultiInputLyricsClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tfidf_input_dim, numeric_input_dim, pos_vocab_size, pos_embedding_dim, output_dim):
        super(MultiInputLyricsClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim)  # Embedding layer for POS tags
        self.tfidf_fc = nn.Linear(tfidf_input_dim, 64)  # Adjust dimensions as needed
        self.numeric_fc = nn.Linear(numeric_input_dim, 32)  # Adjust dimensions as needed
        # Update the dimension for the concatenated layer
        self.final_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, numeric_data, tfidf_data, tokenized_lyrics, pos_tags):
        embedded_lyrics = self.embedding(tokenized_lyrics)
        # Ensure hidden_lyrics is 2D: [batch_size, features]
        hidden_lyrics = embedded_lyrics.squeeze(0) if embedded_lyrics.dim() == 3 else embedded_lyrics
        embedded_pos = self.pos_embedding(pos_tags).sum(dim=1)

        tfidf_out = F.relu(self.tfidf_fc(tfidf_data)).unsqueeze(0)  # Ensure 2D if needed
        numeric_out = F.relu(self.numeric_fc(numeric_data)).unsqueeze(0)  # Ensure 2D if needed
        
        hidden_lyrics_adjusted = hidden_lyrics.mean(dim=1)  # Now [batch_size, features]
        # print(hidden_lyrics_adjusted.dim())

        tfidf_out_2d = tfidf_out.unsqueeze(1) if tfidf_out.dim() == 1 else tfidf_out
        # print(tfidf_out_2d.dim())
        numeric_out_2d = numeric_out.squeeze(0) if numeric_out.dim() == 3 else numeric_out
        # print(numeric_out_2d.dim())
        embedded_pos_2d = embedded_pos.unsqueeze(1) if embedded_pos.dim() == 1 else embedded_pos
        # print(embedded_pos_2d.dim())

        # Now concatenating
        concatenated = torch.cat((hidden_lyrics_adjusted, tfidf_out_2d, numeric_out_2d, embedded_pos_2d), dim=1)
        output = self.final_fc(concatenated)
        return output

nlp = spacy.load("en_core_web_sm") 

'''
Need to find a more effficent way to process POS rather than one hot encoding.
def one_hot_encode(pos_list):
  pos_tags = nlp.get_pipe("tagger").labels
  tag_to_index = {tag: i for i, tag in enumerate(pos_tags)}

  identity_matrix = np.eye(len(tag_to_index))
  one_hot = [[identity_matrix[tag_to_index[tag]] for tag in doc] for doc in pos_list]

  return one_hot
'''
def build_vocab(data):
    """Builds a vocabulary from a list of lists of tokens or tags."""
    counts = Counter([token for token in data])
    vocab = {token: i + 2 for i, token in enumerate(counts)} 
    return vocab, {"<PAD>": 0, "<UNK>": 1}

def encode_sequences(data, vocab):
    """Encodes a list of lists of tokens or tags using the provided vocabulary."""
    return [[vocab.get(token, 1) for token in sequence] for sequence in data]

def pad_sequences(sequences, batch_first=True, padding_value=0):
    """Pads a list of sequences to the same length and converts to a tensor."""
    return rnn_utils.pad_sequence([torch.tensor(seq) for seq in sequences],
                                  batch_first=batch_first, padding_value=padding_value)

def main():
  df = pd.read_csv('./final_df.csv')
  num_genres = df['Genre'].max()
  num_epochs = 20
  data_size = df["Genre"].value_counts().min()

  samples_per_group = data_size  # Number of samples per group
  print("Number of samples per group: ", samples_per_group)
  column_to_group_by = 'Genre'

  # Sample n rows from each group
  df = df.groupby(column_to_group_by).apply(lambda x: x.sample(n=samples_per_group)).reset_index(drop=True)
#   print(df.info())
  ############## Dataset Preprocessing ################ 
  n_cols = ['F_Rhymes','S_Rhymes','FRD','SRD']
  numeric = df[n_cols].copy().reset_index(drop=True)
  numeric_data = numeric.apply(pd.to_numeric, errors='coerce') 
  numeric_data_filled = numeric_data.fillna(0)
  X_numeric = numeric_data_filled.values

  tfidf = df["TF-IDF"].copy().reset_index(drop=True)
  tfidf_data = tfidf.apply(pd.to_numeric, errors='coerce')
  X_tfidf = tfidf_data.values.astype('float32')

  lyrics = df['Tokenized Lyrics'].tolist()
  pos = df['POS'].tolist()
  
  token_vocab, pos_vocab = build_vocab(lyrics)[0], build_vocab(pos)[0]
  
  encoded_lyrics = encode_sequences(lyrics, token_vocab)
  encoded_pos = encode_sequences(pos, pos_vocab)

  X_lyrics = pad_sequences(encoded_lyrics)
  X_pos = pad_sequences(encoded_pos)

#   print(type(X_lyrics))
#   print(type(X_pos))

  y = df['Genre'].to_numpy()

#   print(type(y))

  ############# Train-Val Split ################

  indices = np.arange(data_size)
  train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
  print(len(train_indices), len(val_indices))
  train_numeric = X_numeric[train_indices]
  val_numeric = X_numeric[val_indices]

  train_tfidf = X_tfidf[train_indices]
  val_tfidf = X_tfidf[val_indices]

  train_lyrics = X_lyrics[train_indices]
  val_lyrics = X_lyrics[val_indices]

  train_pos = X_pos[train_indices]
  val_pos = X_pos[val_indices]

  train_genres = y[train_indices]
  val_genres = y[val_indices]

  ############# Data Loading & Model Training ################
  train_dataset = LyricsDataset(train_numeric, train_tfidf, train_lyrics, train_pos, train_genres)
  val_dataset = LyricsDataset(val_numeric, val_tfidf, val_lyrics, val_pos, val_genres)

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  model = MultiInputLyricsClassifier(vocab_size=10000, embedding_dim=64, hidden_dim=128, tfidf_input_dim=32, numeric_input_dim=4, pos_vocab_size=25, pos_embedding_dim=10, output_dim=num_genres)
  optimizer = Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  for epoch in tqdm(range(num_epochs)):
      model.train()
      for batch in train_loader:
          optimizer.zero_grad()
          predictions = model(batch['numeric_data'], batch['tfidf_data'], batch['tokenized_lyrics'], batch['pos_data'])
          loss = criterion(predictions, batch['labels'])
          loss.backward()
          optimizer.step()

  model.predict(val_loader)


  # ############## Test Evaluation ################

  # test_df = pd.read_csv('./test_df.csv')
  
  # genre_ints = {
  #   'Blues': 0,
  #   'Country': 1,
  #   'Metal': 2,
  #   'Pop': 3,
  #   'Rap': 4,
  #   'Rock': 5
  # }

  # test_df['Genre'] = test_df['Genre'].replace(genre_ints)

if __name__ == '__main__':
    main()