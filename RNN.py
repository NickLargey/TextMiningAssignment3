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


class LyricsDataset(Dataset):
    def __init__(self, numeric_data, tfidf_data, tokenized_lyrics, pos_data, labels):
        self.numeric_data = torch.tensor(numeric_data, dtype=torch.float32)
        self.tfidf_data = torch.tensor(tfidf_data, dtype=torch.float32)
        self.tokenized_lyrics = torch.tensor(tokenized_lyrics, dtype=torch.long)
        self.pos_data = torch.tensor(tokenized_lyrics, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)  # Assuming labels are encoded as integers

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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.tfidf_fc = nn.Linear(tfidf_input_dim, 64)  # Adjust dimensions as needed
        self.numeric_fc = nn.Linear(numeric_input_dim, 32)  # Adjust dimensions as needed
        # Update the dimension for the concatenated layer
        self.final_fc = nn.Linear(hidden_dim + 64 + 32 + pos_embedding_dim, output_dim)

    def forward(self, numeric_data, tfidf_data, tokenized_lyrics, pos_tags):
        embedded_lyrics = self.embedding(tokenized_lyrics)
        _, (hidden_lyrics, _) = self.lstm(embedded_lyrics)
        hidden_lyrics = hidden_lyrics.squeeze(0)  # Remove the extra dimension
        
        embedded_pos = self.pos_embedding(pos_tags).sum(dim=1)  # Example way to aggregate POS embeddings

        tfidf_out = F.relu(self.tfidf_fc(tfidf_data))
        numeric_out = F.relu(self.numeric_fc(numeric_data))
        
        # Concatenate all features including the POS embeddings
        concatenated = torch.cat((hidden_lyrics, tfidf_out, numeric_out, embedded_pos), dim=1)
        output = self.final_fc(concatenated)
        return output

nlp = spacy.load("en_core_web_trf") 

def one_hot_encode(pos_list):
  
  pos_tags = nlp.get_pipe("tagger").labels
  tag_to_index = {tag: i for i, tag in enumerate(pos_tags)}

  identity_matrix = np.eye(len(tag_to_index))
  one_hot = [[identity_matrix[tag_to_index[tag]] for tag in doc] for doc in pos_list]

  return one_hot


def build_vocab(data):
    """Builds a vocabulary from a list of lists of tokens or tags."""
    counts = Counter(token for sequence in data for token in sequence)
    return {token: i + 2 for i, token in enumerate(counts)}, {"<PAD>": 0, "<UNK>": 1}

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
  column_to_group_by = 'Genre'

  # Sample n rows from each group
  df = df.groupby(column_to_group_by).apply(lambda x: x.sample(n=samples_per_group)).reset_index(drop=True)



  numeric = df.copy(['F_Rhymes','S_Rhymes','FRD','SRD'])
  numeric_data = numeric.apply(pd.to_numeric, errors='coerce')
  numeric_data_filled = numeric_data.fillna(0)
  X_numeric = numeric_data_filled.values

  tfidf = df.copy('TF_IDF')
  tfidf_data = tfidf.apply(pd.to_numeric, errors='coerce')
  X_tfidf = tfidf_data.values.astype('float32')

  lyrics = df['Tokenized Lyrics'].tolist()
  pos = df['POS'].tolist()
  token_vocab, pos_vocab = build_vocab(lyrics), build_vocab(pos)[0]
  encoded_lyrics = encode_sequences(lyrics, token_vocab)
  encoded_pos = encode_sequences(pos, pos_vocab)

  X_lyrics = pad_sequences(encoded_lyrics)
  X_pos = pad_sequences(encoded_pos)

  y = df.copy('Genre')

  indices = np.arange(df.shape[0])
  train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

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


  train_dataset = LyricsDataset(train_numeric, train_tfidf, train_lyrics, train_pos, train_genres)
  val_dataset = LyricsDataset(val_numeric, val_tfidf, val_lyrics, val_pos, val_genres)

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  model = MultiInputLyricsClassifier(vocab_size=10000, embedding_dim=64, hidden_dim=128, tfidf_input_dim=100, numeric_input_dim=4, pos_vocab_size=25, pos_embedding_dim=15, output_dim=num_genres)
  optimizer = Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  for epoch in range(num_epochs):
      model.train()
      for batch in train_loader:
          optimizer.zero_grad()
          predictions = model(batch['numeric_data'], batch['tfidf_data'], batch['tokenized_lyrics'])
          loss = criterion(predictions, batch['labels'])
          loss.backward()
          optimizer.step()

  model.predict(val_loader)

  genre_ints = {
    'Blues': 0,
    'Country': 1,
    'Metal': 2,
    'Pop': 3,
    'Rap': 4,
    'Rock': 5
  }
if __name__ == '__main__':
    main()