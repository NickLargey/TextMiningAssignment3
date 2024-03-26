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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.optim import Adam
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report


class LyricsDataset(Dataset):
    def __init__(self, numeric_data, tfidf_data, tokenized_lyrics, labels):
        self.numeric_data = torch.tensor(numeric_data, dtype=torch.float32)
        self.tfidf_data = torch.tensor(tfidf_data, dtype=torch.float32)
        self.tokenized_lyrics = torch.tensor(tokenized_lyrics, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)  # Assuming labels are encoded as integers

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'numeric_data': self.numeric_data[idx],
            'tfidf_data': self.tfidf_data[idx],
            'tokenized_lyrics': self.tokenized_lyrics[idx],
            'labels': self.labels[idx]
        }



class MultiInputLyricsClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tfidf_input_dim, numeric_input_dim, output_dim):
        super(MultiInputLyricsClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.tfidf_fc = nn.Linear(tfidf_input_dim, 64)  # Adjust dimensions as needed
        self.numeric_fc = nn.Linear(numeric_input_dim, 32)  # Adjust dimensions as needed
        self.final_fc = nn.Linear(hidden_dim + 64 + 32, output_dim)  # Sum of all previous layers' outputs

    def forward(self, numeric_data, tfidf_data, tokenized_lyrics):
        embedded = self.embedding(tokenized_lyrics)
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden.squeeze(0)  # Remove the extra dimension
        
        tfidf_out = F.relu(self.tfidf_fc(tfidf_data))
        numeric_out = F.relu(self.numeric_fc(numeric_data))
        
        concatenated = torch.cat((hidden, tfidf_out, numeric_out), dim=1)
        output = self.final_fc(concatenated)
        return output

nlp = spacy.load("en_core_web_trf") 

def one_hot_encode(pos_list):
  
  pos_tags = nlp.get_pipe("tagger").labels
  tag_to_index = {tag: i for i, tag in enumerate(pos_tags)}

  identity_matrix = np.eye(len(tag_to_index))
  one_hot = [[identity_matrix[tag_to_index[tag]] for tag in doc] for doc in pos_list]

  return one_hot


def main():
  df = pd.read_csv('./final_df.csv')
  num_genres = df['Genre'].max()
  num_epochs = 20
  data_size = df["Genre"].value_counts().min()

  n_samples_per_group = data_size  # Number of samples per group
  column_to_group_by = 'Genre'

  # Sample n rows from each group
  df = df.groupby(column_to_group_by).apply(lambda x: x.sample(n=n_samples_per_group)).reset_index(drop=True)



  numeric = df.copy(['F_Rhymes','S_Rhymes','FRD','SRD'])
  numeric_data = numeric.apply(pd.to_numeric, errors='coerce')
  numeric_data_filled = numeric_data.fillna(0)
  X_numeric = numeric_data_filled.values

  tfidf = df.copy('TF_IDF')
  tfidf_data = tfidf.apply(pd.to_numeric, errors='coerce')
  X_tfidf = tfidf_data.values.astype('float32')

  lyrics = df.copy('Tokenized Lyrics')
  tokenized_lyrics_list = lyrics['Tokenized Lyrics'].apply(list).tolist()
  
  max_length = max(len(lyric) for lyric in tokenized_lyrics_list)
  X_lyrics = pad_sequences(tokenized_lyrics_list, maxlen=max_length, padding='post', dtype='int64')

  y = df.copy('Genre')


  train_numeric, val_numeric, train_tfidf, val_tfidf, train_lyrics, val_lyrics, train_genres, val_genres = train_test_split(X_numeric, X_tfidf, X_lyrics, y, test_size=.2, random_state=42)

  train_dataset = LyricsDataset(train_numeric, train_tfidf, train_lyrics, train_genres)
  val_dataset = LyricsDataset(val_numeric, val_tfidf, val_lyrics, val_genres)

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  model = MultiInputLyricsClassifier(vocab_size=10000, embedding_dim=64, hidden_dim=128, tfidf_input_dim=100, numeric_input_dim=4, output_dim=num_genres)
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

if __name__ == '__main__':
    main()