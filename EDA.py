"""
Task 3
who did it: Nick Largey
Mar 25,2024
Behrooz Mansouri
470 Text Mining and Analytics
"""

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import pronouncing # for rhyming words
import spacy
from spacy import displacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

from pathlib import Path
from tqdm import tqdm
import pprint
import seaborn as sns
from collections import Counter
from langdetect import detect, LangDetectException

spacy.prefer_gpu()

pp = pprint.PrettyPrinter(indent=4)
punct_re = re.compile(r'[^\w\s]')


def EDA_rhymes(dataframe):
  df = pd.DataFrame()
  df = dataframe.copy()

  first_rhymes = []
  second_rhymes = []
  f_rd = []
  s_rd = []

  for i, row in tqdm(df.iterrows()):
    cat = row['Genre']
    song_title = row['Song Title']
    try:
      with open(f"./Training Songs/{cat}/{song_title}.txt", 'r') as file:
        text = file.read()
        sentence = re.sub(punct_re,'', text)
        sentences = [s for s in sentence.split('\n') if s.strip()]
        rhymes = []
        f_cnt = 0
        s_cnt = 0
        
        for words in sentences:
          end_word = words.split(' ')[-1]
          rhyme = pronouncing.rhymes(end_word)
          rhymes.append((end_word, rhyme))

        keys = [k[0] for k in rhymes]  # Get a list of keys
       
        for i in range(len(keys)):
          current_key = keys[i]
          if i + 1 < len(keys):
            next_key = i + 1
            if current_key in rhymes[next_key][1] or current_key == next_key:
              f_cnt += 1
          if i + 2 < len(keys):
            next_next_key = i + 2
            if current_key in rhymes[next_next_key][1] or current_key == next_next_key:
              s_cnt += 1

        first_rhymes.append(f_cnt)
        second_rhymes.append(s_cnt)
        f_rd.append(f_cnt/len(text))
        s_rd.append(s_cnt/len(text))

    except:
        first_rhymes.append(0)
        second_rhymes.append(0)
        f_rd.append(0)
        s_rd.append(0)

  return first_rhymes, second_rhymes, f_rd, s_rd


def compute_tf(document):
    # Compute term frequency for each term in the document
    tf_document = Counter(document)
    num_words = len(document)
    return {word: count / num_words for word, count in tf_document.items()}


def compute_idf(documents):
    # Compute document frequency for each term
    num_documents = len(documents)
    df = Counter()
    for document in documents:
        df.update(set(document))
    
    # Compute IDF, adding 1 to denominator to avoid division by zero
    return {word: math.log(num_documents / (freq + 1)) for word, freq in df.items()}


def tfidf(documents):
    # Compute TF for each document
    tfs = [compute_tf(doc) for doc in documents]
    
    # Compute IDF
    idf = compute_idf(documents)
    
    # Compute TF-IDF
    tf_idf_documents = []
    for tf in tfs:
        tf_idf = {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}
        tf_idf_documents.append(tf_idf)
    
    return tf_idf_documents


def EDA_TF_IDF(dataframe):
  df = pd.DataFrame()
  df = dataframe.copy()

  documents = df['Lyrics'].tolist() 
  documents = [[word.lower() for word in doc.split()] for doc in documents]  # Convert all words to lowercase
  # Compute TF for each document
  tfs = [compute_tf(doc) for doc in documents]
  
  # Compute IDF
  idf = compute_idf(documents)
  
  # Compute TF-IDF
  tf_idf_documents = []
  for tf in tqdm(tfs):
      tf_idf = {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}
      tf_idf_documents.append(tf_idf)

  return tf_idf_documents


def EDA_visualizer(nlp, dataframe, counts):
  df = dataframe.copy()
  for cat, max in counts.iteritems():
    rand_idx = np.random.randint(0, max)
    title = df[df['Genre'] == cat].iloc[rand_idx]['Song Title']
    with open(f"./Training Songs/{cat}/{title}.txt", 'r') as file:
      text = file.read()
      doc = nlp(text)

      options = {"bg": "#09a3d5", "distance": 50,
                 "color": "black","add_lemma": True, "compact":True, "font": "Arial"}
      svg = displacy.render(doc, style="dep",options=options , jupyter=False)
      output_path = Path(f'./images/{cat} - {title}.svg')
      output_path.open("w", encoding="utf-8").write(svg)



def EDA_tokenize(nlp, dataframe):
  tokenizer = Tokenizer(nlp.vocab)

  df = pd.DataFrame()
  df = dataframe.copy()
  lyrics = df['Lyrics'].tolist()  
  
  pos = []
  tokenized_lyrics = []
  
  for doc in tqdm(lyrics):
    tokens = tokenizer(doc)
    tokenized_lyrics.append([token.text for token in tokens])
    p_doc = nlp(doc)
    doc_pos = [token.pos_ for token in p_doc]
    pos.append(doc_pos) 
    
  return tokenized_lyrics, pos


def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def to_csv(dataframe):
  mask = dataframe['Lyrics'].apply(is_english)
  filtered_df = dataframe[mask]
  filtered_df.to_csv('filtered_lyrics.csv', index=False)


def EDA_corr_heat_map(dataframe):
  Xy_train = dataframe.copy()
  corr = Xy_train[['Genre','F_Rhymes', 'S_Rhymes', 'FRD', 'SRD', 'TF-IDF', 'POS', 'Tokenized Lyrics']].corr()
  heatmap = sns.heatmap(corr, vmin=-1, vmax=1, cmap='BrBG', annot=True)
  heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':20}, pad=14)
  plt.savefig('images/correlation_heatmap.png', dpi=300)

def main():
  # df = pd.read_csv('./lyrics.csv')
  # to_csv(df)
  filtered_df = pd.read_csv('./filtered_lyrics.csv')
  genre_ints = {
      'Blues': 0,
      'Country': 1,
      'Metal': 2,
      'Pop': 3,
      'Rap': 4,
      'Rock': 5
  }

  counts = filtered_df['Genre'].value_counts()
  EDA_visualizer(filtered_df, counts)
  nlp = spacy.load("en_core_web_trf")
  final_df = pd.DataFrame()
  final_df["Genre"] = filtered_df['Genre']
  
  f,s, rd, srd = EDA_rhymes(filtered_df)   
  term = EDA_TF_IDF(filtered_df)
  token, pos = EDA_tokenize(nlp, filtered_df)

  final_df['Genre'] = final_df['Genre'].replace(genre_ints)
  final_df['F_Rhymes'] = f
  final_df['S_Rhymes'] = s
  final_df['FRD'] = rd
  final_df['SRD'] = srd
  final_df['TF-IDF'] = term
  final_df['POS'] = pos
  final_df['Tokenized Lyrics'] = token

  final_df.to_csv('final_df.csv', index=False)

  EDA_corr_heat_map(final_df)


if __name__ == '__main__':
  main()