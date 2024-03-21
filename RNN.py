import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import re
import pronouncing # for rhyming words
import cmudict # for syllable count and section structure
import spacy
from spacy import displacy
from pathlib import Path
from tqdm import tqdm
import pprint
import seaborn as sns
from collections import Counter
from langdetect import detect, LangDetectException


pp = pprint.PrettyPrinter(indent=4)
punct_re = re.compile(r'[^\w\s]')

def EDA_rhymes(dataframe, counts):
  df = pd.DataFrame()
  df = dataframe.copy()

  sentences = {}

  i = 0
  while i < 10:
    for cat, max in tqdm(counts.iteritems()):
      rand_idx = np.random.randint(0, max)
      song_title = df[df['Genre'] == cat].iloc[rand_idx]['Song Title']

      try:
        with open(f"./Training Songs/{cat}/{song_title}.txt", 'r') as file:
          if cat not in sentences:
            sentences[cat] = []
          text = file.read()
          sentence = re.sub(punct_re,'', text)
          sentence = sentence.split('\n')
          sentences[cat].append(sentence)
      except:
        continue
  
    i += 1

  rhymes = {}
  for cat, sentence in sentences.items():
    for i,phrase in enumerate(range(len(sentence)-1)):
      end_word = phrase[-1]
      rhyme = pronouncing.rhymes(end_word)
      rhymes[end_word] = rhyme 

  pp.pprint(sentences)


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

def tfidf_from_df(df):
    documents = df['Lyrics'].tolist()  # Convert the DataFrame column to a list of documents
    
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

  tf_idf_scores = []

  for cat in df.groupby('Genre'):
    tf_idf_score = tfidf_from_df(cat)
    tf_idf_scores.append(tf_idf_score)
  # For demonstration, convert to PyTorch tensors (optional)
  for doc_id, scores in enumerate(tf_idf_scores):
      # Assume vocabulary is the union of all words in all documents
      vocabulary = set(word for document in df['Lyrics'] for word in document)
      tensor = torch.zeros(len(vocabulary))
      word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
      for word, score in scores.items():
          tensor[word_to_idx[word]] = score
      print(f"Document {doc_id} TF-IDF scores as tensor:\n{tensor}")


def EDA_visualizer(dataframe, counts):
  df = pd.DataFrame()
  df = dataframe.copy()

  spacy.prefer_gpu()
  nlp = spacy.load("en_core_web_trf")

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


# Bag of Words (BoW): Represents the presence of words within the text. While simple, BoW can be surprisingly effective for text classification. However, it ignores word order and context.
# Term Frequency-Inverse Document Frequency (TF-IDF): Weighs the words based on how unique they are to a document. TF-IDF can help emphasize words that are distinctive to certain genres.
# Sentiment Analysis: The sentiment of lyrics might correlate with certain genres. For example, happier sentiments could be more prevalent in pop songs, while darker, 
      # more melancholic sentiments might be more common in some subgenres of rock or metal.
# Lexical Diversity: Measures how varied an artist's vocabulary is within their lyrics. Some genres might exhibit higher lexical diversity than others.
# Topic Modeling Features: Techniques like Latent Dirichlet Allocation (LDA) can identify topics within lyrics. The prevalence of certain topics might be indicative of specific genres.
# Linguistic Features: This includes the use of specific parts of speech (adjectives, nouns, verbs), which can vary by genre. For example, more aggressive language might be prevalent in certain genres like rap or metal.
# Stylometric Features: These are based on the writing style, including sentence length, the use of certain punctuation, rhyme patterns, and other stylistic elements. Some genres might have 
      # a more complex structure or use more figurative language.
# Metadata: While not directly related to the lyrics, metadata such as the artist, album, and year of release can provide contextual clues that are helpful for genre classification.
# Cultural References: References to specific places, people, events, or cultural elements can hint at a song's genre, especially for genres closely tied to particular themes or communities.    

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False
    
def to_csv(dataframe):
  mask = dataframe['Lyrics'].apply(is_english)
  filtered_df = dataframe[mask]
  filtered_df.to_csv('filtered_lyrics.csv', index=False)


def main():
  # df = pd.read_csv('./lyrics.csv')
  # to_csv(df)
  filtered_df = pd.read_csv('./filtered_lyrics.csv')
  # counts = filtered_df['Genre'].value_counts()
  # print(counts)
  # rhyme_runs = {}
  # EDA_visualizer(filtered_df, counts)
  # EDA_rhymes(filtered_df, counts)   
  EDA_TF_IDF(filtered_df)


if __name__ == '__main__':
  main()