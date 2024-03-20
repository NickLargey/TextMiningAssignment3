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


pp = pprint.PrettyPrinter(indent=4)
punct_re = r'[^\w\s]'

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
          sentence = text.split('\n')
          sentences[cat].append(sentence)
      except:
        continue
  
    i += 1

  rhymes = {}
  for cat, sentence in sentences.items():
    for phrase in sentence:

      end_word = phrase[-1]
      rhyme = pronouncing.rhymes(end_word)
  pp.pprint(sentences)


def EDA_visualizer(dataframe, counts):
  df = pd.DataFrame()
  df = dataframe.copy()

  spacy.prefer_gpu()
  nlp = spacy.load("en_core_web_trf")

  for cat, max in counts.iteritems():
    rand_idx = np.random.randint(0, max)
    # text = df[df['Genre'] == cat].iloc[rand_idx]['Lyrics']
    title = df[df['Genre'] == cat].iloc[rand_idx]['Song Title']
    with open(f"./Training Songs/{cat}/{title}.txt", 'r') as file:
      text = file.read()
      doc = nlp(text)

      options = {"bg": "#09a3d5", "distance": 50,
                 "color": "black","add_lemma": True, "compact":True, "font": "Arial"}
      svg = displacy.render(doc, style="dep",options=options , jupyter=False)
      output_path = Path(f'./images/{cat} - {title}.svg')
      output_path.open("w", encoding="utf-8").write(svg)





def main():
  df = pd.read_csv('./lyrics.csv')
  counts = df['Genre'].value_counts()
  print(counts)
  rhyme_runs = {}
  # EDA_visualizer(df, counts)
  EDA_rhymes(df, counts)   

if __name__ == '__main__':
  main()