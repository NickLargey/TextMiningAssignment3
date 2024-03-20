import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import pronouncing
import cmudict
import spacy
from spacy import displacy
from pathlib import Path


def EDA(dataframe):
  df = pd.DataFrame()
  df = dataframe.copy()

  counts = df['Genre'].value_counts()
  print(counts)
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
  EDA(df)   

if __name__ == '__main__':
  main()