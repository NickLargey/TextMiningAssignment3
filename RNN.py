import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import spacy
from spacy import displacy


def EDA(dataframe):
  df = pd.copy(dataframe)

  nlp = spacy.load("en_core_web_sm")
  text = df.iloc[0]['lyrics']
  doc = nlp(text)
  displacy.serve(doc, style="dep")





def main():
  df = pd.read_csv('./lyrics.csv')
  EDA(df)   

if __name__ == '__main__':
  main()