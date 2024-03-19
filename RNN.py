import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


def main():
  df = pd.read_csv('./lyrics.csv')
  print(df.head())    

if __name__ == '__main__':
  main()