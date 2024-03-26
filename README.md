# README: Task3

## Overview

This Python script, 'Task3.py', uses word embeddings and a feed-forward neural network to classify the lyrics based on their genres.

## Project Information

- **Author:** Sarah Lawrence
- **Course:** COS 470: Text Mining and Analytics
- **Assignmnet:** 3
- **Due Date:** Mar 25, 2024

## Script Structure

The script is organized into several functions, each responsible for its own task those tasks are as follows.

## Functions

### 'csv_info' Function
- Input:
  - data_path: the path to retrieve the information
  - genres_count: a dictionary with the no counts for each genre
  - genre_lyrics_dict: the dictionary the information will be put into
- Output:
  -  dictionary of all the genres and their lyrics
- Overall:
  - storing all the genres and their lyric from the CSV file

### 'split_data_by_genre' Function
- Input:
  - Validation_percent: this value is 0.1. It is how the data will be split 90% 10%
  - gGenre_lyrics_dict: this value has all the genre and their lyrics collected from the csv
  - Lowest_genre_count: this is the genre that had the least amount of songs
- Output:
 -  Train data: This dictionary for each genre has the tokenized lyrics of each song. This dictionary has 90% of the data.
 -  Validation data: This dictionary for each genre has the tokenized lyrics of each song but only stores 10% of the data.
- Overall:
  - This function takes in the genre and lyrics.
  - Randomly selects unique genres and lyrics till it gets the same amount of genres and lyrics for each genre.
  - Then it shuffles the genres and lyrics
  - Split the data into validation and training

### 'tokenize_lyrics' Function

- 

### 'get_average_embedding' Function

- 

### 'main' Function
- Overall
  - 

## Result

- ** plot for the Loss values of the training and validation sets
- ** Table of F1 scores for the test and model 


## Usage
1. Ensure the following is downloaded (only need to do this step once)
  - nltk.download('punkt')
  - nltk.download('stopwords')
2. Ensure that the following are installed:

    ```bash
    pip install tensorflow
    pip install gensim
    pip install tqdm
    ```
3. Ensure that everything is in the same directory
   This includes:
  - test.csv
  - lyrics.csv
  - Task3.py
