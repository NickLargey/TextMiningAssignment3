# README: Task 1

## Overview

This Python script, 'scraping.py', web scraping lyrics.com for the genre and the lyrics of each song.

## Project Information

- **Author:** Nick Largey and Sarah Lawrence
- **Course:** COS 470: Text Mining and Analytics
- **Assignment** 3
- **Due Date:** Mar 25, 2024

## Script Structure

The script is organized into several functions, each responsible for its own task those tasks are as follows.

## Functions

### 'scrape_lyrics' Function
- Overall:
  - storing titles and lyrics from the sight
 
### 'get_all_genre_songs' Function
- Overall:
  - getting all the links that access the song lyrics
 
### 'songs_in_genre_files' Function
- Overall:
  - going through all the pages
  - using get_all_genre_songs to get all the links on each page
  - then using scrape_lyrics to scrape the lyrics

### 'compare_folders' Function
- Overall:
  - compares the test titles to the scraped song titles

### 'to_csv' Function
- Overall:
  - putting all the scraped information into a csv

### 'main' Function
- Overall:
  - generates folders
  - runners each genre tasks in genres_dict independently through songs_in_genre_files
  - removing training songs that match with test songs
  - puts the training and test into a CSV file

## Result

- ** lyrics.csv
- ** test.csv

## Usage
1. Ensure that the following are installed:

    ```bash
    pip install beautifulsoup4
    ```
2. This will take a while to run
   - It's recommended to use the provided lyrics.csv if you want to quickly move to the next tasks

# README: Task 3

## Overview

This Python script, 'Task3.py', uses word embeddings and a feed-forward neural network to classify the lyrics based on their genres.

## Project Information

- **Author:** Sarah Lawrence
- **Course:** COS 470: Text Mining and Analytics
- **Assignment** 3
- **Due Date:** Mar 25, 2024

## Script Structure

The script is organized into several functions, each responsible for its own task those tasks are as follows.

## Functions

### 'csv_info' Function
- Overall:
  - storing all the genres and their lyric from the CSV file

### 'split_data_by_genre' Function
- Overall:
  - This function takes in the genre and lyrics.
  - Randomly selects unique genres and lyrics till it gets the same amount of genres and lyrics for each genre.
  - Then it shuffles the genres and lyrics
  - Split the data into validation and training

### 'tokenize_lyrics' Function
- Overall:
  - this tokenizes the lyrics passed to it  


### 'get_average_embedding' Function
- Overall:
  - this tokenizes the lyrics passed to it  

### 'main' Function
- Overall
  - gets CSV information from lyrics
  - splits data into validation and training data
  - train Word2Vec with training data
  - calculate average embeddings
  - create a feedforward neural network model
  - train the model
  - saves information for loss plot
  - runs test data through model
  - gets F1 scores from the test and model
  - displays f1 scores and loss plot

## Result

- ** plot for the Loss values of the training and validation sets
- ** Table of F1 scores for the test and model 


## Usage
1. Ensure the following is downloaded (only need to do this step once)
  - nltk.download('punkt')
  - nltk.download('stopwords')
2. Ensure that the following are installed:

    ```bash
    pip install tensorflow gensim tqdm
    ```
3. Ensure that everything is in the same directory
   This includes:
  - test.csv
  - lyrics.csv
  - Task3.py
