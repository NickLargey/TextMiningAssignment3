"""
Task 3
who did it: Sarah Lawrence
Mar 20,2024
Behrooz Mansouri
470 Text Mining and Analytics
"""
import os
import csv
import random
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from tqdm import tqdm 

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# nltk.download('punkt')
# nltk.download('stopwords')

#pip install tensorflow
#pip install gensim
#pip install tqdm
#conda activate Task3


def split_data_by_genre(validation_percent=0.1, genre_lyrics_dict=None, lowest_genre_count=None):
    train_data = {}
    validation_data = {}

    random_selections = []
    for _ in range(lowest_genre_count):
        for genre, lyrics in genre_lyrics_dict.items():
            random_lyric = random.sample(lyrics,lowest_genre_count)
            random_selections.append((genre, random_lyric))
    random.shuffle(random_selections)

    validation_size = int(lowest_genre_count * validation_percent)

    for category,lyrics_list in random_selections:
        validation = random.sample(lyrics_list, validation_size)
        validation_data[category] = validation
        train_data[category] = [pair for pair in lyrics_list if pair not in validation_data[category]]     
    return train_data, validation_data

def tokenize_lyrics(lyrics):
    tokens = word_tokenize(lyrics.lower())
    stop_words = set(stopwords.words('english'))
    # Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]
     # Remove stop words
    filtered_tokens = [word for word in tokens if word not in stop_words] 
    return filtered_tokens

def csv_info(data_path,genres_count,genre_lyrics_dict):
    with open(data_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        # Count total rows for tqdm
        total_rows = sum(1 for row in csv_reader)  
        # Reset pointer for csv
        file.seek(0)
        next(csv_reader)
        
        # Collecting the csv informtion 
        for row in tqdm(csv_reader, total=total_rows, desc='CSV Processing'):
            lyrics = row['Lyrics']
            genre = row['Genre']

            genres_count[genre] += 1

            tokenized_lyrics = tokenize_lyrics(lyrics)
            
            # Storing genre and lyrics
            if genre in genre_lyrics_dict:
                genre_lyrics_dict[genre].append(tokenized_lyrics)
            else:
                # If genre not in dictionary, create a new list with lyrics
                genre_lyrics_dict[genre] = [tokenized_lyrics]
    
    return genre_lyrics_dict   


def get_average_embedding(tokens,word2vec_model):
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def main():
    

    # Getting csv information from lyrics
    #print("Collecting informtion from lyrcs has started")
    find_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(find_directory, 'lyrics.csv')
    genres_count = {"Pop": 0, "Rap": 0, "Rock": 0, "Metal": 0, "Country": 0, "Blues": 0}
    genre_lyrics_dict = {}
    genres = []

    genre_lyrics_dict = csv_info(data_path,genres_count,genre_lyrics_dict)

    # Valdation and Training data
    lowest_genre_count = min(len(lyrics_list) for lyrics_list in genre_lyrics_dict.values()) 
    train_data, validation_data = split_data_by_genre(validation_percent=0.1,genre_lyrics_dict=genre_lyrics_dict,lowest_genre_count=lowest_genre_count)

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=[tokens for sublist in train_data.values() for tokens in sublist],
                              vector_size=100, window=5, min_count=1, workers=4)
    
    # Calculate average embeddings for training data
    for genre, token_list in tqdm(train_data.items(), desc='Training Data Processing'):
        train_data[genre] = [get_average_embedding(tokens, word2vec_model) for tokens in token_list]
        genres.append(genre)
    # Calculate average embeddings for validation data
    for genre, token_list in tqdm(validation_data.items(), desc='Validation Data Processing'):
        validation_data[genre] = [get_average_embedding(tokens, word2vec_model) for tokens in token_list]
    
    # Convert data to numpy arrays for model training
    X_train = np.concatenate([v for v in train_data.values()])
    X_val = np.concatenate([v for v in validation_data.values()])
    
    # Encode labels
    le = LabelEncoder()
    y_train = np.concatenate([np.full(len(v), k) for k, v in train_data.items()])
    y_val = np.concatenate([np.full(len(v), k) for k, v in validation_data.items()])
    
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    
    # Convert labels to one-hot encoding
    num_classes = len(np.unique(y_train_encoded))
    y_train_onehot = np.eye(num_classes)[y_train_encoded]
    y_val_onehot = np.eye(num_classes)[y_val_encoded]

    # Define a feedforward neural network model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),  # Word2Vec vector_size = 100 so input_shape = 100
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer with num_classes
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train_onehot, epochs=30, batch_size=32, validation_data=(X_val, y_val_onehot))

    # Creating plot for the Loss values of the training and valdation sets
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('loss_plot.pdf')
    #plt.show()

    # Getting csv information from test
    test_data_path = os.path.join(find_directory, 'test.csv')
    genre_test_dict = {}
    test_data = {}
    genres_count = {"Pop": 0, "Rap": 0, "Rock": 0, "Metal": 0, "Country": 0, "Blues": 0}

    test_data = csv_info(test_data_path,genres_count,genre_test_dict)

    # Embedding tokens 
    for genre, token_list in tqdm(test_data.items(), desc='Test Data Processing'):
        test_data[genre] = [get_average_embedding(tokens, word2vec_model) for tokens in token_list]
    
    # Convert data to numpy arrays
    X_test = np.concatenate([v for v in test_data.values()])

    # Encode labels
    y_test = np.concatenate([np.full(len(v), k) for k, v in test_data.items()])
    y_test_encoded = le.transform(y_test)

    #Evaluate F1 scores from the test and modle 
    # Modle F1-Score
    modle_f1_genre = {}
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_val_encoded, y_pred_classes, average='weighted')
    f1_per_genre = f1_score(y_val_encoded, y_pred_classes, average=None)
    for genre, f1_score_genre in tqdm(zip(genres, f1_per_genre), desc='Modle F1 Score Processing'):    
       modle_f1_genre[genre] = f1_score_genre

    # Test F1-Score    
    test_f1_genre = {}
    y_pred_test = model.predict(X_test)
    y_pred_classes_test = np.argmax(y_pred_test, axis=1)
    test_avrage_f1 = f1_score(y_val_encoded, y_pred_classes, average='weighted')
    f1_test = f1_score(y_test_encoded, y_pred_classes_test, average=None)
    for genre, f1_test in tqdm(zip(genres, f1_test), desc='Test F1 Score Processing'):    
       test_f1_genre[genre] = f1_test
    
   # Data for the table
    genres = []
    genres = list(modle_f1_genre.keys())
    rows_model = ['Test', 'Modle']
    cols_model =  genres + ['Average']
    table_data = []

    for genre in genres:
        table_data.append([test_f1_genre[genre], modle_f1_genre[genre]])
    table_data.append([f1,test_avrage_f1])

    # Table size setup
    _, axs = plt.subplots(figsize=(6, 6))
    # Hide axes
    axs.axis('off')

    # Create the table
    axs.table(cellText=table_data,
                    rowLabels=cols_model,
                    colLabels=rows_model,
                    cellLoc='center',
                    loc='center')

    plt.title('F1 Scores for Test and Modle')
   
    # Save the table as a PDF
    plt.savefig('f1_scores_table.pdf')
    plt.show()
    
    print("All Done! :)")
            
if __name__ == "__main__":
    main()