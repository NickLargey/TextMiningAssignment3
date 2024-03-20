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
from sklearn.metrics import f1_score, classification_report



import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# nltk.download('punkt')
# nltk.download('stopwords')

#pip install tensorflow
#pip install gensim


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

def get_average_embedding(tokens,word2vec_model):
    embeddings = [word2vec_model.wv[word] for word in tokens if word in word2vec_model.wv]
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)

def main():
    
    # valdating and training
    find_directory = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(find_directory, 'lyrics.csv')
    genres_count = {"Pop": 0, "Rap": 0, "Rock": 0, "Metal": 0, "Country": 0, "Blues": 0}
    genre_lyrics_dict = {}

    with open(data_path, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            lyrics = row['Lyrics']
            genre = row['Genre']

            genres_count[genre] += 1
            
            # Storing genre and lyrics
            if genre in genre_lyrics_dict:
                genre_lyrics_dict[genre].append(lyrics)
            else:
                # If genre not in dictionary, create a new list with lyrics
                genre_lyrics_dict[genre] = [lyrics]

    # Valdation and Training data
    lowest_genre_count = min(len(lyrics_list) for lyrics_list in genre_lyrics_dict.values()) 
    train_data, validation_data = split_data_by_genre(validation_percent=0.1,genre_lyrics_dict=genre_lyrics_dict,lowest_genre_count=lowest_genre_count)

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=[tokens for sublist in train_data.values() for tokens in sublist],
                              vector_size=100, window=5, min_count=1, workers=4)
    
    # Calculate average embeddings for training data
    for genre, lyrics_list in train_data.items():
        train_data[genre] = [get_average_embedding(tokens, word2vec_model) for tokens in lyrics_list]

    # Calculate average embeddings for validation data
    for genre, lyrics_list in validation_data.items():
        validation_data[genre] = [get_average_embedding(tokens, word2vec_model) for tokens in lyrics_list]
    
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
    history = model.fit(X_train, y_train_onehot, epochs=10, batch_size=32, validation_data=(X_val, y_val_onehot))

    # Evaluate the model
    _, accuracy = model.evaluate(X_val, y_val_onehot)
    print("Modle Accuracy:", accuracy)

    # Access loss history
    valdation_loss = history.history['val_loss']
    training_loss = history.history['loss']
    print("Training Losses:")
    for epoch, loss_value in enumerate(training_loss, 1):
        print(f"Epoch {epoch}: {loss_value}")
    print("Valdation Losses:")
    for epoch, loss_value in enumerate(valdation_loss, 1):
        print(f"Epoch {epoch}: {loss_value}")

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('loss_plot.pdf')
    plt.show()

    # Results showing F1-Score of your model 
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_val_encoded, y_pred_classes, average='weighted')
    # F1-Score per genre
    report = classification_report(y_val_encoded, y_pred_classes)
    print('Overall F1-Score:', f1)
    print('F1-Score per genre:\n', report)

    plt.figure(figsize=(8, 6))
    cell_text = [[f1]]
    rows = ['Overall F1-Score']
    cols = ['Score']
    plt.table(cellText=cell_text, rowLabels=rows, colLabels=cols, loc='center')

    # Hide axes
    plt.axis('off')

    # Save the table as a PDF
    plt.savefig('f1_scores_table.pdf')
    plt.show()
            
if __name__ == "__main__":
    main()