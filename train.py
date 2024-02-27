#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup
import re
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import TextVectorization, Embedding, Dense, GlobalAveragePooling1D, Input
import keras.callbacks
import pickle

def load_data(sqlite_file):
    connection = sqlite3.connect(sqlite_file)
    df = pd.read_sql("SELECT * FROM tvmaze", con=connection)
    df2 = pd.read_sql("SELECT * FROM tvmaze_genre", connection)
    df3 = pd.read_sql("SELECT t.description as Description, GROUP_CONCAT(tg.genre) as Genre from tvmaze t JOIN tvmaze_genre tg on t.tvmaze_id = tg.tvmaze_id group by description;", connection)
    connection.close()
    return df, df2, df3

def clean_data(df3):
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Operate on a copy to prevent SettingWithCopyWarning
    df3 = df3.copy()
    df3['Description'] = df3['Description'].apply(clean_text)
    df3['Description'] = df3['Description'].apply(remove_stopwords)
    df3 = df3[~(df3['Description'].isin(['', 'dupe:']) | df3['Description'].str.endswith('dupe:'))]
    df3.reset_index(drop=True, inplace=True)
    df3['Description'] = df3['Description'].apply(tokenize_and_lemmatize)
    df3['Genre'] = df3['Genre'].str.split(',')

    return df3



def clean_text(text):
    if text is None:
        return ''
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = ' '.join(text.split())
    return text

def remove_stopwords(text):
    if isinstance(text, str):
        stop_words = set(nltk.corpus.stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(words)
    else:
        return text

def tokenize_and_lemmatize(text):
    if isinstance(text, str):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        text = text.lower()
        tokens = nltk.tokenize.word_tokenize(text)
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)
    else:
        return text

def prepare_target(df3, df2):
    # Ensure 'Genre' column exists in df2
    if 'genre' not in df2.columns:
        raise KeyError("The 'Genre' column is not found in df2. Please check the DataFrame structure.")

    genres = df2['genre'].unique()
    target = np.zeros((df3.shape[0], len(genres)))
    categories_forward_lookup = {genre: i for i, genre in enumerate(genres)}
    for i, cs in zip(df3.index, df3.Genre):
        for c in cs:
            category_number = categories_forward_lookup[c]
            target[i, category_number] = 1
    return target, categories_forward_lookup

def train_model(X_train, y_train, X_validation, y_validation):
    max_tokens = 50000
    output_sequence_length = 100
    embedding_dim = 32

    vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length)
    vectorizer.adapt(X_train)

    inputs = Input(shape=(1,), dtype=tf.string)
    vectorized = vectorizer(inputs)
    embedded = Embedding(max_tokens + 1, embedding_dim)(vectorized)
    averaged = GlobalAveragePooling1D()(embedded)
    layer1 = Dense(128, activation='relu')(averaged)
    output = Dense(len(categories_forward_lookup), activation='sigmoid')(layer1)

    model = Model(inputs=[inputs], outputs=[output])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=200, validation_data=(X_validation, y_validation), callbacks=[early_stopping])
    return model

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    return accuracy

def save_model(model, lookup_dict):
    model.save('model')
    with open('categories_forward_lookup.pkl', 'wb') as f:
        pickle.dump(lookup_dict, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classification model from SQLite data")
    parser.add_argument("--training-data", required=True, help="Path to the SQLite database file")
    args = parser.parse_args()

    df, df2, df3 = load_data(args.training_data)
    df3 = clean_data(df3)
    target, categories_forward_lookup = prepare_target(df3, df2)

    X_train, X_test, y_train, y_test = train_test_split(df3['Description'], target, test_size=0.2, random_state=142)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=142)

    model = train_model(X_train, y_train, X_validation, y_validation)
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"Model training complete. Accuracy: {accuracy:.2f}")

    save_model(model, categories_forward_lookup)

