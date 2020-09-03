# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:26:56 2020

@author: Shubham
"""

import csv
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

#print(tf.__version__)
dataset = pd.read_csv('fraud_email_.csv')
dataset = dataset.dropna()
dataset.head(20)
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, -1].values

stopwordsList = stopwords.words("english")

vectorizer = CountVectorizer(stop_words = stopwordsList)
X = vectorizer.fit_transform(X).toarray()
input_dimensions = len(X[1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_dim=int(input_dimensions)))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
ann.fit(X_train, y_train, batch_size = 32, epochs = 10)

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)

cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred) * 100
print("\n\n\nAccuracy of model : %.2f" % acc_score)

def message_preprocessing(message):
    message = message.lower()
    message_words = word_tokenize(message)
    message_words= [word for word in message_words if word.isalnum()]
    message_wordsList = []
    for word in message_words:
        if word.lower() not in stopwordsList:
            message_wordsList.append(word)
    
    wordsList = vectorizer.get_feature_names()
    message_list = [0] * len(wordsList)
    
    for x in message_wordsList:
        for y in wordsList:
            if(x == y):
                ind = wordsList.index(x)
                message_list[ind] = int(message_list[ind]) + 1
    return [message_list]

def add_message_to_dataset(message, prediction):
    with open('fraud_email_.csv', 'a+', encoding="utf8", newline='') as writeFile:
        csv_writer = csv.writer(writeFile)
        csv_writer.writerow([message, prediction])

message = input('Enter the message : ')
mwords_list = message_preprocessing(str(message))
message_predict = ann.predict(mwords_list)
message_predict = 1 if message_predict > 0.5 else 0

if(message_predict == 1):
    print('\n\n\nSpam')
else:
    print('\n\n\nGood to read')

#add_message_to_dataset(message, message_predict)