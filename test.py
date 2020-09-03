# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 20:26:56 2020

@author: Shubham
"""

#import csv
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu', input_dim=96568))
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

message = input('Enter the message : ')
#message = 'Rajput made his film debut in the buddy drama Kai Po Che! (2013), for which he received a nomination for the Filmfare Award for Best Male Debut. He then starred in the romantic comedy Shuddh Desi Romance (2013) and as the titular detective in the action thriller Detective Byomkesh Bakshy! (2015). His highest-grossing releases came with a supporting role in the satire PK (2014), followed by the titular role in the sports biopic M.S. Dhoni: The Untold Story (2016). For his performance in the latter, he received his first nomination for the Filmfare Award for Best Actor.[7][8] Rajput went on to star in the commercially successful films Kedarnath (2018) and Chhichhore (2019).[9][10] His last film, Dil Bechara (2020), was released posthumously on Hotstar.'
mwords_list = message_preprocessing(str(message))
message_predict = ann.predict(mwords_list)
message_predict = 1 if message_predict > 0.5 else 0

if(message_predict == 1):
    print('\n\n\nSpam')
else:
    print('\n\n\nGood to read')

#def add_message_to_dataset(message, prediction):
#    with open('fraud_email_.csv', 'r', encoding="utf8") as readFile:
#        reader = csv.reader(readFile)
#        lines = list(reader)
#        lines.append([str(message), str(prediction)])
#    with open('fraud_email_.csv', 'w', encoding="utf8", newline='') as writeFile:
#        writer = csv.writer(writeFile)
#        for x in lines:
#            writer.writerow(x)
#
#add_message_to_dataset(message, message_predict)